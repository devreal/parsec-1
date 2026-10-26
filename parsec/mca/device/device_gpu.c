/*
 * Copyright (c) 2021-2026 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024-2026 NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      Stony Brook University. All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/device_gpu.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/argv.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"

#include <limits.h>
#include <threads.h>

#define PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL 1024
#define PARSEC_DEVICE_CLEAR_EVERY 128

#if defined(PARSEC_PROF_TRACE)
static int parsec_gpu_movein_key_start;
static int parsec_gpu_movein_key_end;
static int parsec_gpu_moveout_key_start;
static int parsec_gpu_moveout_key_end;
static int parsec_gpu_own_GPU_key_start;
static int parsec_gpu_own_GPU_key_end;
static int parsec_gpu_allocate_memory_key;
static int parsec_gpu_free_memory_key;
static int parsec_gpu_use_memory_key_start;
static int parsec_gpu_use_memory_key_end;
static int parsec_gpu_prefetch_key_start;
static int parsec_gpu_prefetch_key_end;
static int parsec_gpu_profiling_initiated = 0;
#endif  /* defined(PROFILING) */
int parsec_gpu_output_stream = -1;
int parsec_gpu_verbosity;
int32_t parsec_gpu_exec_delay_batch = 1;

static thread_local int active_w2r_tasks = 0;


/**
 * @brief Prepare a task for execution on the GPU. Basically, does some upstream initialization,
 * setup the profiling information and then calls directly into the task submission body. Upon
 * return from the body handle the state machine of the task, taking care of the special cases
 * such as AGAIN and ASYNC.
 * @returns An error if anything unexpected came out of the task submission body, otherwise
 */
static int
parsec_device_kernel_exec( parsec_device_gpu_module_t      *gpu_device,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream);

/* The return value of these functions is either a parsec_hook_return_t for <= 0 values,
 * or a positive number which represents that something has been scheduled on the gpu_stream.
 * Step functions used by parsec_device_progress_stream() with batching enabled receive a
 * ring of tasks instead, and store this value for each task of the ring in its last_status.
 */
typedef int(*parsec_gpu_step_function_t)(parsec_device_gpu_module_t  *gpu_device,
                                         parsec_gpu_task_t           *gpu_task,
                                         parsec_gpu_exec_stream_t    *gpu_stream);
static inline int
parsec_device_check_space_needed(parsec_device_gpu_module_t *gpu_device,
                                 parsec_gpu_task_t *gpu_task)
{
    int i;
    int space_needed = 0;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_data_t *original;
    parsec_data_copy_t *data;
    const parsec_flow_t *flow;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        data = this_task->data[i].data_in;
        if (data == NULL) continue;

        original = data->original;
        if( NULL != PARSEC_DATA_GET_COPY(original, gpu_device->super.device_index) ) {
            continue;
        }
        space_needed++;
    }
    return space_needed;
}

#if defined(PARSEC_PROF_TRACE)
void parsec_device_init_profiling(void)
{
    if(parsec_gpu_profiling_initiated == 0) {
        parsec_profiling_add_dictionary_keyword("gpu", "fill:#66ff66",
                                                0, NULL,
                                                &parsec_gpu_own_GPU_key_start, &parsec_gpu_own_GPU_key_end);
        parsec_profiling_add_dictionary_keyword("movein", "fill:#33FF33",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_movein_key_start, &parsec_gpu_movein_key_end);
        parsec_profiling_add_dictionary_keyword("moveout", "fill:#ffff66",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_moveout_key_start, &parsec_gpu_moveout_key_end);
        parsec_profiling_add_dictionary_keyword("prefetch", "fill:#66ff66",
                                                sizeof(parsec_profile_data_collection_info_t),
                                                PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                &parsec_gpu_prefetch_key_start, &parsec_gpu_prefetch_key_end);
        parsec_profiling_add_dictionary_keyword("gpu_mem_alloc", "fill:#FF66FF",
#if (PARSEC_SIZEOF_SIZE_T == 4)
                                                sizeof(uint32_t), "size{uint32_t}",
#elif (PARSEC_SIZEOF_SIZE_T == 8)
                                                sizeof(uint64_t), "size{uint64_t}",
#else
#error "Unsupported case: sizeof(size_t) is neither 8 nor 4"
#endif // PARSEC_SIZEOF_SIZE_T
                                                &parsec_gpu_allocate_memory_key, &parsec_gpu_free_memory_key);
        parsec_profiling_add_dictionary_keyword("gpu_mem_use", "fill:#FF66FF",
                                                sizeof(parsec_device_gpu_memory_prof_info_t),
                                                PARSEC_DEVICE_GPU_MEMORY_PROF_INFO_CONVERTER,
                                                &parsec_gpu_use_memory_key_start, &parsec_gpu_use_memory_key_end);
        parsec_gpu_profiling_initiated = 1;
    }
}
#endif

void parsec_device_enable_debug(void)
{
    if(parsec_gpu_output_stream == -1) {
        parsec_gpu_output_stream = parsec_device_output;
        if( parsec_gpu_verbosity >= 0 ) {
            parsec_gpu_output_stream = parsec_output_open(NULL);
            parsec_output_set_verbosity(parsec_gpu_output_stream, parsec_gpu_verbosity);
        }
    }
}


void* parsec_device_pop_workspace(parsec_device_gpu_module_t* gpu_device,
                                  parsec_gpu_exec_stream_t* gpu_stream, size_t size)
{
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (parsec_gpu_workspace_t *)malloc(sizeof(parsec_gpu_workspace_t));
        gpu_stream->workspace->total_workspace = PARSEC_GPU_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = PARSEC_GPU_MAX_WORKSPACE - 1;

        for( int i = 0; i < PARSEC_GPU_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                 "GPU[%d:%s] Succeeded Allocating workspace %d (device_ptr %p)",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 i, gpu_stream->workspace->workspace[i]);
#if defined(PARSEC_PROF_TRACE)
            if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
               (gpu_device->exec_stream[0]->prof_event_track_enable ||
                gpu_device->exec_stream[1]->prof_event_track_enable)) {
                parsec_profiling_trace_flags(gpu_stream->profiling,
                                             parsec_gpu_allocate_memory_key, (int64_t)
                                             gpu_stream->workspace->workspace[i], gpu_device->super.device_index,
                                             &size, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
            }
#endif
        }
    }
    if (gpu_stream->workspace->stack_head < 0) {
        parsec_fatal("parsec_device_pop_workspace: user requested more than %d GPU workspaces which is the current hard-coded limit per GPU stream\n", PARSEC_GPU_MAX_WORKSPACE);
        return NULL;
    }
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return work;
}

int parsec_device_push_workspace(parsec_device_gpu_module_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_GPU_MAX_WORKSPACE);
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return 0;
}

int parsec_device_free_workspace(parsec_device_gpu_module_t * gpu_device)
{
    (void)gpu_device;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *gpu_stream = gpu_device->exec_stream[i];
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
#if defined(PARSEC_PROF_TRACE)
                if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
                   (gpu_device->exec_stream[0]->prof_event_track_enable ||
                    gpu_device->exec_stream[1]->prof_event_track_enable)) {
                    parsec_profiling_trace_flags(gpu_stream->profiling,
                                                 parsec_gpu_allocate_memory_key, (int64_t)
                                                 gpu_stream->workspace->workspace[i], gpu_device->super.device_index,
                                                 NULL, PARSEC_PROFILING_EVENT_COUNTER);
                }
#endif
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s] Release workspace %d (device_ptr %p)",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     j, gpu_stream->workspace->workspace[j]);
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */
    return 0;
}

#if defined(PARSEC_DEBUG_NOISIER)
char *parsec_device_describe_gpu_task( char *tmp, size_t len, parsec_gpu_task_t *gpu_task )
{
    char buffer[64];
    parsec_data_t *data;
    switch( gpu_task->task_type ) {
        case PARSEC_GPU_TASK_TYPE_KERNEL:
            return parsec_task_snprintf(tmp, len, gpu_task->ec);
        case PARSEC_GPU_TASK_TYPE_PREFETCH:
            assert(NULL != gpu_task->ec);
            assert(NULL != gpu_task->ec->data[0].data_in );
            data = gpu_task->ec->data[0].data_in->original;
            if( NULL == data || NULL == data->dc )
                snprintf(tmp, len, "PREFETCH for unbound data %p", data);
            else {
                data->dc->key_to_string(data->dc, data->key, buffer, 64);
                snprintf(tmp, len, "PREFETCH for %s (data %p)", buffer, data);
            }
            return tmp;
        case PARSEC_GPU_TASK_TYPE_WARMUP:
            assert(NULL != gpu_task->copy->original && NULL != gpu_task->copy->original->dc);
            gpu_task->copy->original->dc->key_to_string(gpu_task->copy->original->dc, gpu_task->copy->original->key, buffer, 64);
            snprintf(tmp, len, "WARMUP %s on device %d",
                     buffer, gpu_task->copy->device_index);
            return tmp;
        case PARSEC_GPU_TASK_TYPE_D2HTRANSFER:
            snprintf(tmp, len, "Device to Host Transfer");
            return tmp;
        case PARSEC_GPU_TASK_TYPE_D2D_COMPLETE:
            snprintf(tmp, len, "D2D Transfer Complete for data copy %p [ref_count %d]",
                     gpu_task->ec->data[0].data_out, gpu_task->ec->data[0].data_out->super.super.obj_reference_count);
            return tmp;
        default:
            snprintf(tmp, len, "*** Internal Error: unknown gpu task type %d ***", gpu_task->task_type);
            return tmp;
    }
}
#endif

void parsec_device_dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    parsec_debug_verbose(0, parsec_gpu_output_stream,
                         "Dev: GPU stream %d{%p} [events = %d, start = %d, end = %d, executed = %d]",
                         exec_stream->name, exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
                         exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        parsec_debug_verbose(0, parsec_gpu_output_stream,
                             "    %d: %s", i, parsec_task_snprintf(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void parsec_device_dump_gpu_state(parsec_device_gpu_module_t* gpu_device)
{
    int i;
    uint64_t data_in_host, data_in_dev = 0;

    data_in_host = gpu_device->super.data_in_from_device[0];
    for(int i = 1; i < gpu_device->super.data_in_array_size; i++) {
        data_in_dev += gpu_device->super.data_in_from_device[i];
    }

    parsec_output(parsec_gpu_output_stream, "\n\n");
    parsec_output(parsec_gpu_output_stream, "Device %d:%d (%p) epoch\n", gpu_device->super.device_index,
                  gpu_device->super.device_index, gpu_device, gpu_device->data_avail_epoch);
    parsec_output(parsec_gpu_output_stream, "\tpeer mask %x executed tasks with %llu streams %d\n",
                  gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->num_exec_streams);
    parsec_output(parsec_gpu_output_stream, "\tstats transferred [in: %llu from host %llu from other device out: %llu] required [in: %llu out: %llu]\n",
                  (unsigned long long)data_in_host, (unsigned long long)data_in_dev,
                  (unsigned long long)gpu_device->super.data_out_to_host,
                  (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->num_exec_streams; i++ ) {
        parsec_device_dump_exec_stream(gpu_device->exec_stream[i]);
    }
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_lru) ) {
        parsec_output(parsec_gpu_output_stream, "#\n# LRU list\n#\n");
        i = 0;
        PARSEC_LIST_ITERATOR(&gpu_device->gpu_mem_lru, item,
                             {
                                 parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                 parsec_output(parsec_gpu_output_stream, "  %d. elem %p flags 0x%x GPU mem %p\n",
                                               i, gpu_copy, gpu_copy->flags, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    }
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        parsec_output(parsec_gpu_output_stream, "#\n# Owned LRU list\n#\n");
        i = 0;
        PARSEC_LIST_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                             {
                                 parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                 parsec_output(parsec_gpu_output_stream, "  %d. elem %p flags 0x%x GPU mem %p\n",
                                               i, gpu_copy, gpu_copy->flags, gpu_copy->device_private);
                                 parsec_dump_data_copy(gpu_copy);
                                 i++;
                             });
    }
    parsec_output(parsec_gpu_output_stream, "\n\n");
}


static parsec_flow_t parsec_device_data_prefetch_flow = {
    .name = "PREFETCH FLOW",
    .flow_flags = PARSEC_FLOW_ACCESS_READ,
    .flow_index = 0,
};

static parsec_task_class_t parsec_device_data_prefetch_tc = {
    .name = "DEVICE PREFETCH",
    .flags = 0,
    .task_class_id = 0,
    .nb_flows = 1,
    .nb_parameters = 0,
    .nb_locals = 0,
    .dependencies_goal = 0,
    .params = { NULL, },
    .in = { &parsec_device_data_prefetch_flow, NULL },
    .out = { NULL, },
    .priority = NULL,
    .properties = NULL,
    .initial_data = NULL,
    .final_data = NULL,
    .data_affinity = NULL,
    .key_functions = NULL,
    .make_key = NULL,
    .get_datatype = NULL,
    .prepare_input = NULL,
    .incarnations = NULL,
    .prepare_output = NULL,
    .find_deps = NULL,
    .iterate_successors = NULL,
    .iterate_predecessors = NULL,
    .release_deps = NULL,
    .complete_execution = NULL,
    .new_task = NULL,
    .release_task = NULL,
    .fini = NULL
};

/**
 * Release a gpu copy to the zone allocator.
 */
void parsec_device_release_gpu_copy(parsec_device_gpu_module_t* gpu_device, parsec_data_copy_t *gpu_elem)
{
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
#if defined(PARSEC_PROF_TRACE)
    if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
        (gpu_device->exec_stream[0]->prof_event_track_enable ||
        gpu_device->exec_stream[1]->prof_event_track_enable)) {
        parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                     parsec_gpu_free_memory_key, (int64_t)gpu_elem->device_private,
                                     gpu_device->super.device_index,
                                     NULL, PARSEC_PROFILING_EVENT_COUNTER);
        parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                     parsec_gpu_use_memory_key_end,
                                     (uint64_t)gpu_elem->device_private,
                                     gpu_device->super.device_index, NULL, 0);
    }
#endif // PARSEC_PROF_TRACE
    if (0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)) {
        //assert( 0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        //assert(gpu_elem->device_private != NULL);
        parsec_data_t* data = gpu_elem->original;
        if (data != NULL) {
            /* have to take the lock to protect against double frees */
            parsec_atomic_lock(&data->lock);
        }
        if (gpu_elem->device_private != NULL) {
            zone_free( gpu_device->memory, (void*)(gpu_elem->device_private) );
            gpu_elem->device_private = NULL;
            gpu_device->data_avail_epoch++;
        }
        if (data != NULL) {
            /* release the lock */
            parsec_atomic_unlock(&data->lock);
        }
    }
    PARSEC_OBJ_RELEASE(gpu_elem);
    assert( NULL == gpu_elem );
#endif // PARSEC_GPU_ALLOC_PER_TILE
}

static int
parsec_device_release_resources_prefetch_task(parsec_device_gpu_module_t* gpu_device,
                        parsec_gpu_task_t** out_task)
{
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    parsec_gpu_task_t *gpu_task = *out_task;
    (void)gpu_device;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]: Releasing resources for task %s (%p with ec %p)",
                         gpu_device->super.device_index, gpu_device->super.name, parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task),
                         gpu_task, gpu_task->ec);
    assert( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type );
    PARSEC_DATA_COPY_RELEASE( gpu_task->ec->data[0].data_in);
    free( gpu_task->ec );
    gpu_task->ec = NULL;
    return 0;
}

#if defined(PARSEC_DEBUG_NOISIER)
static char *parsec_device_debug_advice_to_string(int advice)
{
    switch(advice) {
    case PARSEC_DEV_DATA_ADVICE_PREFETCH:
        return "Prefetch";
    case PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE:
        return "Set Preferred Device";
    case PARSEC_DEV_DATA_ADVICE_WARMUP:
        return "Mark data as recently used";
    default:
        assert(0);
        return "Undefined advice";
    }
}
#endif

int
parsec_device_data_advise(parsec_device_module_t *dev, parsec_data_t *data, int advice)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)dev;
#if defined(PARSEC_DEBUG_NOISIER)
    char buffer[64];
    if(NULL != data->dc) {
        data->dc->key_to_string(data->dc, data->key, buffer, 64);
    } else {
        snprintf(buffer, 64, "unbound data");
    }
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]: User provides advice %s of %s (%p)",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_device_debug_advice_to_string(advice),
                         buffer,
                         data);

    switch(advice) {
    case PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE:
        data->preferred_device = dev->device_index;
        break;
    case PARSEC_DEV_DATA_ADVICE_PREFETCH:
        {
            if( parsec_type_contiguous(data->device_copies[ data->owner_device ]->dtt) != PARSEC_SUCCESS){
                parsec_warning( "GPU[%d:%s]: PARSEC_DEV_DATA_ADVICE_PREFETCH cannot be applied to non contiguous types @%s:%d",
                                gpu_device->super.device_index, gpu_device->super.name, __func__, __LINE__);
                return PARSEC_ERROR;
            }
            parsec_gpu_task_t* gpu_task = NULL;
            gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
            gpu_task->task_type = PARSEC_GPU_TASK_TYPE_PREFETCH;
            gpu_task->release_device_task = free;  /* by default free the device task */
            gpu_task->ec = calloc(1, sizeof(parsec_task_t));
            PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
            gpu_task->ec->task_class = &parsec_device_data_prefetch_tc;
            /* The pending heap compares gpu_task->priority, not gpu_task->ec->priority:
             * priority inheritance from ec normally happens in
             * parsec_device_kernel_scheduler(), which this direct-enqueue path bypasses.
             * Set it explicitly so this task is ranked the same as its ec priority
             * instead of relying on the calloc()-zeroed default. */
            gpu_task->priority = gpu_task->ec->priority;
            gpu_task->flow[0] = &parsec_device_data_prefetch_flow;
            gpu_task->flow_nb_elts[0] = data->device_copies[ data->owner_device ]->original->nb_elts;
            gpu_task->stage_in  = parsec_default_gpu_stage_in;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Retain data copy %p [ref_count %d]",
                                 data->device_copies[ data->owner_device ],
                                 data->device_copies[ data->owner_device ]->super.super.obj_reference_count);
            PARSEC_OBJ_RETAIN(data->device_copies[ data->owner_device ]);
            gpu_task->ec->data[0].data_in = data->device_copies[ data->owner_device ];
            gpu_task->ec->data[0].data_out = NULL;
            gpu_task->ec->data[0].source_repo_entry = NULL;
            gpu_task->ec->data[0].source_repo = NULL;
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: data copy %p [ref_count %d] linked to prefetch gpu task %p on GPU copy %p [ref_count %d]",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_task->ec->data[0].data_in, gpu_task->ec->data[0].data_in->super.super.obj_reference_count,
                                 gpu_task, gpu_task->ec->data[0].data_out);
            parsec_lifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
            return PARSEC_SUCCESS;
        }
        break;
    case PARSEC_DEV_DATA_ADVICE_WARMUP:
        return PARSEC_ERR_NOT_IMPLEMENTED;
        break;
    default:
        assert(0);
        return PARSEC_ERR_NOT_FOUND;
    }
    return PARSEC_SUCCESS;
}

/**
 * Register a taskpool with a device by checking that the device
 * supports the dynamic function required by the different incarnations.
 * If multiple devices of the same type exists we assume that all have
 * the same capabilities.
 */
int
parsec_device_taskpool_register(parsec_device_module_t* device,
                                parsec_taskpool_t* tp)
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t*)device;
    int32_t rc = PARSEC_ERR_NOT_FOUND;
    uint32_t i, j;

    /**
     * Detect if a particular chore has a dynamic load dependency and if yes
     * load the corresponding module and find the function.
     */
    assert(PARSEC_DEV_IS_GPU(device->type));
    assert(tp->devices_index_mask & (1 << device->device_index));

    for( i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( j = 0; NULL != chores[j].hook; j++ ) {
            if( !(chores[j].type & device->type) )
                continue;
            if( NULL != chores[j].dyld_fn ) {
                /* the function has been set for another device of the same type */
                return PARSEC_SUCCESS;
            }
            if ( NULL == chores[j].dyld ) {
                chores[j].dyld_fn = NULL;  /* No dynamic support required for this kernel */
                rc = PARSEC_SUCCESS;
            } else {
                void* devf = gpu_device->find_incarnation(gpu_device, chores[j].dyld);
                if( NULL != devf ) {
                    chores[j].dyld_fn = devf;
                    rc = PARSEC_SUCCESS;
                }
            }
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        tp->devices_index_mask &= ~(1 << device->device_index);  /* drop support for this device */
        parsec_debug_verbose(10, parsec_gpu_output_stream,
                             "Device %d:%s disabled for taskpool %d:%s (%p)", device->device_index, device->name,
                             tp->taskpool_id, tp->taskpool_name, tp);
    }
    return rc;
}

int
parsec_device_taskpool_unregister(parsec_device_module_t* device, parsec_taskpool_t* tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

/**
 * Attach a device to a PaRSEC context. A device can only be attached to
 * a single context at the time.
 */
int
parsec_device_attach( parsec_device_module_t* device, parsec_context_t* context )
{
    return parsec_mca_device_add(context, device);
}

/**
 * Detach a device from a context. Both the context and the device remain
 * valid, they are simply disconnected.
 * This function should only be called once all tasks and all data related to the
 * context has been removed from the device.
 */
int
parsec_device_detach( parsec_device_module_t* device, parsec_context_t* context )
{
    (void)context;
    return parsec_mca_device_remove(device);
}

/**
 * This function reserve the memory_percentage of the total device memory for PaRSEC.
 * This memory will be managed in chunks of size eltsize. However, multiple chunks
 * can be reserved in a single allocation.
 */
int
parsec_device_memory_reserve( parsec_device_gpu_module_t* gpu_device,
                              int           memory_percentage,
                              int           number_blocks,
                              size_t        eltsize )
{
    int rc;

    size_t alloc_size;
    size_t total_mem, initial_free_mem;
    size_t mem_elem_per_gpu = 0;

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return rc;

    /* Determine how much memory we can allocate */
    rc = gpu_device->memory_info( gpu_device, &initial_free_mem, &total_mem );
    if(PARSEC_SUCCESS != rc)
        return rc;

    if( number_blocks != -1 ) {
        if( number_blocks == 0 ) {
            parsec_warning("GPU[%d:%s] Invalid argument: requesting 0 bytes of memory",
                           gpu_device->super.device_index, gpu_device->super.name);
            return PARSEC_ERROR;
        } else {
            alloc_size = number_blocks * eltsize;
        }
    } else {
        /* number_blocks == -1 means memory_percentage is used */
        alloc_size = (memory_percentage * initial_free_mem) / 100;
        /* round-up in eltsize */
        alloc_size = eltsize * ((alloc_size + eltsize - 1 ) / eltsize);
    }
    if( alloc_size >= initial_free_mem ) {
        /* Mapping more than 100% of GPU memory is obviously wrong
         * Mapping exactly 100% of the GPU memory ends up producing errors about __global__ function call is not configured
         * Mapping 95% works with low-end GPUs like 1060, how much to let available for gpu runtime, I don't know how to calculate */
        parsec_warning("GPU[%d:%s] Requested %zd bytes on GPU device, but only %zd bytes are available -- reducing allocation to 95%% of max available",
                       gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem);
        alloc_size = (95 * initial_free_mem) / 100;
        /* round-up in eltsize */
        alloc_size = eltsize * ((alloc_size + eltsize - 1 ) / eltsize);
    }
    if( alloc_size < eltsize ) {
        /* Handle another kind of jokers entirely, and cases of
         * not enough memory on the device */
        parsec_warning("GPU[%d:%s] Cannot allocate at least one element",
                       gpu_device->super.device_index, gpu_device->super.name);
        return PARSEC_ERROR;
    }

#if defined(PARSEC_GPU_ALLOC_PER_TILE)
    size_t free_mem = initial_free_mem;
    /*
     * We allocate a bunch of tiles that will be used
     * during the computations
     */
    while( (free_mem > eltsize )
           && ((total_mem - free_mem) < alloc_size) ) {
        parsec_gpu_data_copy_t* gpu_elem;
        void *device_ptr;

        rc = gpu_device->memory_allocate(gpu_device, eltsize, &device_ptr);
        if(PARSEC_SUCCESS != rc) {
            size_t _free_mem, _total_mem;
            gpu_device->memory_info(gpu_device, &_free_mem, &_total_mem );
            parsec_inform("GPU[%d:%s] Per context: free mem %zu total mem %zu (allocated tiles %zu)",
                          gpu_device->super.device_index, gpu_device->super.name,_free_mem, _total_mem, mem_elem_per_gpu);
            break;
        }
        gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Allocate GPU copy %p [ref_count %d] for data [%p]",
                            gpu_device->super.device_index, gpu_device->super.name,gpu_elem, gpu_elem->super.obj_reference_count, NULL);
        gpu_elem->device_private = (void*)(long)device_ptr;
        gpu_elem->flags |= PARSEC_DATA_FLAG_PARSEC_OWNED;
        gpu_elem->device_index = gpu_device->super.device_index;
        mem_elem_per_gpu++;
        PARSEC_OBJ_RETAIN(gpu_elem);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Retain and insert GPU copy %p [ref_count %d] in LRU",
                            gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.obj_reference_count);
        parsec_list_nolock_push_back( &gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem );
        gpu_device->memory_info( gpu_device, &free_mem, &total_mem );
    }
    if( 0 == mem_elem_per_gpu && parsec_list_nolock_is_empty( &gpu_device->gpu_mem_lru ) ) {
        parsec_warning("GPU[%d:%s] Cannot allocate memory on GPU %s. Skip it!", gpu_device->super.device_index, gpu_device->super.name, gpu_device->super.name);
    }
    else {
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s] Allocate %zu tiles on the GPU memory",
                             gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu );
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                         "GPU[%d:%s] Allocate %zu tiles on the GPU memory", gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu);
#else
    if( NULL == gpu_device->memory ) {
        void* base_ptr;

        rc = gpu_device->memory_allocate(gpu_device, alloc_size, &base_ptr);
        if(PARSEC_SUCCESS != rc) {
            parsec_warning("GPU[%d:%s] Allocating %zu bytes of memory on the GPU device failed (initial_free_mem was %zu)",
                           gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem);
            gpu_device->memory = NULL;
            return PARSEC_ERROR;
        }

        assert(alloc_size % eltsize == 0); /* we rounded up earlier... */
        mem_elem_per_gpu = alloc_size / eltsize;
        gpu_device->memory = zone_malloc_init( base_ptr, mem_elem_per_gpu, eltsize );
        if( gpu_device->memory == NULL ) {
            parsec_warning("GPU[%d:%s] Failed trying to allocate %zu bytes. We tried to do so based on an initial_free_mem of %zu bytes and elt_size of %zu bytes",
                           gpu_device->super.device_index, gpu_device->super.name, alloc_size, initial_free_mem, eltsize);
            return PARSEC_ERROR;
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                            "GPU[%d:%s] Allocate %zu segments of size %zu on the GPU memory",
                            gpu_device->super.device_index, gpu_device->super.name, mem_elem_per_gpu, eltsize );
    }
#endif
    gpu_device->mem_block_size = eltsize;
    gpu_device->mem_nb_blocks = mem_elem_per_gpu;

    return PARSEC_SUCCESS;
}

/**
 * Release discarded data copies from the LRU list.
 * Returns the number of discarded items released.
 */
static int parsec_device_memory_release_discarded(parsec_device_gpu_module_t* gpu_device,
                                                  parsec_list_t* list)
{
    parsec_list_item_t* item;
    int count = 0;

    if (gpu_device->super.nb_discarded == 0) {
        return 0;
    }

    /* Items are unlinked while iterating, so remember the successor before an
     * item is removed and released. An item must be removed from the list before
     * it is made a singleton (which resets its own links, not its neighbors') and
     * before it is released (which may free it). */
    for (item = PARSEC_LIST_ITERATOR_BEGIN(list);
         item != PARSEC_LIST_ITERATOR_END(list); ) {
        parsec_list_item_t* next = (parsec_list_item_t*)PARSEC_LIST_ITEM_NEXT(item);
        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
        parsec_data_t* original = gpu_copy->original;
        if (NULL != original) {
            parsec_data_copy_t *cpu_copy = original->device_copies[0];
            if (NULL != cpu_copy && (cpu_copy->flags & PARSEC_DATA_FLAG_DISCARDED) && cpu_copy->readers == 0) {
                count++;
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                    "Releasing discarded GPU copy %p from data %p", gpu_copy, original);
                parsec_list_nolock_remove(list, item);
                PARSEC_LIST_ITEM_SINGLETON(item);
                parsec_device_release_gpu_copy(gpu_device, gpu_copy);
            }
        }
        item = next;
    }
    parsec_atomic_fetch_sub_int64(&gpu_device->super.nb_discarded, count);
    return count;
}

static void parsec_device_memory_release_list(parsec_device_gpu_module_t* gpu_device,
                                              parsec_list_t* list)
{
    parsec_list_item_t* item;
    int num_discarded = 0;

    while(NULL != (item = parsec_list_nolock_pop_front(list)) ) {
        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
        parsec_data_t* original = gpu_copy->original;

        if (NULL != original) {
            parsec_data_copy_t *cpu_copy = original->device_copies[0];
            PARSEC_LIST_ITEM_SINGLETON(item);

            if (cpu_copy->flags & PARSEC_DATA_FLAG_DISCARDED) {
                num_discarded++;
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                    "Releasing discarded GPU copy %p from data %p", gpu_copy, original);
                parsec_device_release_gpu_copy(gpu_device, gpu_copy);
                continue;
            }
        }

        PARSEC_DEBUG_VERBOSE(35, parsec_gpu_output_stream,
                            "GPU[%d:%s] Release GPU copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p, in map %p",
                            gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                             original, (NULL != original ? original->dc : NULL));
        assert( gpu_copy->device_index == gpu_device->super.device_index );
        /* warn about device data that has not been pushed back to the host or was discarded */
        if( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state) {
            parsec_warning("GPU[%d:%s] still OWNS the master memory copy for data %d and it is discarding it!",
                           gpu_device->super.device_index, gpu_device->super.name, original->key);
        }
        assert(0 != (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );

#if defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_device->memory_free( gpu_copy->device_private );
        gpu_copy->device_private = NULL;

        /* At this point the data copies should have no attachment to a data_t. Thus,
         * before we get here (aka below parsec_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        PARSEC_OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
#else
        parsec_device_release_gpu_copy(gpu_device, gpu_copy);
#endif
    }
    parsec_atomic_fetch_sub_int64(&gpu_device->super.nb_discarded, num_discarded);
}

/**
 * This function only flushes the data copies pending in LRU, and checks
 * (in debug mode) that the entire allocated memory is free to use */
int
parsec_device_flush_lru( parsec_device_module_t *device )
{
    size_t in_use;
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)device;
    /* Free all remaining memory on GPU */
    parsec_device_memory_release_list(gpu_device, &gpu_device->gpu_mem_lru);
    parsec_device_memory_release_list(gpu_device, &gpu_device->gpu_mem_owned_lru);
    assert(gpu_device->super.nb_discarded == 0);
    parsec_device_free_workspace(gpu_device);
#if !defined(PARSEC_GPU_ALLOC_PER_TILE) && !defined(_NDEBUG)
    if( (in_use = zone_in_use(gpu_device->memory)) != 0 ) {
        parsec_warning("GPU[%d:%s] memory leak detected: %lu bytes still allocated on GPU",
                       device->device_index, device->name, in_use);
        zone_debug(gpu_device->memory, 0, parsec_gpu_output_stream, "flush_lru: ");
        assert(!in_use);
    }
#endif
    return PARSEC_SUCCESS;
}

/**
 * This function release the GPU memory reserved for this device.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
int
parsec_device_memory_release( parsec_device_gpu_module_t* gpu_device )
{
    int rc;

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return rc;

    parsec_device_flush_lru(&gpu_device->super);

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    assert( NULL != gpu_device->memory );
    void* ptr = zone_malloc_fini(&gpu_device->memory);
    rc = gpu_device->memory_free(gpu_device, ptr);
    if(PARSEC_SUCCESS != rc) {
        parsec_warning("GPU[%d:%s]: Failed to free the GPU backend memory.",
                       gpu_device->super.device_index, gpu_device->super.name);
        return rc;
    }
#endif

    return PARSEC_SUCCESS;
}

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
/**
 * Allocate, with a single zone_malloc_batch() call, the device memory of all
 * flows of gpu_task that do not have a copy on the device yet. The resulting
 * pointers are stored in prealloc[flow index] (NULL for flows that need no new
 * copy or could not be allocated). The allocation is best effort: flows left
 * without memory go through the regular allocation and eviction path in
 * parsec_device_data_reserve_space().
 */
static void
parsec_device_data_prealloc_space( parsec_device_gpu_module_t* gpu_device,
                                   parsec_gpu_task_t *gpu_task,
                                   void **prealloc )
{
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t *flow;
    parsec_data_t *master;
    void *ptrs[MAX_PARAM_COUNT];
    size_t sizes[MAX_PARAM_COUNT];
    int flows[MAX_PARAM_COUNT];
    int i, nb_items = 0, nb_allocated;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        prealloc[i] = NULL;
        flow = gpu_task->flow[i];
        /* Skip CTL flows only */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if( NULL == this_task->data[i].data_in ) continue;
        if( 0 == gpu_task->flow_nb_elts[i] ) continue;
        master = this_task->data[i].data_in->original;
        parsec_atomic_lock(&master->lock);
        if( NULL == PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index) ) {
            flows[nb_items] = i;
            sizes[nb_items] = gpu_task->flow_nb_elts[i];
            nb_items++;
        }
        parsec_atomic_unlock(&master->lock);
    }
    if( 0 == nb_items ) return;

    nb_allocated = zone_malloc_batch(gpu_device->memory, sizes, nb_items, ptrs);
    for( i = 0; i < nb_allocated; i++ ) {
        prealloc[flows[i]] = ptrs[i];
    }
}

/**
 * Release the preallocated device memory that has not been used by
 * parsec_device_data_reserve_space().
 */
static void
parsec_device_data_prealloc_release( parsec_device_gpu_module_t* gpu_device,
                                     void **prealloc, int nb_flows )
{
    void *ptrs[MAX_PARAM_COUNT];
    int i, nb_items = 0;

    for( i = 0; i < nb_flows; i++ ) {
        if( NULL == prealloc[i] ) continue;
        ptrs[nb_items++] = prealloc[i];
        prealloc[i] = NULL;
    }
    if( nb_items > 0 )
        zone_free_batch(gpu_device->memory, ptrs, nb_items);
}
#endif  /* !defined(PARSEC_GPU_ALLOC_PER_TILE) */

/**
 * Try to find memory space to move all data on the GPU. We attach a device_elem to
 * a memory_elem as soon as a device_elem is available. If we fail to find enough
 * available elements, we push all the elements handled during this allocation
 * back into the pool of available device_elem, to be picked up by another call
 * (this call will remove them from the current task).
 * The memory of all flows that need a new copy on the device is first allocated
 * in a single batch; flows that could not be served by the batch fall back to
 * individual allocations, evicting data from the LRU if necessary.
 * Returns:
 *   PARSEC_HOOK_RETURN_DONE:  All gpu_mem/mem_elem have been initialized
 *   PARSEC_HOOK_RETURN_NEXT:  The task needs to rescheduled
 */
static inline int
parsec_device_data_reserve_space( parsec_device_gpu_module_t* gpu_device,
                                  parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    parsec_data_t* master, *oldmaster;
    const parsec_flow_t *flow;
    int i, j, data_avail_epoch = 0, copy_readers_update = 0;
    parsec_gpu_data_copy_t *gpu_mem_lru_cycling = NULL;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    void *prealloc[MAX_PARAM_COUNT];
#endif

    /* defensive (valgrind reported uninitialized accesses to temp_loc)*/
    memset(temp_loc, 0, sizeof(temp_loc));

#if defined(PARSEC_DEBUG_NOISIER)
    char task_name[MAX_TASK_STRLEN];
    parsec_task_snprintf(task_name, MAX_TASK_STRLEN, this_task);
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    (void)copy_readers_update; // potentially unused

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    parsec_device_data_prealloc_space(gpu_device, gpu_task, prealloc);
#endif

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Investigating flow %s:%d flags %d",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             flow->name, i, flow->flow_flags);
        temp_loc[i] = NULL;
        if (this_task->data[i].data_in == NULL)
            continue;

        master   = this_task->data[i].data_in->original;
        parsec_atomic_lock(&master->lock);
        gpu_elem = PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;

        /* There is already a copy on the device */
        if( NULL != gpu_elem ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Flow %s:%i has a copy on the device %p%s",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 flow->name, i, gpu_elem,
                                 gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ? " [in transfer]" : "");
            if ( gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {
                /* The data is under transfer, which is fine for RO data since we always force an event
                 * at the end of this step so we do not need to have a special case for this. The forced
                 * event will ensure the data will be available on the GPU by the time this task will move
                 * to the next step. For WRITE flows, we have to abort here and come back later because
                 * transfer_ownership will bark at WRITE flows that are under transfer.
                 */
                if (0 != (PARSEC_FLOW_ACCESS_WRITE & flow->flow_flags)) {
                    this_task->data[i].data_out = NULL;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                        "GPU[%d:%s]:%s:\tWrite flow under transfer for copy %p",
                                        gpu_device->super.device_index, gpu_device->super.name, task_name,
                                        gpu_elem);
                    for (j = 0; j <= i; ++j) {
                        if (temp_loc[j] == NULL) continue;
                        if( 0 == (temp_loc[j]->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
                        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                            "GPU[%d:%s]:%s:\tAdd copy %p [ref_count %d] back to the LRU list",
                                            gpu_device->super.device_index, gpu_device->super.name, task_name,
                                            temp_loc[j], temp_loc[j]->super.super.obj_reference_count);
                        parsec_list_nolock_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
                    }
                    parsec_atomic_unlock(&master->lock);
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
                    parsec_device_data_prealloc_release(gpu_device, prealloc, this_task->task_class->nb_flows);
#endif
                    return PARSEC_HOOK_RETURN_AGAIN;
                }

            }
            parsec_atomic_unlock(&master->lock);
            continue;
        }

#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Allocate GPU copy %p sz %zu [ref_count %d] for data %p",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             gpu_elem, gpu_task->flow_nb_elts[i], gpu_elem->super.super.obj_reference_count, master);
        gpu_elem->flags = PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED;
    malloc_data:
        copy_readers_update = 0;
        assert(0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        if( NULL != prealloc[i] ) {
            /* use the memory allocated in the batch */
            gpu_elem->device_private = prealloc[i];
            prealloc[i] = NULL;
        } else {
            gpu_elem->device_private = zone_malloc(gpu_device->memory, gpu_task->flow_nb_elts[i]);
        }
        if( NULL == gpu_elem->device_private ) {
#endif

        find_another_data:
            temp_loc[i] = NULL;
            /* Look for a data_copy to free */
            lru_gpu_elem = (parsec_gpu_data_copy_t*)parsec_list_nolock_pop_front(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the begining of
                 * the LRU (in order to be reused asap) and return with error.
                 */
            release_temp_and_return:
#if defined(PARSEC_DEBUG_NOISIER)
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\tRequest space on GPU failed for flow %s index %d/%d for task %s",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     flow->name, i, this_task->task_class->nb_flows, task_name );
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
                for( j = 0; j <= i; j++ ) {
                    /* This flow could be a control flow */
                    if( NULL == temp_loc[j] ) continue;
                    /* This flow could be non-parsec-owned, in which case we can't reclaim it */
                    if( 0 == (temp_loc[j]->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:%s:\tAdd copy %p [ref_count %d] back to the LRU list",
                                         gpu_device->super.device_index, gpu_device->super.name, task_name,
                                         temp_loc[j], temp_loc[j]->super.super.obj_reference_count);
                    /* push them at the head to reach them again at the next iteration */
                    parsec_list_nolock_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
                }
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
                PARSEC_OBJ_RELEASE(gpu_elem);
#endif
                parsec_atomic_unlock(&master->lock);
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
                parsec_device_data_prealloc_release(gpu_device, prealloc, this_task->task_class->nb_flows);
#endif
                return PARSEC_HOOK_RETURN_NEXT;
            } else if (NULL != lru_gpu_elem->original) {
                /* account for discarded data */
                parsec_data_copy_t* cpu_copy = lru_gpu_elem->original->device_copies[0];
                if (cpu_copy->flags & PARSEC_DATA_FLAG_DISCARDED) {
                    parsec_atomic_fetch_dec_int64(&gpu_device->super.nb_discarded);
                }
            }

            PARSEC_LIST_ITEM_SINGLETON(lru_gpu_elem);
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s: Evaluate LRU-retrieved GPU copy %p [ref_count %d] original %p",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count,
                                 lru_gpu_elem->original);

            if( gpu_mem_lru_cycling == lru_gpu_elem ) {
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: Cycle detected on allocating memory for %s",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name);
                temp_loc[i] = lru_gpu_elem;  /* save it such that it gets pushed back into the LRU */
                goto release_temp_and_return;
            }

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the parsec_device_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s: Drop LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count, lru_gpu_elem->original);
                /* We do not add the copy back into the LRU. This means that for now this copy is not
                 * tracked via the LRU (despite being only used in read mode) and instead is dangling
                 * on other tasks. Thus, it will eventually need to be added back into the LRU when
                 * current task using it completes.
                 */
                goto find_another_data;
            }
            /* It's also possible that the ref_count of that element is bigger than 1
             * In that case, it's because some task completion did not execute yet, and
             * we need to keep it in the list until it reaches 1.
             */
            if( lru_gpu_elem->super.super.obj_reference_count > 1 ) {
                /* It's also possible (although unlikely) that we livelock here:
                 * if gpu_mem_lru has *only* elements with readers == 0 but
                 * ref_count > 1, then we might pop/push forever. We save the
                 * earliest element found and if we see it again it means we
                 * run over the entire list without finding a suitable replacement.
                 * We need to make progress on something else. This remains safe for as long as the
                 * LRU is only modified by a single thread (in this case the current thread).
                 */
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s: Push back LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name,
                                     lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count, lru_gpu_elem->original);
                assert(0 != (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
                parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                goto find_another_data;
            }

            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
            if ( NULL != lru_gpu_elem->original ) {
                /* Let's check we're not trying to steal one of our own data */
                oldmaster = lru_gpu_elem->original;
                if( !parsec_atomic_trylock( &oldmaster->lock ) ) {
                    /* Even if we have the lock on oldmaster, any other thread
                     * might be adding/removing other elements to the list, so we
                     * need to protect all accesses to gpu_mem_lru with the locked version */
                    assert(0 != (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
                    parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                    gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                    goto find_another_data;
                }
                for( j = 0; j < i; j++ ) {
                    if( NULL == this_task->data[j].data_in ) continue;
                    if( this_task->data[j].data_in->original == oldmaster ) {
                        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                             "GPU[%d:%s]:%s: Drop LRU-retrieved GPU copy %p [ref_count %d] already in use by same task %d:%d original %p",
                                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                                             lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count, i, j, lru_gpu_elem->original);
                        /* If we are the owner of this tile we need to make sure it remains available for
                         * other tasks or we run in deadlock situations.
                         */
                        parsec_atomic_unlock( &oldmaster->lock );
                        goto find_another_data;
                    }
                }
                /* There is still one last thing to ensure: if another accelerator uses this copy as a source
                 * for a d2d transfer it will mark it by atomically increasing the readers. So, we need to
                 * avoid altering the copy while they are using it, by protecting the access to the readers
                 * with a cas.
                 */
                if( !parsec_atomic_cas_int32(&lru_gpu_elem->readers, 0, -PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL) ) {
                    assert(lru_gpu_elem->readers > 0);
                    /* we can't use this copy, push it back */
                    parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
                    gpu_mem_lru_cycling = (NULL == gpu_mem_lru_cycling) ? lru_gpu_elem : gpu_mem_lru_cycling;  /* update the cycle detector */
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:%s: Push back LRU-retrieved GPU copy %p [readers %d, ref_count %d] original %p : Concurrent accesses",
                                         gpu_device->super.device_index, gpu_device->super.name, task_name,
                                         lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->super.super.obj_reference_count,
                                         lru_gpu_elem->original);
                    parsec_atomic_unlock( &oldmaster->lock );
                    goto find_another_data;
                }
                copy_readers_update = PARSEC_DEVICE_DATA_COPY_ATOMIC_SENTINEL;
                /* Check if this copy is the last dangling reference to the oldmaster. This is safe to do as we own one of the data refcounts. */
                int do_unlock = oldmaster->super.obj_reference_count != 1;
                parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
                parsec_atomic_wmb();
                /* detach could have released the oldmaster if it only had a single refcount */
                if( do_unlock )
                    parsec_atomic_unlock( &oldmaster->lock );

                /* The data is not used, it's not one of ours, and it has been detached from the device
                 * so no other device can use it as a source for their copy : we can free it or reuse it */
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\ttask %s:%d repurpose copy %p [ref_count %d] to data %p instead of %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name, this_task->task_class->name, i, lru_gpu_elem,
                                     lru_gpu_elem->super.super.obj_reference_count, master, oldmaster);
            }
            else {
                PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:%s:\ttask %s:%d found detached memory from previously destructed data %p",
                                     gpu_device->super.device_index, gpu_device->super.name, task_name, this_task->task_class->name, i, lru_gpu_elem);
                oldmaster = NULL;
            }
            gpu_device->super.nb_evictions++;
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
            /* Let's free this space, and try again to malloc some space */
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:%s Release GPU copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p",
                                 gpu_device->super.device_index, gpu_device->super.name, task_name,
                                 lru_gpu_elem, lru_gpu_elem->device_private, lru_gpu_elem->super.super.obj_reference_count,
                                 oldmaster);
            parsec_device_release_gpu_copy(gpu_device, lru_gpu_elem);
            data_avail_epoch++;
            goto malloc_data;
        }
        PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                             "GPU[%d:%s] Succeeded Allocating GPU copy %p at real address %p [ref_count %d] for data %p",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->device_private, gpu_elem->super.super.obj_reference_count, master);
#if defined(PARSEC_PROF_TRACE)
        if((gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) &&
                        (gpu_device->exec_stream[0]->prof_event_track_enable ||
                         gpu_device->exec_stream[1]->prof_event_track_enable)) {
            parsec_profiling_trace_flags(gpu_device->exec_stream[0]->profiling,
                                         parsec_gpu_allocate_memory_key, (int64_t)gpu_elem->device_private,
                                         gpu_device->super.device_index,
                                         &gpu_task->flow_nb_elts[i], PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
        }
#endif
#else
        gpu_elem = lru_gpu_elem;
        /* The readers must be manipulated via atomic operations to avoid race conditions
         * with threads that would use them as candidate for updating their own copies.
         */
        if (copy_readers_update != 0) {
            parsec_atomic_fetch_add_int32(&gpu_elem->readers, copy_readers_update);
        }
#endif

        /* Do not push it back into the LRU for now to prevent others from discovering
         * this copy and trying to acquire it. If we fail to find all the copies we need
         * we will push it back in the release_temp_and_return, otherwise they will become
         * available once properly updated.
         */
        gpu_elem->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
        gpu_elem->version = UINT_MAX;  /* scrap value for now */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]: GPU copy %p [ref_count %d] gets created with version 0",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count);
        parsec_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;
        /* set the new datacopy type to the correct one */
        this_task->data[i].data_out->dtt = this_task->data[i].data_in->dtt;
        temp_loc[i] = gpu_elem;
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:%s: Retain and insert GPU copy %p [ref_count %d] in LRU",
                             gpu_device->super.device_index, gpu_device->super.name, task_name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count);
        assert(0 != (gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) );
        // TODO: is this assert right?
        //assert(master->device_copies[0]->device_private == NULL ||
        //       master->device_copies[0]->coherency_state != PARSEC_DATA_COHERENCY_INVALID);
        parsec_atomic_unlock(&master->lock);
    }
#if !defined(PARSEC_GPU_ALLOC_PER_TILE)
    /* memory preallocated for flows that found a copy on the device in the meantime
     * (e.g., multiple flows of the task referring to the same data) */
    parsec_device_data_prealloc_release(gpu_device, prealloc, this_task->task_class->nb_flows);
#endif
    if( data_avail_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Generic fallback for parsec_device_memcpy_multi_async_fn_t: issues @p nb_items
 *    individual gpu->memcpy_async() calls. Does not stop at the first failure -- every
 *    item is attempted regardless, and the first error encountered (if any) is returned.
 *    Used by backends that do not (yet) provide a native batched copy primitive.
 */
int
parsec_device_generic_memcpy_multi_async(parsec_device_gpu_module_t *gpu, parsec_gpu_exec_stream_t *gpu_stream,
                                         void **dsts, void **srcs, size_t *sizes,
                                         parsec_device_transfer_direction_t *directions, int nb_items)
{
    int ret, rc = PARSEC_SUCCESS;

    for(int i = 0; i < nb_items; i++) {
        ret = gpu->memcpy_async(gpu, gpu_stream, dsts[i], srcs[i], sizes[i], directions[i]);
        if( (PARSEC_SUCCESS == rc) && (PARSEC_SUCCESS != ret) )
            rc = ret;
    }
    return rc;
}

/* Append the copies needed to stage in every flow set in flow_mask of gtask to
 * the parallel arrays dsts/srcs/sizes/directions, starting at index *nb_items.
 * The arrays must have room for MAX_PARAM_COUNT additional entries. Upon return
 * *nb_items accounts for the new entries and, if any entry was added, *dev is
 * the device that must issue the copies.
 */
static void
parsec_default_gpu_stage_in_collect(parsec_gpu_task_t                  *gtask,
                                    uint32_t                            flow_mask,
                                    void                              **dsts,
                                    void                              **srcs,
                                    size_t                             *sizes,
                                    parsec_device_transfer_direction_t *directions,
                                    int                                *nb_items,
                                    parsec_device_gpu_module_t        **dev)
{
    parsec_data_copy_t * source;
    parsec_data_copy_t * dest;
    parsec_device_gpu_module_t *src_dev;
    parsec_device_gpu_module_t *dst_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;
    int n = *nb_items;

    for(int i = 0; i < task->task_class->nb_flows; i++) {
        if( !(flow_mask & (1U << i)) ) continue;
        source = gtask->sources[i];
        assert(source->device_private != NULL);
        dest = task->data[i].data_out;
        src_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(source->device_index);
        dst_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(dest->device_index);

        if((src_dev->super.type & PARSEC_DEV_ANY_TYPE) == (dst_dev->super.type & PARSEC_DEV_ANY_TYPE)) {
            assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
            dir = parsec_device_gpu_transfer_direction_d2d;
        } else {
            dir = parsec_device_gpu_transfer_direction_h2d;
        }

        count = (source->original->nb_elts <= dest->original->nb_elts) ?
            source->original->nb_elts : dest->original->nb_elts;
        dsts[n]       = dest->device_private;
        srcs[n]       = source->device_private;
        sizes[n]      = count;
        directions[n] = dir;
        n++;
        *dev = dst_dev;
    }
    *nb_items = n;
}

/* Default stage_in function to transfer data to the GPU device.
 * Transfer the <count> contiguous bytes for every flow set in flow_mask from
 * task->data[i].data_in to task->data[i].data_out, as a single multi-item transfer.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_in(parsec_gpu_task_t        *gtask,
                            uint32_t                  flow_mask,
                            parsec_gpu_exec_stream_t *gpu_stream)
{
    int ret, nb_items = 0;
    parsec_device_gpu_module_t *dst_dev = NULL;
    void *dsts[MAX_PARAM_COUNT];
    void *srcs[MAX_PARAM_COUNT];
    size_t sizes[MAX_PARAM_COUNT];
    parsec_device_transfer_direction_t directions[MAX_PARAM_COUNT];

    parsec_default_gpu_stage_in_collect(gtask, flow_mask, dsts, srcs, sizes, directions,
                                        &nb_items, &dst_dev);
    if( 0 == nb_items )
        return PARSEC_HOOK_RETURN_DONE;
    ret = dst_dev->memcpy_multi_async( dst_dev, gpu_stream, dsts, srcs, sizes, directions, nb_items );
    if(PARSEC_SUCCESS != ret)
        return PARSEC_HOOK_RETURN_ERROR;
    return PARSEC_HOOK_RETURN_DONE;
}

/* Append the copies needed to stage out every flow set in flow_mask of gtask to
 * the parallel arrays dsts/srcs/sizes/directions, starting at index *nb_items.
 * The arrays must have room for MAX_PARAM_COUNT additional entries. Upon return
 * *nb_items accounts for the new entries and, if any entry was added, *dev is
 * the device that must issue the copies.
 * Returns PARSEC_HOOK_RETURN_DONE, or PARSEC_HOOK_RETURN_ERROR if a host buffer
 * could not be allocated.
 */
static int
parsec_default_gpu_stage_out_collect(parsec_gpu_task_t                  *gtask,
                                     uint32_t                            flow_mask,
                                     void                              **dsts,
                                     void                              **srcs,
                                     size_t                             *sizes,
                                     parsec_device_transfer_direction_t *directions,
                                     int                                *nb_items,
                                     parsec_device_gpu_module_t        **dev)
{
    parsec_data_copy_t * source;
    parsec_data_copy_t * dest;
    parsec_device_gpu_module_t *dst_dev, *src_dev;
    parsec_task_t *task = gtask->ec;
    size_t count;
    parsec_device_transfer_direction_t dir;
    int i, n = *nb_items;

    for(i = 0; i < task->task_class->nb_flows; i++){
        if(flow_mask & (1U << i)){
            source = task->data[i].data_out;
            dest = source->original->device_copies[0];
            dst_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(dest->device_index);
            src_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get(source->device_index);

            count = (source->original->nb_elts <= dest->original->nb_elts) ? source->original->nb_elts :
                        dest->original->nb_elts;
            if( (src_dev->super.type & PARSEC_DEV_ANY_TYPE) == (dst_dev->super.type & PARSEC_DEV_ANY_TYPE) ) {
                assert( src_dev->peer_access_mask & (1 << dst_dev->super.device_index) );
                dir = parsec_device_gpu_transfer_direction_d2d;
            } else {
                dir = parsec_device_gpu_transfer_direction_d2h;
                if (dest->device_private == NULL && dest->alloc_cb != NULL) {
                    dest->alloc_cb(dest, 0); // allocate on host
                    PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream,
                                         "GPU[%d:%s]: Allocated host memory for CPU copy %p data %p ptr %p",
                                         src_dev->super.device_index, src_dev->super.name, dest, source->original, dest->device_private);
                }
                if (dest->device_private == NULL) {
                    parsec_warning("Application failed to allocate host memory for pushout data %p copy %p!", source->original, dest);
                    *nb_items = n;
                    return PARSEC_HOOK_RETURN_ERROR;
                }
            }
            dsts[n]       = dest->device_private;
            srcs[n]       = source->device_private;
            sizes[n]      = count;
            directions[n] = dir;
            n++;
            *dev = src_dev;
        }
    }
    *nb_items = n;
    return PARSEC_HOOK_RETURN_DONE;
}

/* Default stage_out function to transfer data from the GPU device.
 * Transfer the <count> contiguous bytes for every flow set in flow_mask from
 * task->data[i].data_in to task->data[i].data_out, as a single multi-item transfer.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_gpu_stage_out(parsec_gpu_task_t        *gtask,
                             uint32_t                  flow_mask,
                             parsec_gpu_exec_stream_t *gpu_stream)
{
    int ret, nb_items = 0;
    parsec_device_gpu_module_t *src_dev = NULL;
    void *dsts[MAX_PARAM_COUNT];
    void *srcs[MAX_PARAM_COUNT];
    size_t sizes[MAX_PARAM_COUNT];
    parsec_device_transfer_direction_t directions[MAX_PARAM_COUNT];

    ret = parsec_default_gpu_stage_out_collect(gtask, flow_mask, dsts, srcs, sizes, directions,
                                               &nb_items, &src_dev);
    if( PARSEC_HOOK_RETURN_DONE != ret )
        return ret;
    if( 0 == nb_items )
        return PARSEC_HOOK_RETURN_DONE;
    ret = src_dev->memcpy_multi_async( src_dev, gpu_stream, dsts, srcs, sizes, directions, nb_items );
    if(PARSEC_SUCCESS != ret) {
        return PARSEC_HOOK_RETURN_ERROR;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* Capacity of the copy descriptor arrays used to merge the default stage_in/stage_out
 * transfers of the tasks in a ring into batched copies. The arrays are flushed
 * whenever a task might not fit anymore.
 */
#define PARSEC_DEVICE_MAX_BATCH_COPIES (4 * MAX_PARAM_COUNT)

typedef struct parsec_device_copy_batch_s {
    int                                nb_items;
    int                                nb_tasks;
    parsec_gpu_task_t                 *tasks[PARSEC_DEVICE_MAX_BATCH_SIZE];  /**< tasks with copies in the batch */
    void                              *dsts[PARSEC_DEVICE_MAX_BATCH_COPIES];
    void                              *srcs[PARSEC_DEVICE_MAX_BATCH_COPIES];
    size_t                             sizes[PARSEC_DEVICE_MAX_BATCH_COPIES];
    parsec_device_transfer_direction_t directions[PARSEC_DEVICE_MAX_BATCH_COPIES];
} parsec_device_copy_batch_t;

/* Issue all pending copies of @p batch as a single batched copy on @p gpu_stream.
 * If the copy cannot be issued, all tasks that contributed copies to the batch get
 * @p error_rc as last_status.
 */
static void
parsec_device_copy_batch_flush(parsec_device_gpu_module_t *gpu_device,
                               parsec_device_copy_batch_t *batch,
                               parsec_gpu_exec_stream_t   *gpu_stream,
                               int                         error_rc)
{
    int rc;

    if( 0 != batch->nb_items ) {
        rc = gpu_device->memcpy_multi_async(gpu_device, gpu_stream, batch->dsts, batch->srcs,
                                            batch->sizes, batch->directions, batch->nb_items);
        if( PARSEC_SUCCESS != rc ) {
            parsec_warning("GPU[%d:%s]: batched copy of %d items for %d tasks failed rc=%d @%s:%d",
                           gpu_device->super.device_index, gpu_device->super.name,
                           batch->nb_items, batch->nb_tasks, rc, __func__, __LINE__);
            for( int i = 0; i < batch->nb_tasks; i++ ) {
                batch->tasks[i]->last_status = error_rc;
            }
        }
    }
    batch->nb_items = 0;
    batch->nb_tasks = 0;
}

/**
 * If the most current version of the data is not yet available on the GPU memory
 * decide whether a transfer is needed, and if so mark this flow's bit in
 * *transfer_mask. The caller is responsible for issuing the actual gpu_task->stage_in
 * call (once, covering every flow whose bit got set) after all flows have been decided.
 * Returns hook special return codes or a positive number:
 *    HOOK_DONE: The most recent version of the data is already available on the GPU
 *    1: A transfer is needed (bit set in *transfer_mask) or already scheduled
 *   HOOK_AGAIN: Retry this flow later, no ownership/coherency state has changed
 */
static inline int
parsec_device_data_stage_in( parsec_device_gpu_module_t* gpu_device,
                             const parsec_flow_t *flow,
                             parsec_data_pair_t* task_data,
                             parsec_gpu_task_t *gpu_task,
                             parsec_gpu_exec_stream_t *gpu_stream,
                             uint32_t *transfer_mask )
{
    int32_t type = flow->flow_flags;
    parsec_data_copy_t *candidate = task_data->data_in;  /* best candidate for now */
    parsec_data_t* original = candidate->original;
    parsec_gpu_data_copy_t* gpu_elem = task_data->data_out;
    size_t nb_elts = gpu_task->flow_nb_elts[flow->flow_index];
    int transfer_from = -1;

    if( gpu_task->task_type == PARSEC_GPU_TASK_TYPE_PREFETCH ) {
        PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream,
                             "GPU[%d:%s]: Prefetch task %p is staging in",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_task);
    }

    parsec_atomic_lock( &original->lock );

    gpu_task->sources[flow->flow_index] = candidate;  /* default source for the transfer */
    /**
     * If the data will be accessed in write mode, remove it from any GPU data management
     * lists until the task is completed.
     */
    if( PARSEC_FLOW_ACCESS_WRITE & type ) {
        if (gpu_elem->readers > 0 ) {
            if( !((1 == gpu_elem->readers) && (PARSEC_FLOW_ACCESS_READ & type)) ) {
                parsec_warning("GPU[%d:%s]:\tWrite access to data copy %p [ref_count %d] with existing readers [%d]\n"
                               "\tPossible anti-dependency, or concurrent accesses: please prevent that with CTL dependencies\n",
                               gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->readers);
            }
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tDetach writable GPU copy %p [ref_count %d] from any lists",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_elem, gpu_elem->super.super.obj_reference_count);
        /* make sure the element is not in any tracking lists */
        parsec_list_item_ring_chop((parsec_list_item_t*)gpu_elem);
        PARSEC_LIST_ITEM_SINGLETON(gpu_elem);
    }

    transfer_from = parsec_data_start_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

#if 0
    // JS: this case should be handled by transfer-ownership because the host copy should be invalid if its version is 0
    /* If data is from NEW (it doesn't have a source_repo_entry and is not a direct data collection reference),
     * and nobody has touched it yet, then we don't need to pull it in, we have created it already, that's enough. */
    /*
     * TODO: this test is not correct for anything but PTG
     */
    if( (NULL == task_data->source_repo_entry) &&
        (NULL == task_data->data_in->original->dc) &&
        (0 == task_data->data_in->version) )
        transfer_from = -1;
#endif // 0

    /* Update the transferred required_data_in size */
    gpu_device->super.required_data_in += original->nb_elts;

    if( -1 == transfer_from ) {  /* Do not need to be transferred */
        gpu_elem->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;

        parsec_data_end_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);

        if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) ) {
            gpu_elem->version = candidate->version + 1;
        }

        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tNO Move for data copy %p v%d [ref_count %d, key %x]",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->version, gpu_elem->super.super.obj_reference_count, original->key);
        parsec_atomic_unlock( &original->lock );
        /* TODO: data keeps the same coherence flags as before */
        return PARSEC_HOOK_RETURN_DONE;
    }
    /* If it is already under transfer, don't schedule the transfer again.
     * This happens if the task refers twice (or more) to the same input flow */
    if( gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tMove data copy %p [ref_count %d, key %x] of %zu bytes: data copy is already under transfer, ignoring double request",
                             gpu_device->super.device_index, gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, nb_elts);
        parsec_atomic_unlock( &original->lock );
        return 1;  /* positive returns have special meaning and are used for optimizations */
    }

    /* Try to find an alternate source, to avoid always transferring from the host to the device.
     * Current limitations: only for read-only data used read-only on the hosting GPU. */
    parsec_device_gpu_module_t *candidate_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get( candidate->device_index );
    if( (PARSEC_FLOW_ACCESS_READ & type) && !(PARSEC_FLOW_ACCESS_WRITE & type) ) {
        int potential_alt_src = 0;
        PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tSelecting candidate data copy %p [ref_count %d] on data %p",
                             gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original);
        if( (gpu_device->super.type & PARSEC_DEV_ANY_TYPE) == (candidate_dev->super.type & PARSEC_DEV_ANY_TYPE) ) {
            if( gpu_device->peer_access_mask & (1 << candidate_dev->super.device_index) ) {
                /* We can directly do D2D, so let's skip the selection */
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tskipping candidate lookup: data_in copy %p on %s has PEER ACCESS",
                                     gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, candidate_dev->super.name);
                goto src_selected;
            }
        }

        /* If gpu_elem is not invalid, then it is already there and the right version,
         * and we're not going to transfer from another source, skip the selection */
        if( gpu_elem->coherency_state != PARSEC_DATA_COHERENCY_INVALID ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tskipping candidate lookup: VALID COPY for %p already on this GPU at %p",
                                 gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, gpu_elem);
            goto src_selected;
        }

        for(int t = 1; t < (int)parsec_nb_devices; t++) {
            parsec_device_gpu_module_t *target = (parsec_device_gpu_module_t*)parsec_mca_device_get(t);
            if( !(gpu_device->peer_access_mask & (1 << target->super.device_index)) ) {
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tskipping device: %s has NO PEER ACCESS",
                                     gpu_device->super.device_index, gpu_device->super.name, target->super.name);
                continue;
            }
            assert( PARSEC_DEV_IS_GPU(target->super.type) );

            candidate = original->device_copies[t];
            if( (NULL == candidate) || (candidate->version < task_data->data_in->version) ) {
                PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tcopy %p:%d cannot be a candidate VERSION MISMATCH with %p:%d",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     candidate, candidate?(int)candidate->version:-1, task_data->data_in, task_data->data_in->version);
                continue;
            }

            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tData copy %p [ref_count %d] on GPU device %d is a potential alternative source for data_in %p on data %p",
                                 gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.device_index, task_data->data_in, original);
            if(PARSEC_DATA_COHERENCY_INVALID == candidate->coherency_state) {
                /* We're already pulling this data on candidate...
                 * If there is another candidate that already has it, we'll use
                 * that one; otherwise, we'll fall back on the CPU version. */
                potential_alt_src = 1;
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tData copy %p [ref_count %d] on GPU device %d is invalid, continuing to look for alternatives",
                                     gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.device_index);
                continue;
            }
            /* We have a candidate for the d2d transfer. */
            int readers = parsec_atomic_fetch_inc_int32(&candidate->readers);
            if( readers >= 0 ) {
                parsec_atomic_rmb();
                /* Coordination protocol with the owner of the candidate. If the owner had repurposed the copy, by the
                 * time we succesfully increase the readers, the device copy will be associated with a different data.
                 */
                if( (candidate->original == original) && (candidate->version >= task_data->data_in->version) ) {
                    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tData copy %p [ref_count %d] on PaRSEC device %s is the best candidate to do Device to Device copy, increasing its readers to %d",
                                         gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name, candidate->readers+1);
                    candidate_dev = target;
                    goto src_selected;
                }
            }
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tCandidate %p [ref_count %d] on PaRSEC device %s is being repurposed by owner device. Looking for another candidate",
                                 gpu_device->super.device_index, gpu_device->super.name, candidate, candidate->super.super.obj_reference_count, target->super.name);
            /* We are trying to use a candidate that is repurposed by the owner device. Let's find another one */
            parsec_atomic_fetch_add_int32(&candidate->readers, -1);
        }
        if( potential_alt_src ) {
            parsec_data_copy_t *cpu_copy = original->device_copies[0];

            /* We found a potential GPU source, but it is not ready now. The
             * CPU fallback is safe only if the host copy is a ready copy of the
             * version this task expects; otherwise, retry the task later. */
            if( (NULL == cpu_copy) ||
                (cpu_copy->version != task_data->data_in->version) ||
                (PARSEC_DATA_COHERENCY_INVALID == cpu_copy->coherency_state) ||
                (PARSEC_DATA_STATUS_UNDER_TRANSFER == cpu_copy->data_transfer_status) ||
                (NULL == cpu_copy->device_private) ) {
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]:\tThere is a potential alternative source for data_in %p [ref_count %d] in original %p to go in copy %p [ref_count %d], but neither the GPU nor the CPU copy is ready; retry later",
                                     gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original, gpu_elem, gpu_elem->super.super.obj_reference_count);
                /* No ownership/coherency state has been changed yet, so it is
                 * safe to defer this task until a valid source copy is ready. */
                parsec_atomic_unlock( &original->lock );
                return PARSEC_HOOK_RETURN_AGAIN;
            }

            /* We found a potential alternative source, but it's not ready now.
             * Use the CPU copy because it has the expected version. */
            /** TODO: when considering RW accesses, don't forget to chop gpu_elem
             *        from its queue... */
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tThere is a potential alternative source for data_in %p [ref_count %d] in original %p to go in copy %p [ref_count %d], but it is not ready, falling back on CPU source",
                                 gpu_device->super.device_index, gpu_device->super.name, task_data->data_in, task_data->data_in->super.super.obj_reference_count, original, gpu_elem, gpu_elem->super.super.obj_reference_count);
        }

        /* We fall back on the CPU copy */
        candidate = original->device_copies[0];
        candidate_dev = (parsec_device_gpu_module_t*)parsec_mca_device_get( candidate->device_index );
    }

 src_selected:
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]:\t\tMove %s data copy %p [ref_count %d, key %x] of %zu bytes\t(src dev: %d, v:%d, ptr:%p, copy:%p [ref_count %d, under_transfer: %d, coherency_state: %d] / dst dev: %d, v:%d, ptr:%p)",
                         gpu_device->super.device_index, gpu_device->super.name,
                         PARSEC_DEV_IS_GPU(candidate_dev->super.type) ? "D2D": "H2D",
                         gpu_elem, gpu_elem->super.super.obj_reference_count, original->key, nb_elts,
                         candidate_dev->super.device_index, candidate->version, (void*)candidate->device_private,
                         candidate, candidate->super.super.obj_reference_count, candidate->data_transfer_status, candidate->coherency_state,
                         gpu_device->super.device_index, gpu_elem->version, (void*)gpu_elem->device_private);

#if defined(PARSEC_PROF_TRACE)
    if( gpu_stream->prof_event_track_enable  ) {
        parsec_profile_data_collection_info_t info;

        if( NULL != original->dc ) {
            info.desc    = original->dc;
            info.data_id = original->key;
        } else {
            assert( PARSEC_GPU_TASK_TYPE_PREFETCH != gpu_task->task_type );
            info.desc    = (parsec_dc_t*)original;
            info.data_id = -1;
        }
        gpu_task->prof_key_end = -1;

        if( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type && (gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_PREFETCH) ) {
            gpu_task->prof_key_end = parsec_gpu_prefetch_key_end;
            gpu_task->prof_event_id = (int64_t)gpu_elem->device_private;
            gpu_task->prof_tp_id = gpu_device->super.device_index;
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   parsec_gpu_prefetch_key_start,
                                   gpu_task->prof_event_id,
                                   gpu_task->prof_tp_id,
                                   &info);
        }
        if(PARSEC_GPU_TASK_TYPE_PREFETCH != gpu_task->task_type && (gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_IN) ) {
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   parsec_gpu_movein_key_start,
                                   (int64_t)gpu_elem->device_private,
                                   gpu_device->super.device_index,
                                   &info);
        }
        if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_MEM_USE) {
            parsec_device_gpu_memory_prof_info_t _info;
            _info.size = (uint64_t)nb_elts;
            _info.data_key = gpu_elem->original->key;
            _info.dc_id = (uint64_t)(gpu_elem->original->dc);
            parsec_profiling_trace_flags(gpu_stream->profiling,
                                         parsec_gpu_use_memory_key_start, (uint64_t)
                                         gpu_elem->device_private,
                                         gpu_device->super.device_index, &_info,
                                         PARSEC_PROFILING_EVENT_HAS_INFO);
        }
    }
#endif
    gpu_task->sources[flow->flow_index] = candidate;  /* save the candidate for release on transfer completion */
    /* Mark this flow as needing a transfer from the source device; the caller issues
     * the actual gpu_task->stage_in call once, covering every flow marked this way. */
    *transfer_mask |= (1U << flow->flow_index);
    assert(candidate_dev->super.device_index < gpu_device->super.data_in_array_size);
    gpu_device->super.data_in_from_device[candidate_dev->super.device_index] += nb_elts;
    if( PARSEC_GPU_TASK_TYPE_KERNEL == gpu_task->task_type )
        gpu_device->super.nb_data_faults += nb_elts;

    /* We assign the version of the data preemptively (i.e. before the task is executing)
     * For read-only data, the GPU copy will get the same version as the source
     * For write-only or read-write data, we increment the version number.
     * The copy is still invalid & marked to be under transfer until the transfer_ownership is ended */
    assert((gpu_elem->version != candidate->version) || (gpu_elem->data_transfer_status == PARSEC_DATA_STATUS_NOT_TRANSFER));
    if( (PARSEC_FLOW_ACCESS_WRITE & type) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) )
        gpu_elem->version = candidate->version + 1;
    else
        gpu_elem->version = candidate->version;
    gpu_elem->data_transfer_status = PARSEC_DATA_STATUS_UNDER_TRANSFER;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: GPU copy %p [ref_count %d] gets the version %d from copy %p version %d [ref_count %d]",
                         gpu_device->super.device_index, gpu_device->super.name,
                         gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->version, candidate, candidate->version, candidate->super.super.obj_reference_count);

    parsec_atomic_unlock( &original->lock );
    return 1;  /* positive returns have special meaning and are used for optimizations */
}

static inline int
parsec_gpu_task_is_singleton(parsec_gpu_task_t *task)
{
    parsec_list_item_t *item = &task->list_item;

#if defined(PARSEC_DEBUG_PARANOID)
    if( ((parsec_list_item_t *)(void *)0xdeadbeefL == item->list_next) ||
        ((parsec_list_item_t *)(void *)0xdeadbeefL == item->list_prev) ) {
        return 0;
    }
#endif
    return (item->list_next == item) && (item->list_prev == item);
}

static inline int
parsec_gpu_task_selected_chore_allows_batch(parsec_task_t *task,
                                            parsec_device_module_t *device);

/* Returns true if @p stream is an execution stream (not the h2d or d2h stream) */
static inline bool
parsec_gpu_stream_is_exec(parsec_device_gpu_module_t *gpu_device,
                          parsec_gpu_exec_stream_t *stream)
{
    return (stream != gpu_device->exec_stream[0]) && (stream != gpu_device->exec_stream[1]);
}

static inline void
parsec_gpu_stream_push_pending(parsec_device_gpu_module_t *gpu_device,
                               parsec_gpu_exec_stream_t *stream,
                               parsec_gpu_task_t *task)
{
    /* Execution streams keep the batchable tasks apart from the other tasks, so
     * that their submission can be delayed without delaying the other tasks (see
     * parsec_device_exec_stream_grab()). Both queues keep the arrival order. */
    if( parsec_gpu_stream_is_exec(gpu_device, stream) ) {
        bool can_batch = parsec_mca_device_type_supports_batch(gpu_device->super.type);
        parsec_gpu_task_t *next;
        do {
            next = (parsec_gpu_task_t*)parsec_list_item_ring_chop(&task->list_item);
            PARSEC_LIST_ITEM_SINGLETON(&task->list_item);
            if( task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL ) {
                parsec_list_nolock_push_front(stream->fifo_pending, &task->list_item);
            } else if( can_batch &&
                       parsec_gpu_task_selected_chore_allows_batch(task->ec, (parsec_device_module_t*)gpu_device) ) {
                parsec_list_nolock_push_back(stream->fifo_batch, &task->list_item);
            } else {
                parsec_list_nolock_push_back(stream->fifo_pending, &task->list_item);
            }
            task = next;
        } while( NULL != task );
        return;
    }
    /* A completed batched kernel returns a proper task ring. Preserve that
     * order when feeding the tasks to the next stream.
     */
    if( !parsec_gpu_task_is_singleton(task) ) {
        parsec_list_nolock_chain_back(stream->fifo_pending, &task->list_item);
        return;
    }
    /* We prioritize internal tasks by pushing them to the front of the queue.
     * Kernels will be handled in the order in which they arrive. */
    if (task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL) {
        parsec_list_nolock_push_front(stream->fifo_pending, (parsec_list_item_t*)task);
    } else if (stream == gpu_device->exec_stream[0] && 0 == parsec_device_check_space_needed(gpu_device, task)) {
        /**
         * Push this task to the front if it has all inputs ready.
         * The sorting is only relevant for stream 0 where we handle inputs.
         * For other streams, we just push the task to the end of the queue.
         */
        parsec_list_nolock_push_front(stream->fifo_pending, (parsec_list_item_t*)task);
    } else {
        parsec_list_nolock_push_back(stream->fifo_pending, &task->list_item);
    }
}

static inline int
parsec_gpu_task_selected_chore_allows_batch(parsec_task_t *task,
                                            parsec_device_module_t *device)
{
    const __parsec_chore_t *chore;

    if( (NULL == task) ||
        (NULL == task->task_class) ||
        (NULL == device) ||
        (task->selected_device != device) ||
        (task->selected_chore < 0) ) {
        return 0;
    }

    chore = &task->task_class->incarnations[task->selected_chore];
    return (PARSEC_DEV_NONE != (chore->type & PARSEC_DEV_ANY_TYPE)) &&
           (0 != (chore->type & device->type)) &&
           (0 != (chore->type & PARSEC_DEV_CHORE_ALLOW_BATCH));
}

/**
 * Returns 1 if the two have the same chore hook.
 */
static inline int parsec_gpu_task_same_chore(parsec_gpu_task_t *task1, parsec_gpu_task_t *task2)
{
    parsec_task_t *t1 = task1->ec;
    parsec_task_t *t2 = task2->ec;

    if( (NULL == t1) ||
        (NULL == t1->task_class) ||
        (NULL == t2) ||
        (NULL == t2->task_class) ||
        (t1->selected_chore != t2->selected_chore) ||
        (t1->task_class->incarnations[t1->selected_chore].hook != t2->task_class->incarnations[t2->selected_chore].hook) ) {
        return 0;
    }

    return 1;
}

/* Append the tasks of fifo_pending accepted by callback to batch_head, and
 * account for them in *nb_tasks. Returns PARSEC_SUCCESS, or the negative
 * callback error code (in which case *nb_tasks still accounts for the tasks
 * appended before the error). */
static int
parsec_gpu_task_collect_batch_from(parsec_list_t *fifo_pending,
                                   parsec_device_module_t *device,
                                   parsec_gpu_task_t *batch_head,
                                   parsec_gpu_task_batch_cb_t callback,
                                   void *callback_data,
                                   int *nb_tasks)
{
    parsec_list_item_t *item, *next;
    int rc;

    assert(NULL != fifo_pending);

    for(item = (parsec_list_item_t *)fifo_pending->ghost_element.list_next;
        item != &fifo_pending->ghost_element;
        item = next) {
        parsec_gpu_task_t *candidate = (parsec_gpu_task_t *)item;
        parsec_task_t *candidate_task = candidate->ec;

        next = (parsec_list_item_t *)item->list_next;
        if( !parsec_gpu_task_selected_chore_allows_batch(candidate_task, device) ) {
            continue;
        }
        rc = callback(candidate, batch_head, callback_data);
        if( rc < 0 ) {
            return rc;
        }
        if( 0 == rc ) {
            (void)parsec_list_nolock_remove(fifo_pending, item);
            (void)parsec_list_item_ring_push(&batch_head->list_item, item);
            (*nb_tasks)++;
        }
    }
    return PARSEC_SUCCESS;
}

int
parsec_gpu_task_collect_batch(parsec_gpu_exec_stream_t *gpu_stream,
                              parsec_gpu_task_t *batch_head,
                              parsec_gpu_task_batch_cb_t callback,
                              void *callback_data)
{
    parsec_device_gpu_module_t *gpu_device;
    parsec_task_t *head_task;
    parsec_device_module_t *device;
    int i, first = -1, nb_exec_streams, nb_tasks = 0;
    int rc;

    assert(NULL != gpu_stream);
    assert(NULL != batch_head);
    assert(NULL != callback);

    parsec_list_item_singleton(&batch_head->list_item);

    head_task = batch_head->ec;
    if( (NULL == head_task) || (NULL == head_task->selected_device) ) {
        return nb_tasks;
    }
    device = head_task->selected_device;
    if( !parsec_mca_device_type_supports_batch(device->type) ||
        !parsec_gpu_task_selected_chore_allows_batch(head_task, device) ) {
        return nb_tasks;
    }

    /* Start with the pending tasks of the calling stream. Execution streams keep
     * their batchable tasks in fifo_batch, the other streams in fifo_pending. */
    rc = parsec_gpu_task_collect_batch_from(gpu_stream->fifo_batch, device, batch_head,
                                            callback, callback_data, &nb_tasks);
    if( rc < 0 ) {
        return rc;
    }
    rc = parsec_gpu_task_collect_batch_from(gpu_stream->fifo_pending, device, batch_head,
                                            callback, callback_data, &nb_tasks);
    if( rc < 0 ) {
        return rc;
    }

    /* Then steal from the other execution streams of the device (exec_stream[2..]).
     * Tasks pending on an execution stream have completed the push stage, so they
     * can execute on any execution stream. All pending queues are only accessed
     * by the thread managing the device. */
    gpu_device = (parsec_device_gpu_module_t*)device;
    nb_exec_streams = gpu_device->num_exec_streams - 2;
    for( i = 0; i < nb_exec_streams; i++ ) {
        if( gpu_device->exec_stream[2+i] == gpu_stream ) {
            first = i;
            break;
        }
    }
    if( first < 0 ) {
        /* not called from an execution stream, don't steal */
        return nb_tasks;
    }
    for( i = 1; i < nb_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *stream = gpu_device->exec_stream[2 + (first + i) % nb_exec_streams];
        rc = parsec_gpu_task_collect_batch_from(stream->fifo_batch, device, batch_head,
                                                callback, callback_data, &nb_tasks);
        if( rc < 0 ) {
            return rc;
        }
    }

    return nb_tasks;
}

static parsec_flow_t parsec_device_d2d_complete_flow = {
    .name = "D2D FLOW",
    .flow_flags = PARSEC_FLOW_ACCESS_READ,
    .flow_index = 0,
};

static parsec_task_class_t parsec_device_d2d_complete_tc = {
    .name = "D2D TRANSFER COMPLETE",
    .flags = 0,
    .task_class_id = 0,
    .nb_flows = 1,
    .nb_parameters = 0,
    .nb_locals = 0,
    .dependencies_goal = 0,
    .params = { NULL, },
    .in = { &parsec_device_d2d_complete_flow, NULL },
    .out = { NULL, },
    .priority = NULL,
    .properties = NULL,
    .initial_data = NULL,
    .final_data = NULL,
    .data_affinity = NULL,
    .key_functions = NULL,
    .make_key = NULL,
    .get_datatype = NULL,
    .prepare_input = NULL,
    .incarnations = NULL,
    .prepare_output = NULL,
    .find_deps = NULL,
    .iterate_successors = NULL,
    .iterate_predecessors = NULL,
    .release_deps = NULL,
    .complete_execution = NULL,
    .new_task = NULL,
    .release_task = NULL,
    .fini = NULL
};

static void
parsec_device_send_transfercomplete_cmd_to_device(parsec_data_copy_t *copy,
                                                  parsec_device_module_t *current_dev,
                                                  parsec_device_module_t *dst_dev)
{
    parsec_gpu_task_t* gpu_task = NULL;
    gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
    gpu_task->task_type = PARSEC_GPU_TASK_TYPE_D2D_COMPLETE;
    gpu_task->release_device_task = free;  /* by default free the device task */
    gpu_task->ec = calloc(1, sizeof(parsec_task_t));
    PARSEC_OBJ_CONSTRUCT(gpu_task->ec, parsec_task_t);
    gpu_task->ec->task_class = &parsec_device_d2d_complete_tc;
    gpu_task->flow[0] = &parsec_device_d2d_complete_flow;
    gpu_task->flow_nb_elts[0] = copy->original->nb_elts;
    gpu_task->ec->priority = INT32_MAX; /* This task should be executed as soon as possible */
    /* The pending heap compares gpu_task->priority, not gpu_task->ec->priority: priority
     * inheritance from ec normally happens in parsec_device_kernel_scheduler(), which this
     * direct-enqueue path bypasses. Set it explicitly so this command is not starved behind
     * normal work. */
    gpu_task->priority = INT32_MAX;
    gpu_task->stage_in  = parsec_default_gpu_stage_in;
    gpu_task->stage_out = parsec_default_gpu_stage_out;
    gpu_task->ec->data[0].data_in = copy;  /* We need to set not-null in data_in, so that the fake flow is
                                            * not ignored when poping the data from the fake task */
    gpu_task->ec->data[0].data_out = copy; /* We "free" data[i].data_out if its readers reaches 0 */
    gpu_task->ec->data[0].source_repo_entry = NULL;
    gpu_task->ec->data[0].source_repo = NULL;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_key_end = -1; /* D2D complete tasks are pure internal management, we do not trace them */
#endif
    (void)current_dev;
    PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                         "GPU[%d:%s]: data copy %p [ref_count %d] D2D transfer is complete, sending order to count it "
                         "to GPU Device %d:%s",
                         current_dev->device_index, current_dev->name, gpu_task->ec->data[0].data_out,
                         gpu_task->ec->data[0].data_out->super.super.obj_reference_count,
                         dst_dev->device_index, dst_dev->name);
    parsec_lifo_push( &(((parsec_device_gpu_module_t*)dst_dev)->pending), (parsec_list_item_t*)gpu_task );
}

static int
parsec_device_callback_complete_push(parsec_device_gpu_module_t   *gpu_device,
                                     parsec_gpu_task_t           **gpu_task,
                                     parsec_gpu_exec_stream_t     *gpu_stream)
{
    (void)gpu_stream;

    parsec_gpu_task_t *gtask = *gpu_task;
    parsec_task_t *task;
    int32_t i;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
#endif
    const parsec_flow_t        *flow;
    /**
     * Even though gpu event return success, the PUSH may not be
     * completed if no PUSH is required by this task and the PUSH is
     * actually done by another task, so we need to check if the data is
     * actually ready to use
     */
    assert(gpu_stream == gpu_device->exec_stream[0]);
    task = gtask->ec;
    PARSEC_DEBUG_VERBOSE(19, parsec_gpu_output_stream,
                         "GPU[%d:%s]: parsec_device_callback_complete_push, PUSH of %s",
                         gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task));

    for( i = 0; i < task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == task->data[i].data_in ) continue;
        /* We also don't push back non-parsec-owned copies */
        if(NULL != task->data[i].data_out &&
           0 == (task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)) continue;

        flow = gtask->flow[i];
        assert( flow );
        assert( flow->flow_index == i );
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if(PARSEC_DATA_STATUS_UNDER_TRANSFER == task->data[i].data_out->data_transfer_status ) {
            /* only the task who did the PUSH can modify the status */
            parsec_atomic_lock(&task->data[i].data_out->original->lock);
            parsec_data_end_transfer_ownership_to_copy(task->data[i].data_out->original,
                                                       gpu_device->super.device_index,
                                                       flow->flow_flags);
            task->data[i].data_out->data_transfer_status = PARSEC_DATA_STATUS_COMPLETE_TRANSFER;

            parsec_data_copy_t* source = gtask->sources[i];
            parsec_device_gpu_module_t *src_device =
                    (parsec_device_gpu_module_t*)parsec_mca_device_get( source->device_index );
            if (task->data[i].data_in->flags & PARSEC_DATA_FLAG_EVICTED) {
                /**
                 * The device copy had been evicted to the host and brought back in.
                 * If this is the only device on which that data is used we can release
                 * the host memory back to the application. If there are other devices
                 * we cannot release the host memory because the data may actually be used
                 * by a host task (e.g., after being sent there from a different device)
                 * or be used as input to the other device.
                 */
                parsec_data_copy_t *cpu_copy = task->data[i].data_out->original->device_copies[0];
                parsec_data_copy_t *gpu_copy = task->data[i].data_out;
                parsec_data_t *original = task->data[i].data_out->original;
                /* release host memory if requested */
                if (cpu_copy->device_private != NULL &&
                    cpu_copy->release_cb != NULL) {
                    bool may_release = true;
                    /* check if there are any other device copies */
                    for (uint32_t i = 1; i < parsec_nb_devices; ++i) {
                        parsec_data_copy_t *copy = original->device_copies[i];
                        if (NULL != copy && copy != gpu_copy) {
                            may_release = false;
                            break;
                        }
                    }
                    if (may_release) {
                        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                            "GPU[%d:%s]: CPU copy %p [ref_count %d] memory %p eager release",
                                            gpu_device->super.device_index, gpu_device->super.name,
                                            cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->device_private);
                        cpu_copy->release_cb(cpu_copy, 0);
                        cpu_copy->device_private = NULL;
                        cpu_copy->version = 0;
                        cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
                        gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_EXCLUSIVE;
                    }
                }
                task->data[i].data_in->flags ^= PARSEC_DATA_FLAG_EVICTED;
            }

            parsec_atomic_unlock(&task->data[i].data_out->original->lock);

#if defined(PARSEC_PROF_TRACE)
            if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_IN) {
                PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                       parsec_gpu_movein_key_end,
                                       (int64_t)(int64_t)task->data[i].data_out->device_private,
                                       gpu_device->super.device_index,
                                       NULL);
            }
#endif
            if( PARSEC_DEV_IS_GPU(src_device->super.type) ) {
                int om;
                while(1) {
                    /* There are two ways out:
                     *   either we exit with om = 0, and then nobody was managing src_device,
                     *   and nobody can start managing src_device until we make it change from -1 to 0
                     *   (but anybody who has work to do will wait until that happens), or
                     *   we exit with om > 0, then there is a manager for that thread, and we have
                     *   increased mutex to warn the manager that there is another task for it to do.
                     */
                    om = src_device->mutex;
                    if(om == 0) {
                        /* Nobody at the door, let's try to lock the door */
                        if( parsec_atomic_cas_int32(&src_device->mutex, 0, -1) )
                            break;
                        continue;
                    }
                    if(om < 0 ) {
                        /* Damn, another thread is also trying to do an atomic operation on src_device,
                         * we give it some time and try again */
                        struct timespec delay;
                        delay.tv_nsec = 100;
                        delay.tv_sec = 0;
                        nanosleep(&delay, NULL);
                        continue;
                    }
                    /* There is a manager, let's try to reserve another task to do.
                     * If that fails, the manager may have leaved, try a gain. */
                    if( parsec_atomic_cas_int32(&src_device->mutex, om, om+1) )
                        break;
                }
                if( 0 == om ) {
                    int rc;
                    /* Nobody is at the door to handle that event on the source of that data...
                     * we do the command directly */
                    parsec_atomic_lock( &source->original->lock );
                    int readers = parsec_atomic_fetch_sub_int32(&source->readers, 1) - 1;
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tExecuting D2D transfer complete for copy %p [ref_count %d] for "
                                         "device %s -- readers now %d",
                                         gpu_device->super.device_index, gpu_device->super.name, source,
                                         source->super.super.obj_reference_count, src_device->super.name,
                                         readers);
                    assert(readers >= 0);
                    if(0 == readers && 0 != (source->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)) {
                        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                             "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available",
                                             gpu_device->super.device_index, gpu_device->super.name, source,
                                             source->super.super.obj_reference_count);
                        parsec_list_item_ring_chop((parsec_list_item_t*)source);
                        PARSEC_LIST_ITEM_SINGLETON(source);
                        parsec_lifo_push(&src_device->gpu_mem_inbox, (parsec_list_item_t*)source);
                        src_device->data_avail_epoch++;
                    }
                    parsec_atomic_unlock( &source->original->lock );
                    /* Notify any waiting thread that we're done messing with that device structure */
                    rc = parsec_atomic_cas_int32(&src_device->mutex, -1, 0); (void)rc;
                    assert(rc);
                } else {
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                         "GPU[%d:%s]:\tSending D2D transfer complete command to %s for copy %p "
                                         "[ref_count %d] -- readers is still %d",
                                         gpu_device->super.device_index, gpu_device->super.name, src_device->super.name, source,
                                         source->super.super.obj_reference_count, source->readers);
                    parsec_device_send_transfercomplete_cmd_to_device(source,
                                                                      (parsec_device_module_t*)gpu_device,
                                                                      (parsec_device_module_t*)src_device);
                }
            }
            continue;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tparsec_device_callback_complete_push, PUSH of %s: task->data[%d].data_out = %p [ref_count = %d], and %s because transfer_status is %d",
                             gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task),
                             i, task->data[i].data_out, task->data[i].data_out->super.super.obj_reference_count,
                             (task->data[i].data_out->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER) ? "all is good" : "Assertion",
                             task->data[i].data_out->data_transfer_status);
        if( task->data[i].data_out->data_transfer_status == PARSEC_DATA_STATUS_UNDER_TRANSFER ) {  /* data is not ready */
            /**
             * As long as we have only one stream to push the data on the GPU we should never
             * end up in this case. Remove previous assert if changed.
             */
            return PARSEC_HOOK_RETURN_ERROR;
        }
    }
    gtask->complete_stage = NULL;

    if( PARSEC_GPU_TASK_TYPE_PREFETCH == gtask->task_type ) {
        parsec_data_copy_t *gpu_copy = task->data[0].data_out;
#if defined(PARSEC_DEBUG_NOISIER)
        char tmp[MAX_TASK_STRLEN];
        assert(NULL != gpu_copy);
        if( NULL != gpu_copy->original->dc )
            gpu_copy->original->dc->key_to_string(gpu_copy->original->dc, gpu_copy->original->key, tmp, MAX_TASK_STRLEN);
        else
            snprintf(tmp, MAX_TASK_STRLEN, "unbound data");
#endif
        PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tPrefetch for data copy %p [ref_count %d] (%s) done. readers = %d, device_index = %d, version = %d, flags = %d, state = %d, data_transfer_status = %d",
                             gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count,
                             tmp,
                             gpu_copy->readers, gpu_copy->device_index, gpu_copy->version,
                             gpu_copy->flags, gpu_copy->coherency_state, gpu_copy->data_transfer_status);
        int readers = parsec_atomic_fetch_sub_int32(&gpu_copy->readers, 1);
        if( 0 == readers ) {
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tMake copy %p [ref_count %d] available after prefetch from gpu_task %p, ec %p",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, gtask, gtask->ec);
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
        (void)parsec_device_release_resources_prefetch_task(gpu_device, gpu_task);
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Returns true if tasks are in flight or pending on @p stream.
 */
static inline bool
parsec_device_stream_has_tasks( parsec_gpu_exec_stream_t* stream )
{
    return (NULL != stream->tasks[stream->end]) ||
           !parsec_list_nolock_is_empty(stream->fifo_pending) ||
           !parsec_list_nolock_is_empty(stream->fifo_batch);
}

/**
 * Retrieve the task (or task ring) attached to the oldest event of @p stream,
 * if that event has completed.
 *
 * @returns 1 if the event completed (*task is set), 0 if there is no event or the
 * oldest event is still pending, and PARSEC_HOOK_RETURN_AGAIN if the event could
 * not be queried.
 */
static inline int
parsec_device_stream_pop_completed( parsec_device_gpu_module_t* gpu_device,
                                    parsec_gpu_exec_stream_t* stream,
                                    parsec_gpu_task_t** task )
{
    int rc;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
#endif

    *task = NULL;
    if( NULL == stream->tasks[stream->end] )
        return 0;
    rc = gpu_device->event_query(gpu_device, stream, stream->end);
    if( 1 != rc ) {
        return (0 == rc) ? 0 : PARSEC_HOOK_RETURN_AGAIN;
    }
    *task = stream->tasks[stream->end];
    PARSEC_DEBUG_VERBOSE(19, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Completed %s on stream %s{%p}",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(task_str, MAX_TASK_STRLEN, (*task)->ec),
                         stream->name, (void*)stream);
    stream->tasks[stream->end] = NULL;
    stream->end = (stream->end + 1) % stream->max_events;
    return 1;
}

/**
 * Record an event on @p stream and attach @p task (a single task or a task ring)
 * to it. The caller must make sure there is room on the stream.
 */
static inline void
parsec_device_stream_record( parsec_device_gpu_module_t* gpu_device,
                             parsec_gpu_exec_stream_t* stream,
                             parsec_gpu_task_t* task )
{
    int rc;

    assert( NULL == stream->tasks[stream->start] );
    rc = gpu_device->event_record(gpu_device, stream, stream->start);
    assert(PARSEC_SUCCESS == rc); (void)rc;
    stream->tasks[stream->start] = task;
    stream->start = (stream->start + 1) % stream->max_events;
    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Submitted %s(task %p) on stream %s{%p}",
                         gpu_device->super.device_index, gpu_device->super.name,
                         (NULL != task->ec) ? task->ec->task_class->name : "(released)", (void*)task->ec,
                         stream->name, (void*)stream);
}

static inline void
parsec_device_stream_trace_end( parsec_gpu_exec_stream_t* stream,
                                parsec_gpu_task_t* task )
{
#if defined(PARSEC_PROF_TRACE)
    if( stream->prof_event_track_enable ) {
        if( task->prof_key_end != -1 ) {
            PARSEC_PROFILING_TRACE(stream->profiling, task->prof_key_end, task->prof_event_id, task->prof_tp_id, NULL);
        }
    }
#else
    (void)stream; (void)task;
#endif /* (PARSEC_PROF_TRACE) */
}

/* Append @p task (a single task or a task ring) at the end of @p *ring. */
static inline void
parsec_gpu_task_ring_append( parsec_gpu_task_t** ring,
                             parsec_gpu_task_t* task )
{
    if( NULL == *ring ) {
        *ring = task;
    } else {
        parsec_list_item_ring_merge(&(*ring)->list_item, &task->list_item);
    }
}

/**
 * Detach the unit of work @p iter from its ring and return the next unit, or NULL
 * if @p iter was the last one. When batching, each task of the ring is a unit.
 * Otherwise the whole ring is a single unit: a batched submit function may have
 * turned the task into a ring that must be kept together (see
 * parsec_gpu_task_collect_batch()).
 */
static inline parsec_gpu_task_t*
parsec_gpu_task_ring_next_unit( parsec_gpu_task_t* iter,
                                bool batch )
{
    parsec_gpu_task_t *next;

    if( !batch )
        return NULL;
    next = (parsec_gpu_task_t*)parsec_list_item_ring_chop((parsec_list_item_t*)iter);
    PARSEC_LIST_ITEM_SINGLETON(iter);
    return next;
}

/**
 * Extract the next ring of tasks to progress from the pending queue of @p stream.
 * Without batching, the ring contains a single task. With batching, the ring
 * gathers up to PARSEC_DEVICE_MAX_BATCH_SIZE pending tasks whose selected
 * incarnation allows batching on @p gpu_device and that share the incarnation
 * of the first task. Pending tasks that do not match are skipped and left in
 * the queue, in order, for the following rings. A task that is not batchable
 * always forms a ring of its own.
 */
static inline parsec_gpu_task_t*
parsec_device_stream_grab_ring( parsec_device_gpu_module_t* gpu_device,
                                parsec_gpu_exec_stream_t* stream,
                                bool batch )
{
    parsec_list_t *fifo_pending = stream->fifo_pending;
    parsec_list_item_t *item, *next;
    parsec_gpu_task_t *ring, *task;
    int nb_tasks = 1;

    ring = (parsec_gpu_task_t*)parsec_list_nolock_pop_front(fifo_pending);
    if( NULL == ring )
        return NULL;
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)ring);

    if( !batch ||
        !parsec_gpu_task_selected_chore_allows_batch(ring->ec, (parsec_device_module_t*)gpu_device) )
        return ring;

    for( item = (parsec_list_item_t*)fifo_pending->ghost_element.list_next;
         (item != &fifo_pending->ghost_element) && (nb_tasks < stream->max_batch);
         item = next ) {
        next = (parsec_list_item_t*)item->list_next;
        task = (parsec_gpu_task_t*)item;
        if( !parsec_gpu_task_selected_chore_allows_batch(task->ec, (parsec_device_module_t*)gpu_device) ||
            !parsec_gpu_task_same_chore(task, ring) ) {
            /* the task does not match the current batch, leave it for a following one */
            continue;
        }
        (void)parsec_list_nolock_remove(fifo_pending, item);
        PARSEC_LIST_ITEM_SINGLETON(item);
        parsec_list_item_ring_push(&ring->list_item, item);
        nb_tasks++;
    }
    return ring;
}

/**
 * Returns true if the batchable tasks pending on the execution stream @p stream
 * are delayed: the stream still has work in flight. These tasks could not start
 * before that work completed anyway, and delaying their submission allows the
 * submit function to batch them with the tasks arriving in the meantime.
 */
static inline bool
parsec_device_stream_delay_batch( parsec_gpu_exec_stream_t* stream )
{
    return parsec_gpu_exec_delay_batch && (NULL != stream->tasks[stream->end]);
}

/**
 * Extract the next task to submit from the pending queues of the execution
 * stream @p stream. Non-batchable tasks can always be submitted, batchable tasks
 * only if they are not delayed (see parsec_device_stream_delay_batch()). If both
 * kinds are eligible, the task with the highest priority is selected, and the
 * oldest one in case of a tie.
 */
static inline parsec_gpu_task_t*
parsec_device_exec_stream_grab( parsec_gpu_exec_stream_t* stream )
{
    parsec_gpu_task_t *task = NULL, *batch_task = NULL;
    parsec_list_t *queue = stream->fifo_pending;
    parsec_list_item_t *item;

    if( !parsec_list_nolock_is_empty(stream->fifo_pending) )
        task = (parsec_gpu_task_t*)stream->fifo_pending->ghost_element.list_next;
    if( !parsec_list_nolock_is_empty(stream->fifo_batch) && !parsec_device_stream_delay_batch(stream) )
        batch_task = (parsec_gpu_task_t*)stream->fifo_batch->ghost_element.list_next;
    if( (NULL != batch_task) &&
        ((NULL == task) ||
         (batch_task->priority > task->priority) ||
         ((batch_task->priority == task->priority) && (batch_task->heap_seq < task->heap_seq))) ) {
        queue = stream->fifo_batch;
    } else if( NULL == task ) {
        return NULL;
    }
    item = parsec_list_nolock_pop_front(queue);
    PARSEC_LIST_ITEM_SINGLETON(item);
    return (parsec_gpu_task_t*)item;
}

/**
 * If the execution stream @p stream is idle and has no pending tasks, take over
 * a batchable task delayed on another execution stream, so that it does not wait
 * for that stream to drain. Once submitted, the task can collect the other
 * delayed tasks into its batch (see parsec_gpu_task_collect_batch()).
 */
static void
parsec_device_stream_take_delayed( parsec_device_gpu_module_t* gpu_device,
                                   parsec_gpu_exec_stream_t* stream )
{
    parsec_gpu_exec_stream_t *other;
    parsec_list_item_t *item;

    if( !parsec_gpu_exec_delay_batch ||
        (NULL != stream->tasks[stream->end]) ||
        !parsec_list_nolock_is_empty(stream->fifo_pending) ||
        !parsec_list_nolock_is_empty(stream->fifo_batch) )
        return;
    for( int i = 2; i < gpu_device->num_exec_streams; i++ ) {
        other = gpu_device->exec_stream[i];
        if( (other == stream) || !parsec_device_stream_delay_batch(other) ||
            parsec_list_nolock_is_empty(other->fifo_batch) )
            continue;
        item = parsec_list_nolock_pop_front(other->fifo_batch);
        PARSEC_LIST_ITEM_SINGLETON(item);
        parsec_list_nolock_push_back(stream->fifo_batch, item);
        return;
    }
}

/**
 * Release the tasks in @p ring that have been withdrawn from the device by a
 * data movement stage while other tasks are still managed by the device (so
 * that this thread remains the manager of the device).
 */
static void
parsec_device_release_withdrawn_tasks( parsec_device_gpu_module_t* gpu_device,
                                       parsec_gpu_task_t* ring )
{
    parsec_gpu_task_t *iter = ring, *next;
    int cnt = 0;

    do {
        next = (parsec_gpu_task_t*)parsec_list_item_ring_chop((parsec_list_item_t*)iter);
        PARSEC_LIST_ITEM_SINGLETON(iter);
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]: gpu_task %p freed",
                             gpu_device->super.device_index, gpu_device->super.name, iter);
        if( NULL != iter->release_device_task ) {
            iter->release_device_task(iter);
        }
        cnt++;
        iter = next;
    } while( NULL != iter );
    /* The tasks still managed by the device keep the count positive. Beware:
     * prefetch tasks are pushed into the pending queue without being accounted
     * for in the mutex, so the count cannot be checked here. */
    (void)parsec_atomic_fetch_sub_int32(&gpu_device->mutex, cnt);
}

/**
 * Sort a task into the ring matching the return code @p rc of one of its stage
 * functions: tasks that completed the stage go into @p done_ring, tasks withdrawn
 * from the device (PARSEC_HOOK_RETURN_ASYNC) into @p async_ring, and on any other
 * error the task is kept in @p error_task (only the first error is kept).
 */
static inline void
parsec_device_sort_stage_result( int rc,
                                 parsec_gpu_task_t* task,
                                 parsec_gpu_task_t** done_ring,
                                 parsec_gpu_task_t** async_ring,
                                 parsec_gpu_task_t** error_task,
                                 int* error_rc )
{
    if( PARSEC_HOOK_RETURN_DONE == rc ) {
        parsec_gpu_task_ring_append(done_ring, task);
    } else if( PARSEC_HOOK_RETURN_ASYNC == rc ) {
        parsec_gpu_task_ring_append(async_ring, task);
    } else if( NULL == *error_task ) {
        *error_task = task;
        *error_rc = rc;
    }
}

/**
 * This function tries to progress a stream, by picking up ready tasks
 * and applying the progress function. The tasks to be progressed are
 * always taken from the front of the waiting queue, even when a task
 * has been specified as an input argument.
 *
 * If @p batch is false, tasks are handed one at a time to @p progress_fct, and
 * its return value is the outcome for the task. This is the mode used for the
 * execution streams: the submit function of the task is in charge of batching
 * kernels, and may turn the task into a task ring (see
 * parsec_gpu_task_collect_batch()) which is then handled as a single unit.
 *
 * If @p batch is true, rings of tasks sharing the same batchable incarnation
 * (see parsec_device_stream_grab_ring()) are handed to @p progress_fct in a
 * single call, so that the progress function can, e.g., merge the data
 * transfers of all tasks into batched copies. The progress function stores
 * the outcome for each task of the ring in its last_status. At most one ring
 * is submitted per call, so that the caller can progress all stages evenly.
 *
 * Outcomes follow the parsec_gpu_step_function_t convention: a positive value
 * if work was queued on the stream, 0 if nothing was queued, or a negative
 * parsec_hook_return_t. All tasks of a ring that queued work, or requested to
 * be retried (PARSEC_HOOK_RETURN_AGAIN), share a single event on the stream.
 *
 * Beware: this function does not generate errors by itself, instead
 * it propagates upward the return code of the progress function.
 *
 * @returns
 *   PARSEC_HOOK_RETURN_DONE: *out_task is the ring of tasks that completed the
 *                            stage (or NULL).
 *   PARSEC_HOOK_RETURN_ASYNC: *out_task is the ring of tasks withdrawn from the
 *                             device, and no task completed the stage.
 *   PARSEC_HOOK_RETURN_NEXT: some tasks were delayed because of physical
 *                            constraints (e.g., not enough memory on the device),
 *                            and no task completed the stage. *out_task is NULL.
 *   PARSEC_HOOK_RETURN_AGAIN: the stream could not be queried. *out_task is NULL.
 *   other errors: *out_task is the task that raised the error.
 */
static int
parsec_device_progress_stream( parsec_device_gpu_module_t* gpu_device,
                               parsec_gpu_exec_stream_t* stream,
                               parsec_gpu_step_function_t progress_fct,
                               parsec_gpu_task_t* task,
                               parsec_gpu_task_t** out_task,
                               bool batch )
{
    parsec_gpu_task_t *done_ring = NULL;   /* tasks that completed the stage */
    parsec_gpu_task_t *async_ring = NULL;  /* tasks withdrawn from the device */
    parsec_gpu_task_t *next_ring = NULL;   /* tasks delayed by physical constraints */
    parsec_gpu_task_t *again_ring = NULL;  /* tasks to be retried on this stream */
    parsec_gpu_task_t *error_task = NULL;
    parsec_gpu_task_t *ring = NULL, *event_ring, *iter, *next;
    int rc, error_rc = PARSEC_HOOK_RETURN_DONE;

    /* We prioritize internal tasks by pushing them to the front of the queue.
     * Kernels will be handled in the order in which they arrive. */
    if( NULL != task ) {
        parsec_gpu_stream_push_pending(gpu_device, stream, task);
    }
    *out_task = NULL;

    rc = parsec_device_stream_pop_completed(gpu_device, stream, &ring);
    if( rc < 0 ) {
        return rc;
    }
    if( 1 == rc ) {
        /* Complete the stage for all tasks sharing the event, in submission order */
        iter = ring;
        do {
            next = parsec_gpu_task_ring_next_unit(iter, batch);
            parsec_device_stream_trace_end(stream, iter);
            if( PARSEC_HOOK_RETURN_AGAIN == iter->last_status ) {
                /* we can now reschedule the task on the same execution stream */
                PARSEC_DEBUG_VERBOSE(2, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: GPU task %p[%p] is ready to be rescheduled on the same GPU device and same stream",
                                     gpu_device->super.device_index, gpu_device->super.name, (void*)iter, (void*)iter->ec);
                parsec_gpu_task_ring_append(&again_ring, iter);
            } else {
                assert( PARSEC_HOOK_RETURN_ASYNC != iter->last_status );
                rc = PARSEC_HOOK_RETURN_DONE;
                if( iter->complete_stage )
                    rc = iter->complete_stage(gpu_device, &iter, stream);
                /* the task can be withdrawn by the system */
                if( NULL != iter )
                    parsec_device_sort_stage_result(rc, iter, &done_ring, &async_ring, &error_task, &error_rc);
            }
            iter = next;
        } while( NULL != iter );

        /* Hand the completed tasks to the upper level right away. */
        if( (NULL != done_ring) || (NULL != async_ring) || (NULL != error_task) ) {
            if( NULL != again_ring ) {
                /* only possible when batching, so these are single tasks */
                assert(batch);
                parsec_list_nolock_chain_front(stream->fifo_pending, &again_ring->list_item);
            }
            /* Execution streams delaying batchable tasks submit them as soon as the
             * stream drained, instead of waiting for the next visit of the stream. */
            if( batch || !parsec_gpu_exec_delay_batch || (NULL != error_task) )
                goto complete;
        }
    }

    /* Retry the tasks that requested it first (they belong to the same batch,
     * or form the unit created by the submit function), otherwise take tasks
     * from the pending queue. When batching, a single ring is processed per
     * call. Without batching, tasks are submitted as long as there is room on
     * the stream. Stop as soon as tasks are delayed or an error occurred. */
    ring = again_ring;
    while( (NULL == stream->tasks[stream->start]) &&
           (NULL == next_ring) && (NULL == error_task) ) {
        if( NULL == ring ) {
            ring = batch ? parsec_device_stream_grab_ring(gpu_device, stream, batch)
                         : parsec_device_exec_stream_grab(stream);
        }
        if( NULL == ring ) {  /* No tasks, we're done */
            break;
        }

        rc = progress_fct( gpu_device, ring, stream );
        if( !batch ) {
            ring->last_status = rc;
        }

        event_ring = NULL;
        iter = ring;
        ring = NULL;
        do {
            next = parsec_gpu_task_ring_next_unit(iter, batch);
            rc = iter->last_status;
            if( (rc > 0) || (PARSEC_HOOK_RETURN_AGAIN == rc) ||
                ((0 == rc) && !parsec_device_skip_empty_events) ) {
                /**
                 * Do not skip the gpu event generation. The problem is that some of the inputs
                 * might be in the pipe of being transferred to the GPU. If we activate this task
                 * too early, it might get executed before the data is available on the GPU.
                 * Obviously, this lead to incorrect results.
                 * The progress result is kept in last_status while the task is parked behind
                 * the event. In particular, AGAIN is not an error: it means the event completion
                 * path must retry this same stage instead of advancing the task.
                 */
                parsec_gpu_task_ring_append(&event_ring, iter);
            } else if( 0 == rc ) {
                /* Nothing was added on the stream for this task, skip the event */
                parsec_device_stream_trace_end(stream, iter);
                rc = PARSEC_HOOK_RETURN_DONE;
                if( iter->complete_stage )
                    rc = iter->complete_stage(gpu_device, &iter, stream);
                if( NULL != iter )
                    parsec_device_sort_stage_result(rc, iter, &done_ring, &async_ring, &error_task, &error_rc);
            } else if( PARSEC_HOOK_RETURN_NEXT == rc ) {
                /* We are running into physical constraints and need to delay the resubmission
                 * of this task as much as possible, but without loosing track of it. */
                parsec_gpu_task_ring_append(&next_ring, iter);
            } else {
                /* Withdrawn from the device, or something else is going on with this task:
                 * return it to the upper level for final decision on its fate. */
                parsec_device_sort_stage_result(rc, iter, &done_ring, &async_ring, &error_task, &error_rc);
            }
            iter = next;
        } while( NULL != iter );

        if( NULL != event_ring ) {
            parsec_device_stream_record(gpu_device, stream, event_ring);
        }
        /* When batching, move on to the next stage once a ring has been processed, so
         * that all stages progress evenly. In any case, hand the tasks that completed
         * the stage to the upper level right away. */
        if( batch || (NULL != done_ring) )
            break;
    }

    assert(NULL == ring);

  complete:
    if( NULL != next_ring ) {
        /* Dont reorder the push_back, see above. */
        iter = next_ring;
        do {
            next = parsec_gpu_task_ring_next_unit(iter, batch);
            parsec_gpu_stream_push_pending(gpu_device, stream, iter);
            iter = next;
        } while( NULL != iter );
    }
    if( NULL != error_task ) {
        /* The device will be disabled, no need to take care of the other tasks */
        *out_task = error_task;
        return error_rc;
    }
    if( NULL != async_ring ) {
        if( NULL == done_ring ) {
            *out_task = async_ring;
            return PARSEC_HOOK_RETURN_ASYNC;
        }
        /* The tasks in done_ring keep this thread managing the device */
        parsec_device_release_withdrawn_tasks(gpu_device, async_ring);
    }
    if( NULL != done_ring ) {
        *out_task = done_ring;
        return PARSEC_HOOK_RETURN_DONE;
    }
    return (NULL != next_ring) ? PARSEC_HOOK_RETURN_NEXT : PARSEC_HOOK_RETURN_DONE;
}

/**
 *  @brief This function prepare memory on the target device for all the inputs and output
 *  of the task, and then initiate the necessary copies from the best location of the input
 *  data. The best location is defined as any other accelerator that has the same version
 *  of the data (taking advantage of faster accelerator-to-accelerator connectors, such as
 *  NVLink), or from the CPU memory if no other candidate is found.
 *  The copies are not issued by this function: the flows to be transferred are
 *  marked in @p transfer_mask, and the caller must issue the transfers for them,
 *  even if an error is returned.
 *
 *  @returns
 *     a positive number: the number of data to be moved.
 *     PARSEC_HOOK_RETURN_DONE: all data is already available on the device.
 *     PARSEC_HOOK_RETURN_NEXT: no room on the device for the data of this task, and
 *                              no memory has been released since the last attempt.
 *     PARSEC_HOOK_RETURN_AGAIN: retry once the operations queued on the stream completed.
 *     PARSEC_HOOK_RETURN_ASYNC: the (prefetch) task has been released.
 */
static int
parsec_device_kernel_push_task( parsec_device_gpu_module_t      *gpu_device,
                                parsec_gpu_task_t               *gpu_task,
                                parsec_gpu_exec_stream_t        *gpu_stream,
                                uint32_t                        *transfer_mask )
{
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t *flow;
    int i, ret = 0, how_many = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    /* If no changes were made to the available memory dont waste time, the reservation
     * would fail again. Report the lack of memory every time (and not only on the
     * attempt that failed), so that the scheduler can trigger evictions whenever no
     * other task progresses. */
    if( gpu_task->last_data_check_epoch == gpu_device->data_avail_epoch )
        return PARSEC_HOOK_RETURN_NEXT;
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Try to Push %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task) );

    if( PARSEC_GPU_TASK_TYPE_PREFETCH == gpu_task->task_type ) {
        if( NULL == gpu_task->ec->data[0].data_in->original ) {
            /* The PREFETCH order comes after the copy was detached and released, ignore it */
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s has been released already, destroying prefetch request",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
            parsec_device_release_resources_prefetch_task(gpu_device, &gpu_task);
            return PARSEC_HOOK_RETURN_ASYNC;
        }
        if( NULL != gpu_task->ec->data[0].data_in->original->device_copies[gpu_device->super.device_index] &&
            gpu_task->ec->data[0].data_in->original->owner_device == gpu_device->super.device_index ) {
            /* There is already a copy of this data in the GPU */
            PARSEC_DEBUG_VERBOSE(3, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: %s data_copy at index %d is %p, destroying prefetch request",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task),
                                 gpu_device->super.device_index,
                                 gpu_task->ec->data[0].data_in->original->device_copies[gpu_device->super.device_index]);
            parsec_device_release_resources_prefetch_task(gpu_device, &gpu_task);
            return PARSEC_HOOK_RETURN_ASYNC;
        }
    }

    /* Do we have enough available memory on the GPU to hold the input and output data ? */
    ret = parsec_device_data_reserve_space( gpu_device, gpu_task );
    if( ret < 0 ) {
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch;
        return ret;
    }

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {

        flow = gpu_task->flow[i];
        /* Skip CTL flows */
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* If there is already a GPU data copy (set by reserve_device_space), and this copy
         * is not parsec-owned, don't stage in */
        if( NULL != this_task->data[i].data_out &&
            (0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) ) continue;

        assert( NULL != parsec_data_copy_get_ptr(this_task->data[i].data_in)
             || NULL != this_task->data[i].data_in->alloc_cb );

        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\t\tIN  Data of %s <%x> on GPU",
                             gpu_device->super.device_index, gpu_device->super.name, flow->name,
                             this_task->data[i].data_out->original->key);
        ret = parsec_device_data_stage_in( gpu_device, flow,
                                           &(this_task->data[i]), gpu_task, gpu_stream, transfer_mask );
        if( ret < 0 ) {
            /* Transfers already decided for earlier flows are left in *transfer_mask:
             * those flows are already marked UNDER_TRANSFER and the caller must issue
             * a matching physical copy for them before propagating this error/retry. */
            return ret;
        }
        how_many += ret;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Push task %s DONE",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
    gpu_task->complete_stage = parsec_device_callback_complete_push;
#if defined(PARSEC_PROF_TRACE)
    gpu_task->prof_key_end = -1; /* We do not log that event as the completion of this task */
#endif
    return how_many;
}

/**
 * Issue the transfers decided for the tasks in @p tasks, as recorded in the flow
 * masks @p masks, on @p gpu_stream. The transfers of all tasks relying on the
 * @p default_stage function are merged into batched copies, the other tasks
 * issue their transfers through their own stage function. Tasks whose transfers
 * could not be issued get @p error_rc as last_status.
 */
static void
parsec_device_issue_transfers( parsec_device_gpu_module_t  *gpu_device,
                               parsec_gpu_exec_stream_t    *gpu_stream,
                               parsec_gpu_task_t          **tasks,
                               uint32_t                    *masks,
                               int                          nb_tasks,
                               bool                         stage_in,
                               int                          error_rc )
{
    parsec_device_copy_batch_t batch;
    parsec_device_gpu_module_t *dev = gpu_device;
    parsec_stage_in_function_t *stage_fct;
    int rc;

    batch.nb_items = 0;
    batch.nb_tasks = 0;
    for( int t = 0; t < nb_tasks; t++ ) {
        parsec_gpu_task_t *gpu_task = tasks[t];
        if( 0 == masks[t] ) continue;

        stage_fct = stage_in ? gpu_task->stage_in : gpu_task->stage_out;
        if( stage_fct == (stage_in ? parsec_default_gpu_stage_in : parsec_default_gpu_stage_out) ) {
            if( batch.nb_items + MAX_PARAM_COUNT > PARSEC_DEVICE_MAX_BATCH_COPIES ) {
                parsec_device_copy_batch_flush(gpu_device, &batch, gpu_stream, error_rc);
            }
            if( stage_in ) {
                parsec_default_gpu_stage_in_collect(gpu_task, masks[t], batch.dsts, batch.srcs,
                                                    batch.sizes, batch.directions, &batch.nb_items, &dev);
                rc = PARSEC_HOOK_RETURN_DONE;
            } else {
                rc = parsec_default_gpu_stage_out_collect(gpu_task, masks[t], batch.dsts, batch.srcs,
                                                          batch.sizes, batch.directions, &batch.nb_items, &dev);
            }
            /* all copies are issued by the device managing the tasks */
            assert(dev == gpu_device);
            batch.tasks[batch.nb_tasks++] = gpu_task;
            if( PARSEC_HOOK_RETURN_DONE != rc ) {
                gpu_task->last_status = error_rc;
            }
        } else if( NULL != stage_fct ) {
            rc = stage_fct(gpu_task, masks[t], gpu_stream);
            if( PARSEC_SUCCESS != rc ) {
                parsec_warning( "GPU[%d:%s]: gpu_task->%s rc=%d @%s:%d\n"
                                "\ttask %s transfer_mask=0x%x\n",
                                gpu_device->super.device_index, gpu_device->super.name,
                                stage_in ? "stage_in" : "stage_out", rc, __func__, __LINE__,
                                gpu_task->ec->task_class->name, masks[t]);
                gpu_task->last_status = error_rc;
            }
        }
    }
    parsec_device_copy_batch_flush(gpu_device, &batch, gpu_stream, error_rc);
    (void)dev;
}

/**
 *  @brief Stage in the data of a ring of tasks. Each task of the ring is handled as
 *  described in parsec_device_kernel_push_task(), and the outcome for each task is
 *  stored in its last_status. The transfers of all tasks are then issued together,
 *  merging the copies of the tasks using the default stage_in function into batched
 *  copies.
 *
 *  @returns PARSEC_HOOK_RETURN_DONE, per-task results are in last_status.
 */
static int
parsec_device_kernel_push( parsec_device_gpu_module_t      *gpu_device,
                           parsec_gpu_task_t               *gpu_tasks,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_gpu_task_t *tasks[PARSEC_DEVICE_MAX_BATCH_SIZE];
    uint32_t masks[PARSEC_DEVICE_MAX_BATCH_SIZE];
    parsec_gpu_task_t *iter = gpu_tasks;
    int nb_tasks = 0;
    bool out_of_memory = false;

    /* Collect the tasks first, the stage function may release some of them. */
    do {
        assert(nb_tasks < PARSEC_DEVICE_MAX_BATCH_SIZE);
        tasks[nb_tasks++] = iter;
        iter = (parsec_gpu_task_t*)iter->list_item.list_next;
    } while( iter != gpu_tasks );

    for( int t = 0; t < nb_tasks; t++ ) {
        masks[t] = 0;
        if( out_of_memory ) {
            /* All tasks of the ring share the same incarnation: once a task fails to
             * find room on the device the following ones would most likely fail too,
             * don't waste time scanning the LRU for them. */
            tasks[t]->last_status = PARSEC_HOOK_RETURN_NEXT;
            continue;
        }
        tasks[t]->last_status = parsec_device_kernel_push_task(gpu_device, tasks[t], gpu_stream, &masks[t]);
        out_of_memory = (PARSEC_HOOK_RETURN_NEXT == tasks[t]->last_status);
    }
    parsec_device_issue_transfers(gpu_device, gpu_stream, tasks, masks, nb_tasks,
                                  true, PARSEC_HOOK_RETURN_ERROR);
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Prepare a task for execution on the GPU. Basically, does some upstream initialization,
 * setup the profiling information and then calls directly into the task submission body. Upon
 * return from the body handle the state machine of the task, taking care of the special cases
 * such as AGAIN and ASYNC.
 * @returns An error if anything unexpected came out of the task submission body, otherwise
 */
static int
parsec_device_kernel_exec( parsec_device_gpu_module_t      *gpu_device,
                           parsec_gpu_task_t               *gpu_task,
                           parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_advance_task_function_t progress_fct = gpu_task->submit;
    parsec_task_t* this_task = gpu_task->ec;
    int rc;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]:\tEnqueue on device %s stream %s"     ,
                         gpu_device->super.device_index, gpu_device->super.name, parsec_task_snprintf(tmp, MAX_TASK_STRLEN,
                         (parsec_task_t *) this_task), gpu_stream->name);
#endif /* defined(PARSEC_DEBUG_NOISIER) */
#if defined(PARSEC_PROF_TRACE)
    if (gpu_stream->prof_event_track_enable &&
        (0 == gpu_task->prof_key_end)) {
        parsec_task_class_t* tc = (parsec_task_class_t*)this_task->task_class;
        PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,
                               PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                          tc->task_class_id),
                               (parsec_task_t *) this_task, 1);
        gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool, tc->task_class_id);
        gpu_task->prof_event_id = tc->key_functions->key_hash(
                                        tc->make_key(this_task->taskpool, ((parsec_task_t *) this_task)->locals), NULL);
        gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;
    }
#endif /* PARSEC_PROF_TRACE */

#if defined(PARSEC_DEBUG_PARANOID)
    const parsec_flow_t *flow;
    for( int i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags)) continue;
        if( 0 == (this_task->data[i].data_out->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
        assert(this_task->data[i].data_out->data_transfer_status != PARSEC_DATA_STATUS_UNDER_TRANSFER);
    }
#endif /* defined(PARSEC_DEBUG_PARANOID) */

    /* The submit hook may turn gpu_task into a batch ring. Start from a clean
     * singleton so stale list links left by release-mode list operations cannot
     * be mistaken for a preexisting ring.
     * However, if the task returned PARSEC_HOOK_RETURN_AGAIN, we must preserve the
     * ring as it may contain the follower tasks of the batch.
     */
    if (gpu_task->last_status != PARSEC_HOOK_RETURN_AGAIN) {
        /* The task is being rescheduled, we need to reset the status */
        PARSEC_LIST_ITEM_SINGLETON(&gpu_task->list_item);
    }

    (void)this_task;
    rc = progress_fct( gpu_device, gpu_task, gpu_stream );
    gpu_task->last_status = rc;
    /* Negative hook returns are scheduler control flow and must be propagated
     * immediately. Non-negative returns mean the submit hook queued GPU work,
     * so ask the stream progress engine to record an execution event.
     */
    if( rc < 0 )
        return rc;
    return 1;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *  The copies are not issued by this function: the flows to be transferred
 *  are marked in @p transfer_mask, and the caller must issue the transfers.
 *
 *  Returns: HOOK_ERROR if any error occurred.
 *           positive: the number of data to be moved.
 */
static int
parsec_device_kernel_pop_task( parsec_device_gpu_module_t   *gpu_device,
                               parsec_gpu_task_t            *gpu_task,
                               parsec_gpu_exec_stream_t     *gpu_stream,
                               uint32_t                     *transfer_mask )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy;
    parsec_data_t              *original;
    size_t                      nb_elts;
    const parsec_flow_t        *flow;
    int how_many = 0, i, update_data_epoch = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    (void)gpu_stream;

    if (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < this_task->locals[0].value; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            /* If the gpu copy is not owned by parsec, we don't manage it at all */
            if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;
            *transfer_mask |= (1U << i);
            how_many++;
        }
        return how_many;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                        "GPU[%d:%s]: Try to Pop %s",
                        gpu_device->super.device_index, gpu_device->super.name,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    size_t pending_bytes = 0;
    int nb_pushout = 0;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* We need to manage all data that has been used as input, even if they were read only */

        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        flow = gpu_task->flow[i];
        if( PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags) )  continue;  /* control flow */

        gpu_copy = this_task->data[i].data_out;

        /* If the gpu copy is not owned by parsec, we don't manage it at all */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;

        original = gpu_copy->original;
        nb_elts = gpu_task->flow_nb_elts[i];

        assert( this_task->data[i].data_in == NULL || original == this_task->data[i].data_in->original );

#if 0
        if( (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_D2D_COMPLETE) && !(flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[i].data_out = original->device_copies[0];
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%d:%s]: pop %s swap flow %d GPU read-only data_out %p [ref_count %d] with the corresponding CPU copy %p [ref_count %d] original %p",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), i,
                                     gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     this_task->data[i].data_out, this_task->data[i].data_out->super.super.obj_reference_count,
                                     original);
        }
#endif // 0

        parsec_atomic_lock(&original->lock);
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_READ ) {
            int current_readers = parsec_atomic_fetch_sub_int32(&gpu_copy->readers, 1) - 1;
            if( current_readers < 0 ) {
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                     "GPU[%d:%s]: While trying to Pop %s, gpu_copy %p [ref_count %d] on flow %d with original %p had a negative number of readers (%d)",
                                     gpu_device->super.device_index, gpu_device->super.name,
                                     parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task),
                                     gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     i, original, current_readers);
            }
            assert(current_readers >= 0);
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tread copy %p [ref_count %d] on flow %s has readers (%i)",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name, current_readers);
            if (flow->flow_flags == PARSEC_FLOW_ACCESS_READ) {
                /* Do not propagate GPU copies to successors */
                this_task->data[i].data_out = original->device_copies[0];
                if( (0 == current_readers) && gpu_copy->coherency_state == PARSEC_DATA_COHERENCY_SHARED ) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                        "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available on flow %s",
                                        gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);
                    parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
                    PARSEC_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                    parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
                    update_data_epoch = 1;
                    parsec_atomic_unlock(&original->lock);
                    continue;  /* done with this element, go for the next one */
                } else if (gpu_copy->coherency_state != PARSEC_DATA_COHERENCY_SHARED) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                        "GPU[%d:%s]:\tMake read-only copy %p [ref_count %d] available to owned LRU",
                                        gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count);
                    parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
                    PARSEC_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                    parsec_list_nolock_push_back(&gpu_device->gpu_mem_owned_lru, (parsec_list_item_t*)gpu_copy);
                    parsec_atomic_unlock(&original->lock);
                    continue;  /* done with this element, go for the next one */
                }
            }
        }
        if( flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == parsec_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                "GPU[%d:%s]:\tOUT Data copy %p [ref_count %d] for flow %s",
                                gpu_device->super.device_index, gpu_device->super.name, gpu_copy, gpu_copy->super.super.obj_reference_count, flow->name);

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += nb_elts;
            assert( ((parsec_list_item_t*)gpu_copy)->list_next == (parsec_list_item_t*)gpu_copy );
            assert( ((parsec_list_item_t*)gpu_copy)->list_prev == (parsec_list_item_t*)gpu_copy );

            assert( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
            if( gpu_task->pushout & (1 << i) ) {
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                    "GPU[%d:%s]:\tMove D2H data <%s:%x> copy %p [ref_count %d] -- D:%p -> H:%p requested",
                                    gpu_device->super.device_index, gpu_device->super.name, flow->name, original->key, gpu_copy, gpu_copy->super.super.obj_reference_count,
                                     (void*)gpu_copy->device_private, original->device_copies[0]->device_private);
#if defined(PARSEC_PROF_TRACE)
                if( gpu_stream->prof_event_track_enable ) {
                    if(gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_DATA_OUT) {
                        parsec_profile_data_collection_info_t info;
                        if( NULL != original->dc ) {
                            info.desc    = original->dc;
                            info.data_id = original->key;
                        } else {
                            info.desc    = (parsec_dc_t*)original;
                            info.data_id = -1;
                        }
                        gpu_task->prof_key_end = parsec_gpu_moveout_key_end;
                        gpu_task->prof_tp_id   = this_task->taskpool->taskpool_id;
                        gpu_task->prof_event_id = this_task->task_class->key_functions->key_hash(this_task->task_class->make_key(this_task->taskpool, this_task->locals), NULL);
                        PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                               parsec_gpu_moveout_key_start,
                                               gpu_task->prof_event_id,
                                               gpu_task->prof_tp_id,
                                               &info);
                    } else {
                        gpu_task->prof_key_end = -1;
                    }
                }
#endif
                /* Defer the actual transfer: accumulate this flow into transfer_mask,
                 * the caller issues the transfers of all flows (and tasks) at once. */
                *transfer_mask |= (1U << flow->flow_index);
                pending_bytes += nb_elts; /* TODO: not hardcoded, use datatype size */
                nb_pushout++;
            } else {
                assert( 0 == gpu_copy->readers );
            }
        }
        parsec_atomic_unlock(&original->lock);
    }

    if( 0 != nb_pushout ) {
        gpu_device->super.data_out_to_host += pending_bytes;
        how_many += nb_pushout;
    }

    if( update_data_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Pop %s DONE (return %d data epoch %"PRIu64")",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), how_many, gpu_device->data_avail_epoch );

    return how_many;
}

/**
 *  @brief Stage out the data of a ring of tasks. Each task of the ring is handled as
 *  described in parsec_device_kernel_pop_task(), and the outcome for each task is
 *  stored in its last_status. The transfers of all tasks are then issued together,
 *  merging the copies of the tasks using the default stage_out function into batched
 *  copies.
 *
 *  @returns PARSEC_HOOK_RETURN_DONE, per-task results are in last_status.
 */
static int
parsec_device_kernel_pop( parsec_device_gpu_module_t   *gpu_device,
                          parsec_gpu_task_t            *gpu_tasks,
                          parsec_gpu_exec_stream_t     *gpu_stream)
{
    parsec_gpu_task_t *tasks[PARSEC_DEVICE_MAX_BATCH_SIZE];
    uint32_t masks[PARSEC_DEVICE_MAX_BATCH_SIZE];
    parsec_gpu_task_t *iter = gpu_tasks;
    int nb_tasks = 0;

    do {
        assert(nb_tasks < PARSEC_DEVICE_MAX_BATCH_SIZE);
        tasks[nb_tasks++] = iter;
        iter = (parsec_gpu_task_t*)iter->list_item.list_next;
    } while( iter != gpu_tasks );

    for( int t = 0; t < nb_tasks; t++ ) {
        masks[t] = 0;
        tasks[t]->last_status = parsec_device_kernel_pop_task(gpu_device, tasks[t], gpu_stream, &masks[t]);
    }
    parsec_device_issue_transfers(gpu_device, gpu_stream, tasks, masks, nb_tasks,
                                  false, PARSEC_HOOK_RETURN_DISABLE);
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
static int
parsec_device_kernel_epilog( parsec_device_gpu_module_t *gpu_device,
                             parsec_gpu_task_t          *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Epilog of %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;


        if( !(gpu_task->flow[i]->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        /* If it is a copy managed by the user, don't bother either */
        if( 0 == (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) continue;

        /**
         * There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         *
         * WARNING: For now we always forward the cpu_copy to the next task, to
         * do that, we lie to the engine by updating the CPU copy to the same
         * status than the GPU copy without updating the data itself. Thus, the
         * cpu copy is really invalid. this is related to Issue #88, and the
         * fact that:
         *      - we don't forward the gpu copy as output
         *      - we always take a cpu copy as input, so it has to be in the
         *        same state as the GPU to prevent an extra data movement.
         */
        assert( PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state );

        /* Don't mess with the host copy if it's not allocated */
#if 0
        // TODO: THIS IS INSANE! Why do we mess with the CPU copy if we haven't done a pushout?
        if ( NULL != cpu_copy->device_private ) {
            gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->version = gpu_copy->version;
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                "GPU[%d:%s]: CPU copy %p [ref_count %d, ptr %p] gets the same version %d as GPU copy %p [ref_count %d]",
                                gpu_device->super.device_index, gpu_device->super.name,
                                cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->device_private, cpu_copy->version,
                                gpu_copy, gpu_copy->super.super.obj_reference_count);
        } else
#endif // 0
        {
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                "GPU[%d:%s]: CPU copy %p [ref_count %d] version not updated to %d of GPU copy %p [ref_count %d] because no host data allocated",
                                gpu_device->super.device_index, gpu_device->super.name,
                                cpu_copy, cpu_copy->super.super.obj_reference_count, cpu_copy->version, gpu_copy, gpu_copy->super.super.obj_reference_count);
        }

        /**
         * Let's lie to the engine by reporting that working version of this
         * data is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;

        assert( 0 <= gpu_copy->readers );

        if (cpu_copy->flags & PARSEC_DATA_FLAG_DISCARDED) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s] Release discarded GPU copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                                 original);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            /* release the original and */
            parsec_device_release_gpu_copy(gpu_device, gpu_copy);
            parsec_atomic_fetch_dec_int64(&gpu_device->super.nb_discarded);
        } else if( gpu_task->pushout & (1 << i)) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU copy %p [ref_count %d] moved to the read LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            cpu_copy->version = gpu_copy->version;
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        } else {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU copy %p [ref_count %d] moved to the owned LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_owned_lru, (parsec_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

/** @brief Release the GPU copies of the data used in WRITE mode.
 *
 * @details This function can be used when the GPU task didn't run
 *          to completion on the device (either due to an error, or
 *          simply because the body requested a reexecution on a
 *          different location). It releases the GPU copies of the
 *          output data, allowing them to be reused by the runtime.
 *          This function has the drawback of kicking in too late,
 *          after all data transfers have been completed toward the
 *          device.
 *
 * @param [IN] gpu_device, the GPU device the the task has been
 *             supposed to execute.
 * @param [IN] gpu_task, the task that has been cancelled, and which
 *             needs it's data returned to the runtime.
 * @return Currently only success.
 */
static int
parsec_device_kernel_cleanout( parsec_device_gpu_module_t *gpu_device,
                               parsec_gpu_task_t          *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i, data_avail_epoch = 0;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                         "GPU[%d:%s]: Cleanup of %s",
                         gpu_device->super.device_index, gpu_device->super.name,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Make sure data_in is not NULL */
        if( NULL == this_task->data[i].data_in ) continue;

        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        if( !(gpu_task->flow[i]->flow_flags & PARSEC_FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        parsec_atomic_lock(&original->lock);
        assert(gpu_copy->super.super.obj_reference_count > 1);
        /* Issue #134 */
        parsec_data_copy_detach(original, gpu_copy, gpu_device->super.device_index);
        gpu_copy->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
        cpu_copy = original->device_copies[0];

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;
        if( 0 != (gpu_copy->flags & PARSEC_DATA_FLAG_PARSEC_OWNED) ) {
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
        parsec_atomic_unlock(&original->lock);
        data_avail_epoch++;
        PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                             "GPU copy %p [ref_count %d] moved to the read LRU in %s\n",
                             gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
    }
    if( data_avail_epoch )  /* Update data availability epoch */
        gpu_device->data_avail_epoch++;
    return 0;
}

/* parsec_lifo_detach_chain() returns tasks in LIFO order (most recently
 * pushed first). parsec_heap_push_chain() stamps the FIFO tie-break
 * sequence number in traversal order, so pushing a LIFO-ordered chain
 * directly would give newer tasks smaller (i.e. "older") sequence numbers,
 * inverting arrival-order fairness among equal-priority tasks within a
 * single detached batch. Reverse the chain in place so it is walked
 * oldest-first, matching true arrival order. */
static parsec_list_item_t *
parsec_gpu_pending_chain_reverse(parsec_list_item_t *chain)
{
    parsec_list_item_t *prev = NULL;
    while (NULL != chain) {
        parsec_list_item_t *next = (parsec_list_item_t *)chain->list_next;
        chain->list_next = prev;
        prev = chain;
        chain = next;
    }
    return prev;
}

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on GPU events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_device_kernel_scheduler( parsec_device_module_t *module,
                                parsec_execution_stream_t *es,
                                void *_gpu_task )
{
    parsec_device_gpu_module_t* gpu_device = (parsec_device_gpu_module_t *)module;
    int rc, exec_stream = 0;
    parsec_gpu_task_t *progress_task = NULL;
    parsec_gpu_task_t *gpu_task = (parsec_gpu_task_t*)_gpu_task;
    parsec_task_t *task_ring = NULL;
    parsec_gpu_task_t *iter = NULL;
    parsec_gpu_task_t *release_ring = NULL;
    int clear_device_counter = PARSEC_DEVICE_CLEAR_EVERY;
    int cnt = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    int pop_null = 0;

#if defined(PARSEC_PROF_TRACE)
    PARSEC_PROFILING_TRACE_FLAGS( es->es_profile,
                                  PARSEC_PROF_FUNC_KEY_END(gpu_task->ec->taskpool,
                                                           gpu_task->ec->task_class->task_class_id),
                                  gpu_task->ec->task_class->key_functions->key_hash(gpu_task->ec->task_class->make_key(gpu_task->ec->taskpool, gpu_task->ec->locals), NULL),
                                  gpu_task->ec->taskpool->taskpool_id, NULL,
                                  PARSEC_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(PARSEC_PROF_TRACE) */

    if (gpu_task != NULL && gpu_task->priority < 0) {
        gpu_task->priority = (gpu_task->ec != NULL) ? gpu_task->ec->priority : 0;
    }

    /* Check the GPU status -- three kinds of values for rc:
     *   - rc < 0: somebody is doing a short atomic operation while there is no manager,
     *             so wait.
     *   - rc == 0: there is no manager, and at the exit of the while, this thread
     *             made rc go from 0 to 1, so it is the new manager of the GPU and
     *             needs to deal with gpu_task
     *   - rc > 0: there is a manager, and at the exit of the while, this thread has
     *             committed new work that the manager will need to do, but the work is
     *             not in the queue yet.
     */
    while(1) {
        rc = gpu_device->mutex;
        struct timespec delay;
        if( rc >= 0 ) {
            if( parsec_atomic_cas_int32( &gpu_device->mutex, rc, rc+1 ) ) {
                break;
            }
        } else {
            delay.tv_nsec = 100;
            delay.tv_sec = 0;
            nanosleep(&delay, NULL);
        }
    }
    if( 0 < rc ) {
        parsec_lifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream, "GPU[%d:%s]: Entering GPU management",
                         gpu_device->super.device_index, gpu_device->super.name);

#if defined(PARSEC_PROF_TRACE)
    if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_start,
                                (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

    rc = gpu_device->set_device(gpu_device);
    if(PARSEC_SUCCESS != rc)
        return PARSEC_HOOK_RETURN_DISABLE;

 check_in_deps:

    /**
     * Drain the gpu_mem_inbox to reclaim lru elements.
     */
    if (!parsec_lifo_is_empty(&gpu_device->gpu_mem_inbox)) {
        parsec_gpu_data_copy_t *gpu_copy;
        while( NULL != (gpu_copy = (parsec_gpu_data_copy_t*)parsec_lifo_pop(&gpu_device->gpu_mem_inbox)) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_gpu_output_stream,
                                 "GPU[%d:%s]:\tPutting inbox copy %p [ref_count %d] attached to %p into the LRU",
                                 gpu_device->super.device_index, gpu_device->super.name,
                                 gpu_copy, gpu_copy->super.super.obj_reference_count,
                                 gpu_copy->original);
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        }
    }

    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%d:%s]:\tUpload data (if any) for %s",
                             gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
    }
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[0],
                                        parsec_device_kernel_push,
                                        gpu_task, &progress_task, true );
    if( rc < 0 ) {  /* In case of error progress_task is the task that raised it */
        if( PARSEC_HOOK_RETURN_ERROR == rc )
            goto disable_gpu;
        /* We are in the early stages, and if there no room on the GPU for a task we need to
         * delay all retries for the same task for a little while. Meanwhile, put the task back
         * trigger a device flush, and keep executing tasks that have their data on the device.
         */
        if( PARSEC_HOOK_RETURN_ASYNC == rc ) {
            gpu_task = progress_task;
            progress_task = NULL;
            goto remove_gpu_task;
        }
        assert(NULL == progress_task);

        /* try to release all discarded copies and try again if succesful */
        if (0 < parsec_device_memory_release_discarded(gpu_device, &gpu_device->gpu_mem_owned_lru)) {
            gpu_task = NULL;
            goto check_in_deps;
        }

        /* If we can extract data go for it, otherwise try to drain the pending tasks.
         * Executing tasks release memory once they complete, so delay the eviction
         * while the next execution stream still has work, and start pushing data out
         * as soon as it runs dry. */
        if ((active_w2r_tasks == 0) &&
            !parsec_device_stream_has_tasks(gpu_device->exec_stream[2 + (exec_stream + 1) % (gpu_device->num_exec_streams - 2)])) {
            gpu_task = parsec_gpu_create_w2r_task(gpu_device, es);
            if( NULL != gpu_task ) {
                active_w2r_tasks++;
                goto get_data_out_of_device;
            }
        }
    }
    gpu_task = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->num_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tExecute %s", gpu_device->super.device_index, gpu_device->super.name,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    }
    parsec_device_stream_take_delayed(gpu_device, gpu_device->exec_stream[2+exec_stream]);
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[2+exec_stream],
                                        parsec_device_kernel_exec,
                                        gpu_task, &progress_task, false );
    if( rc < 0 ) {
        if( (PARSEC_HOOK_RETURN_DISABLE == rc) || (PARSEC_HOOK_RETURN_ERROR == rc) )
            goto disable_gpu;
        if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
            /* Reschedule the task. As the chore_id has been modified,
               another incarnation of the task will be executed. */
            if( NULL != progress_task ) {
                assert(PARSEC_HOOK_RETURN_NEXT == rc);
                parsec_device_kernel_cleanout(gpu_device, progress_task);
                __parsec_reschedule(es, progress_task->ec);
                gpu_task = progress_task;
                progress_task = NULL;
                goto remove_gpu_task;
            }
            gpu_task = NULL;
            goto fetch_task_from_shared_queue;
        }
        gpu_task = progress_task;
        progress_task = NULL;
        goto remove_gpu_task;
    }
    gpu_task = progress_task;

 get_data_out_of_device:
    if( NULL != gpu_task ) {  /* This task has completed its execution */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tRetrieve data (if any) for %s", gpu_device->super.device_index, gpu_device->super.name,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    } else if (clear_device_counter-- == 0) {
        /* every so often if we are idle release discarded data to clean up the host-side copies */
        parsec_device_memory_release_discarded(gpu_device, &gpu_device->gpu_mem_owned_lru);
        parsec_device_memory_release_discarded(gpu_device, &gpu_device->gpu_mem_lru);
        clear_device_counter = PARSEC_DEVICE_CLEAR_EVERY;
    }
    /* Task is ready to move the data back to main memory */
    rc = parsec_device_progress_stream( gpu_device,
                                        gpu_device->exec_stream[1],
                                        parsec_device_kernel_pop,
                                        gpu_task, &progress_task, true );
    if( rc < 0 ) {
        if( (PARSEC_HOOK_RETURN_ERROR == rc) || (PARSEC_HOOK_RETURN_DISABLE == rc) )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a successfully completed task. However, it is not gpu_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        gpu_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    gpu_task = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    {
        parsec_list_item_t *chain = parsec_lifo_detach_chain(&gpu_device->pending);
        if (NULL != chain) {
            chain = parsec_gpu_pending_chain_reverse(chain);
            parsec_heap_push_chain(&gpu_device->pending_heap, chain);
        }
    }
    gpu_task = (parsec_gpu_task_t*)parsec_heap_pop(&gpu_device->pending_heap);
    if( NULL != gpu_task ) {
        pop_null = 0;
        /* parsec_heap_pop() singletonizes the returned item before returning it. */
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch - 1;  /* force at least one tour */
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tGet from shared queue %s", gpu_device->super.device_index, gpu_device->super.name,
                             parsec_device_describe_gpu_task(tmp, MAX_TASK_STRLEN, gpu_task));
        if( PARSEC_GPU_TASK_TYPE_D2D_COMPLETE == gpu_task->task_type ) {
            goto get_data_out_of_device;
        }
        int chain_len = 1;
        bool can_batch = parsec_gpu_task_selected_chore_allows_batch(gpu_task->ec, (parsec_device_module_t*)gpu_device);
        while (can_batch && chain_len < PARSEC_DEVICE_MAX_BATCH_SIZE) {
            parsec_gpu_task_t *candidate = (parsec_gpu_task_t*)parsec_heap_peek(&gpu_device->pending_heap);
            if (NULL == candidate) {
                break;
            }
            if (!parsec_gpu_task_same_chore(gpu_task, candidate)) {
                break;
            }
            // remove the item from the heap
            parsec_heap_pop(&gpu_device->pending_heap);
            parsec_list_item_ring_push((parsec_list_item_t*)gpu_task, (parsec_list_item_t*)candidate);
            chain_len++;
        }
    } else {
        pop_null++;
        if( pop_null % 1024 == 1023 ) {
            PARSEC_DEBUG_VERBOSE(30, parsec_gpu_output_stream,  "GPU[%d:%s]:\tStill waiting for %d tasks to execute, but poped NULL the last %d times I tried to pop something...",
                                 gpu_device->super.device_index, gpu_device->super.name, gpu_device->mutex, pop_null);
        }
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != gpu_task );
    task_ring = NULL;
    release_ring = NULL;
    iter = gpu_task;
    do {
        /* Take the first item off the ring and process it */
        parsec_gpu_task_t *next = (parsec_gpu_task_t*)parsec_list_item_ring_chop((parsec_list_item_t*)iter);
        PARSEC_LIST_ITEM_SINGLETON(iter);

        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,  "GPU[%d:%s]:\tComplete %s",
                             gpu_device->super.device_index, gpu_device->super.name,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, iter->ec));
        if (iter->task_type == PARSEC_GPU_TASK_TYPE_D2HTRANSFER) {
            parsec_gpu_complete_w2r_task(gpu_device, iter, es);
            active_w2r_tasks--;
            /* d2h transfers don't go through remove_gpu_task */
        } else {
            if (iter->task_type == PARSEC_GPU_TASK_TYPE_D2D_COMPLETE) {
                free( iter->ec );
                iter->ec = NULL;
            } else {
                parsec_device_kernel_epilog( gpu_device, iter );
                iter->ec->status = PARSEC_TASK_STATUS_COMPLETE;
                gpu_device->super.executed_tasks++;
                PARSEC_LIST_ITEM_SINGLETON(iter->ec);
                if (NULL == task_ring) {
                    task_ring = iter->ec;
                } else {
                    parsec_list_item_ring_push(&task_ring->super, (parsec_list_item_t*)iter->ec);
                }
            }
            /* still needs release_device_task via remove_gpu_task below */
            if (NULL == release_ring) {
                release_ring = iter;
            } else {
                parsec_list_item_ring_push((parsec_list_item_t*)release_ring, (parsec_list_item_t*)iter);
            }
        }
        iter = next;
    } while (iter != NULL);

    if (task_ring != NULL) {
        __parsec_schedule(es, task_ring, 1);
    }
    iter = NULL;
    task_ring = NULL;
    gpu_task = release_ring;
    release_ring = NULL;
 remove_gpu_task:
    if (gpu_task == NULL) {
        goto fetch_task_from_shared_queue;
    }
    iter = gpu_task;
    cnt = 0;
    do {
        parsec_gpu_task_t *next = (parsec_gpu_task_t*)parsec_list_item_ring_chop((parsec_list_item_t*)iter);
        PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%d:%s]: gpu_task %p freed",
                            gpu_device->super.device_index, gpu_device->super.name,
                            iter);
        PARSEC_LIST_ITEM_SINGLETON(iter);
        if (NULL != iter->release_device_task) {
            iter->release_device_task(iter);
        }
        cnt++;
        iter = next;
    } while (iter != NULL);

    rc = parsec_atomic_fetch_sub_int32( &(gpu_device->mutex), cnt ) - cnt;

    if( 0 == rc ) {  /* I was the last one */
#if defined(PARSEC_PROF_TRACE)
        if( gpu_device->trackable_events & PARSEC_PROFILE_GPU_TRACK_OWN )
            PARSEC_PROFILING_TRACE( es->es_profile, parsec_gpu_own_GPU_key_end,
                                    (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */
        PARSEC_DEBUG_VERBOSE(5, parsec_gpu_output_stream, "GPU[%d:%s]: Leaving GPU management",
                             gpu_device->super.device_index, gpu_device->super.name);
        /* inform the upper layer not to use the task argument, it has been long gone */
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    gpu_task = progress_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    parsec_warning("GPU[%d:%s]: Critical issue related to the GPU discovered. Giving up",
                   gpu_device->super.device_index, gpu_device->super.name);
    return PARSEC_HOOK_RETURN_DISABLE;
}
