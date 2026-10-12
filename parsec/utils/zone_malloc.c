/*
 * Copyright (c) 2012-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/utils/debug.h"

#include "parsec/class/list.h"
#include "parsec/class/parsec_rbtree.h"

#include <stdio.h>

typedef struct zone_malloc_chunk_list_t {
    parsec_rbtree_node_t super;
    parsec_list_t list; /* list of free segments of specific size */
    int nb_units;       /* size of free segments */
} zone_malloc_chunk_list_t;

static inline void
zone_malloc_chunk_list_construct( zone_malloc_chunk_list_t* item )
{
    item->nb_units = 0;
    PARSEC_OBJ_CONSTRUCT(&item->list, parsec_list_t);
}

PARSEC_OBJ_CLASS_INSTANCE(zone_malloc_chunk_list_t, parsec_rbtree_node_t,
                          zone_malloc_chunk_list_construct, NULL);

static zone_malloc_chunk_list_t* allocate_chunk_list(zone_malloc_t *gdata, int nb_units) {
    zone_malloc_chunk_list_t* fl;
    /* add a new segment */
    if (!parsec_lifo_is_empty(&gdata->rbtree_free_list)) {
        fl = (zone_malloc_chunk_list_t*)parsec_lifo_pop(&gdata->rbtree_free_list);
    } else {
        /* allocate new */
        fl = PARSEC_OBJ_NEW(zone_malloc_chunk_list_t);
    }
    PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
    fl->nb_units = nb_units;
    return fl;
}

static inline void zone_malloc_error(const char *msg)
{
    fprintf(stderr, "%s", msg);
}

static inline segment_t *SEGMENT_AT_TID(zone_malloc_t *gdata, int tid)
{
    if( tid < 0 )
        return NULL;
    if( tid >= gdata->max_segment )
        return NULL;
    return &gdata->segments[tid];
}

zone_malloc_t* zone_malloc_init(void* base_ptr, int _max_segment, size_t _unit_size)
{
    zone_malloc_t *gdata;
    segment_t *head;

    if( NULL == base_ptr ) {
        zone_malloc_error("Cannot manage an empty memory region\n");
        return NULL;
    }

    gdata = (zone_malloc_t*)malloc( sizeof(zone_malloc_t) );
    gdata->base         = base_ptr;
    gdata->unit_size    = _unit_size;
    gdata->max_segment  = _max_segment;
    gdata->next_tid     = 0;
    gdata->segments     = (segment_t *)malloc(sizeof(segment_t) * _max_segment);
    parsec_atomic_lock_init(&gdata->lock);
    parsec_rbtree_init(&gdata->rbtree, offsetof(zone_malloc_chunk_list_t, nb_units));
    PARSEC_OBJ_CONSTRUCT(&gdata->rbtree_free_list, parsec_lifo_t);
    for(int i = 0; i < _max_segment; i++) {
        SEGMENT_AT_TID(gdata, i)->status = SEGMENT_UNDEFINED;
        PARSEC_OBJ_CONSTRUCT(&SEGMENT_AT_TID(gdata, i)->super, parsec_list_item_t);
    }
    head = SEGMENT_AT_TID(gdata, 0);
    head->status = SEGMENT_EMPTY;
    head->nb_units = _max_segment;
    head->nb_prev  = 1; /**< This is to force SEGMENT_OF_TID( 0 - prev ) to return NULL */

    zone_malloc_chunk_list_t *fl = PARSEC_OBJ_NEW(zone_malloc_chunk_list_t);
    fl->nb_units = head->nb_units;
    parsec_list_push_back(&fl->list, &head->super);
    parsec_rbtree_insert(&gdata->rbtree, &fl->super);

    return gdata;
}

void* zone_malloc_fini(zone_malloc_t** gdata)
{
    void* base_ptr = (*gdata)->base;

    free( (*gdata)->segments );

    (*gdata)->max_segment = 0;
    (*gdata)->unit_size = 0;
    (*gdata)->base = NULL;

    parsec_rbtree_fini(&(*gdata)->rbtree);
    free(*gdata);
    *gdata = NULL;
    return base_ptr;
}

/* Remove a free segment from its rbtree-backed chunk list, so it can be
 * merged with a neighbor or handed out again. */
static void remove_segment_from_freelist(zone_malloc_t *gdata, segment_t *current_segment) {
    zone_malloc_chunk_list_t *fl;

    /* find chunk list */
    fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, current_segment->nb_units);
    assert(fl != NULL);
    /* remove from list */
    parsec_list_nolock_remove(&fl->list, &current_segment->super);
    if (parsec_list_nolock_is_empty(&fl->list)) {
        /* empty chunk list, remove */
        parsec_rbtree_remove(&gdata->rbtree, &fl->super);
        /* put into free list */
        parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
    }
}

/* Insert a free segment into its rbtree-backed chunk list. */
static void insert_segment_into_freelist(zone_malloc_t *gdata, segment_t *current_segment) {
    zone_malloc_chunk_list_t *fl;

    PARSEC_LIST_ITEM_SINGLETON(&current_segment->super);

    fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, current_segment->nb_units);
    if (fl == NULL) {
        /* create new chunk list and insert into rbtree */
        fl = allocate_chunk_list(gdata, current_segment->nb_units);
        parsec_rbtree_insert(&gdata->rbtree, &fl->super);
    }
    parsec_list_nolock_push_front(&fl->list, &current_segment->super);
}

static inline int zone_malloc_size_to_units(zone_malloc_t *gdata, size_t size)
{
    return (int)((size + gdata->unit_size - 1) / gdata->unit_size);
}

/*
 * Carve consecutive items sizes[0], sizes[1], ... (at most `max_count` of
 * them) out of a SINGLE newly-located free chunk (at most one
 * rbtree/unit-list lookup), writing the resulting pointers into
 * out_ptrs[0..k-1] and returning k. Items are placed back to back in the
 * chunk, in order, until the chunk cannot hold the next item. Items of
 * size 0 need no memory: their pointer is set to NULL and they count as
 * allocated. Returns the number of leading zero-size items (possibly 0)
 * if no free chunk holding the first non-empty item could be found.
 *
 * This is the core of both the single-item zone_malloc() (max_count==1)
 * and zone_malloc_batch(): when the chunk found is big enough to hold
 * several items, all of them are carved out in one pass with only the
 * initial lookup paying the rbtree search cost -- callers that need more
 * than `k` items call this again (see zone_malloc_batch) to look for
 * another chunk, still under the same lock acquisition.
 *
 * Caller must hold gdata->lock.
 */
static int zone_malloc_locked(zone_malloc_t *gdata, const size_t *sizes, int max_count, void **out_ptrs)
{
    segment_t *current_segment, *next_segment, *leftover;
    zone_malloc_chunk_list_t* fl;
    int current_tid, chunk_units, nb_units, k = 0, consumed, leftover_units, prev_units;

    /* leading zero-size items need no memory */
    while (k < max_count && 0 == (nb_units = zone_malloc_size_to_units(gdata, sizes[k]))) {
        out_ptrs[k++] = NULL;
    }
    if (k == max_count) {
        return k;
    }

    /* try to find the smallest possible element for the first item, or one size larger */
    fl = (zone_malloc_chunk_list_t*) parsec_rbtree_find_or_larger(&gdata->rbtree, nb_units);
    if (NULL == fl) {
        return k; /* no segment found */
    }

    current_segment = (segment_t *)parsec_list_nolock_pop_front(&fl->list);
    assert(current_segment->nb_units >= nb_units);
    if (parsec_list_nolock_is_empty(&fl->list)) {
        /* empty chunk list, remove */
        parsec_rbtree_remove(&gdata->rbtree, &fl->super);
        PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
        /* put into free list */
        parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
    }

    current_tid  = (int)(current_segment - gdata->segments);
    chunk_units  = current_segment->nb_units;
    prev_units   = current_segment->nb_prev;
    consumed     = 0;

    /* carve consecutive items as long as the chunk holds them, capped at max_count */
    while (k < max_count) {
        nb_units = zone_malloc_size_to_units(gdata, sizes[k]);
        if (0 == nb_units) {
            out_ptrs[k++] = NULL;
            continue;
        }
        if (consumed + nb_units > chunk_units) {
            break;
        }
        segment_t *piece = SEGMENT_AT_TID(gdata, current_tid + consumed);
        piece->status   = SEGMENT_FULL;
        piece->nb_units = nb_units;
        piece->nb_prev  = prev_units;
        out_ptrs[k++] = (void*)(gdata->base + ((size_t)(current_tid + consumed) * gdata->unit_size));
        prev_units = nb_units;
        consumed  += nb_units;
    }

    /* fix up whatever follows the whole original chunk: its new immediate
     * predecessor is either the leftover remainder, or (if the chunk was
     * consumed exactly) the last carved piece -- never the original chunk,
     * which no longer exists as a single run once more than one item was carved. */
    leftover_units  = chunk_units - consumed;
    next_segment    = SEGMENT_AT_TID(gdata, current_tid + chunk_units);
    if (leftover_units > 0) {
        if (NULL != next_segment) next_segment->nb_prev = leftover_units;
        leftover = SEGMENT_AT_TID(gdata, current_tid + consumed);
        leftover->status   = SEGMENT_EMPTY;
        leftover->nb_prev  = prev_units;
        leftover->nb_units = leftover_units;
        insert_segment_into_freelist(gdata, leftover);
    } else if (NULL != next_segment) {
        next_segment->nb_prev = prev_units;
    }

    return k;
}

void *zone_malloc(zone_malloc_t *gdata, size_t size)
{
    void *ptr = NULL;

    if (0 == size) {
        return NULL;
    }

    parsec_atomic_lock(&gdata->lock);
    int k = zone_malloc_locked(gdata, &size, 1, &ptr);
    parsec_atomic_unlock(&gdata->lock);
    return (1 == k) ? ptr : NULL;
}

/*
 * Allocate `count` regions of sizes[0..count-1] bytes under a single lock
 * acquisition instead of `count` separate ones. Items are allocated in
 * order; consecutive items are carved back to back out of the same free
 * chunk as long as it holds them. When a single free chunk covers the
 * whole batch, this pays the rbtree search cost once instead of `count`
 * times; a fragmented pool falls back to one search per distinct chunk
 * needed to cover the batch (never more than `count`). Items of size 0
 * get a NULL pointer.
 *
 * Returns the number of leading items actually allocated (0..count). On a
 * short return, out_ptrs[0..return value - 1] are valid allocations the
 * caller is responsible for freeing (e.g. via zone_free_batch) if it decides
 * not to proceed with a partially-satisfied batch.
 */
int zone_malloc_batch(zone_malloc_t *gdata, const size_t *sizes, int count, void **out_ptrs)
{
    int done = 0;

    if (count <= 0) {
        return 0;
    }

    parsec_atomic_lock(&gdata->lock);
    while (done < count) {
        int k = zone_malloc_locked(gdata, sizes + done, count - done, out_ptrs + done);
        if (0 == k) break;
        done += k;
    }
    parsec_atomic_unlock(&gdata->lock);
    return done;
}

/* Free one segment; caller must hold gdata->lock. */
static void zone_free_locked(zone_malloc_t *gdata, void *add)
{
    segment_t *current_segment, *next_segment, *prev_segment;
    int current_tid, next_tid, prev_tid;
    off_t offset;

    offset = (char*)add -gdata->base;
    assert( (offset % gdata->unit_size) == 0);
    current_tid = offset / gdata->unit_size;
    current_segment = SEGMENT_AT_TID(gdata, current_tid);

    if( NULL == current_segment ) {
        zone_malloc_error("address to free not allocated\n");
        return;
    }

    if( SEGMENT_EMPTY == current_segment->status ) {
        zone_malloc_error("double free (or other buffer overflow) error in ZONE allocation");
        return;
    }

    /* check if we can merge segments */
    current_segment->status = SEGMENT_EMPTY;

    prev_tid = current_tid - current_segment->nb_prev;
    prev_segment = SEGMENT_AT_TID(gdata, prev_tid);

    next_tid = current_tid + current_segment->nb_units;
    next_segment = SEGMENT_AT_TID(gdata, next_tid);

    if (NULL != prev_segment && prev_segment->status == SEGMENT_EMPTY) {
        remove_segment_from_freelist(gdata, prev_segment);
        /* We can merge prev and current */
        if( NULL != next_segment ) {
            next_segment->nb_prev += prev_segment->nb_units;
        }
        prev_segment->nb_units += current_segment->nb_units;

        /* Pretend we are now our prev, so that we merge with next if needed */
        current_segment = prev_segment;
        current_tid     = prev_tid;
    }

    if (NULL != next_segment && next_segment->status == SEGMENT_EMPTY) {
        remove_segment_from_freelist(gdata, next_segment);
        /* We can merge current and next */
        next_tid += next_segment->nb_units;
        current_segment->nb_units += next_segment->nb_units;
        next_segment = SEGMENT_AT_TID(gdata, next_tid);
        if( NULL != next_segment ) {
            next_segment->nb_prev = current_segment->nb_units;
        }
    }

    insert_segment_into_freelist(gdata, current_segment);
}

void zone_free(zone_malloc_t *gdata, void *add)
{
    parsec_atomic_lock(&gdata->lock);
    zone_free_locked(gdata, add);
    parsec_atomic_unlock(&gdata->lock);
}

/* Free `count` pointers under a single lock acquisition instead of `count`
 * separate ones. NULL pointers (zero-size items of a batch) are skipped. */
void zone_free_batch(zone_malloc_t *gdata, void **ptrs, int count)
{
    parsec_atomic_lock(&gdata->lock);
    for (int i = 0; i < count; i++) {
        if (NULL == ptrs[i]) continue;
        zone_free_locked(gdata, ptrs[i]);
    }
    parsec_atomic_unlock(&gdata->lock);
}

size_t zone_in_use(zone_malloc_t *gdata)
{
    size_t ret = 0;
    segment_t *current_segment;
    int current_tid;
    parsec_atomic_lock(&gdata->lock);
    /* check segments */
    for(current_tid = 0;
        (current_segment = SEGMENT_AT_TID(gdata, current_tid)) != NULL;
        current_tid += current_segment->nb_units) {
        if( current_segment->status == SEGMENT_FULL ) {
            ret += gdata->unit_size * current_segment->nb_units;
        }
    }
    parsec_atomic_unlock(&gdata->lock);
    return ret;
}

typedef struct zone_malloc_rbtree_debug_t {
    int level, output_id;
    const char* prefix;
} zone_malloc_rbtree_debug_t;

static void zone_rbtree_cb(parsec_rbtree_node_t *node, void *data) {
    zone_malloc_chunk_list_t *fl = (zone_malloc_chunk_list_t*)node;
    zone_malloc_rbtree_debug_t* info = (zone_malloc_rbtree_debug_t*)data;
    int len = 0;
    PARSEC_LIST_NOLOCK_ITERATOR(&fl->list, iter, (void)iter; ++len; );
    parsec_debug_verbose(info->level, info->output_id, "%srbtree node: %d segments a %d units",
                         info->prefix, len, fl->nb_units);
}

size_t zone_debug(zone_malloc_t *gdata, int level, int output_id, const char *prefix)
{
    segment_t *current_segment;
    int current_tid;
    size_t ret = 0;

    parsec_atomic_lock(&gdata->lock);
    for(current_tid = 0;
        (current_segment = SEGMENT_AT_TID(gdata, current_tid)) != NULL;
        current_tid += current_segment->nb_units) {
        if( current_segment->status == SEGMENT_EMPTY ) {
            ret += gdata->unit_size * current_segment->nb_units;
            if( NULL != prefix )
                parsec_debug_verbose(level, output_id, "%sfree: %d units (%d bytes) from %p to %p",
                                     prefix,
                                     current_segment->nb_units, gdata->unit_size*current_segment->nb_units,
                                     gdata->base + current_tid * gdata->unit_size,
                                     gdata->base + (current_tid+current_segment->nb_units) * gdata->unit_size - 1);
        } else {
            if( NULL != prefix )
                parsec_debug_verbose(level, output_id, "%sused: %d units (%d bytes) from %p to %p",
                                     prefix,
                                     current_segment->nb_units, gdata->unit_size*current_segment->nb_units,
                                     gdata->base + current_tid * gdata->unit_size,
                                     gdata->base + (current_tid+current_segment->nb_units) * gdata->unit_size - 1);
        }
    }
    zone_malloc_rbtree_debug_t info = {level, output_id, prefix};
    parsec_rbtree_foreach(&gdata->rbtree, zone_rbtree_cb, &info);

    parsec_atomic_unlock(&gdata->lock);

    return ret;
}
