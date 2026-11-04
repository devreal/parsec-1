/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "mempool.h"
#include "parsec/sys/atomic.h"
#include <stdlib.h>
#ifdef PARSEC_HAVE_STRING_H
#include <string.h>
#endif

/* Allocator interface: provide raw storage for an object */
static void *parsec_thread_mempool_obj_alloc( parsec_obj_allocator_t *allocator, parsec_class_t *cls )
{
    parsec_thread_mempool_t *thread_mempool = (parsec_thread_mempool_t *)allocator;
    assert(cls->cls_sizeof <= thread_mempool->parent->elt_size); (void)cls;
    return parsec_thread_mempool_pop(thread_mempool);
}

/* Allocator interface: destruct the object and return it to the thread mempool */
static void parsec_thread_mempool_obj_release( parsec_obj_allocator_t *allocator, parsec_object_t *object )
{
    parsec_thread_mempool_t *thread_mempool = (parsec_thread_mempool_t *)allocator;
    /* only objects created from this mempool can be returned to it */
    assert(object->obj_class->cls_sizeof <= thread_mempool->parent->elt_size);
    parsec_obj_destruct(object);
    parsec_thread_mempool_push(thread_mempool, parsec_list_item_from_storage(object));
}

/** parsec_thread_mempool_construct
 *    constructs the thread-specific memory pool.
 */
static void parsec_thread_mempool_construct( parsec_thread_mempool_t *thread_mempool,
                                            parsec_mempool_t *mempool )
{
    thread_mempool->super.obj_alloc   = parsec_thread_mempool_obj_alloc;
    thread_mempool->super.obj_release = parsec_thread_mempool_obj_release;
    thread_mempool->parent = mempool;
    PARSEC_OBJ_CONSTRUCT(&thread_mempool->mempool, parsec_lifo_t);
    PARSEC_OBJ_CONSTRUCT(&thread_mempool->chunks, parsec_lifo_t);
    thread_mempool->nb_elt = 0;
}

/* Returns the number of elements that were cached in the thread mempool */
static uint64_t parsec_thread_mempool_destruct( parsec_thread_mempool_t *thread_mempool )
{
    void *elt;
    uint64_t nb_cached = 0;
    const int chunked = thread_mempool->parent->elts_per_chunk > 1;

    while(NULL != (elt = parsec_lifo_pop(&thread_mempool->mempool))) {
        /* elements are either raw storage from parsec_lifo_item_alloc, or part of a chunk */
        if( !chunked ) parsec_lifo_item_storage_free(elt);
        nb_cached++;
    }
    /* the chunk headers are the first slot of each chunk, so this releases all elements */
    while(NULL != (elt = parsec_lifo_pop(&thread_mempool->chunks))) {
        free(elt);
    }
    PARSEC_OBJ_DESTRUCT(&thread_mempool->chunks);
    PARSEC_OBJ_DESTRUCT(&thread_mempool->mempool);
    return nb_cached;
}

void parsec_mempool_construct( parsec_mempool_t *mempool,
                              parsec_class_t* obj_class, size_t elt_size,
                              size_t pool_offset,
                              unsigned int nbthreads )
{
    uint32_t tid;

    /* Elements are chained as list items while in the mempool, which overwrites
     * the list item fields of every element, so the back pointer must lie
     * beyond them. Pools without a class are exempt only because one of them,
     * the dependencies mempool, overlaps its back pointer with list_next and
     * re-assigns it after every allocation. */
    assert(NULL == obj_class || pool_offset >= sizeof(parsec_list_item_t));

    mempool->nb_thread_mempools = nbthreads;
    mempool->elt_size = elt_size < sizeof(parsec_list_item_t) ? sizeof(parsec_list_item_t) : elt_size;
    mempool->pool_owner_offset = pool_offset;
    mempool->nb_max_elt = 0;
    mempool->obj_class = obj_class;
    mempool->elts_per_chunk = 1;
    mempool->thread_mempools = (parsec_thread_mempool_t *)malloc(sizeof(parsec_thread_mempool_t) * nbthreads);
    memset( mempool->thread_mempools, 0, sizeof(parsec_thread_mempool_t) * nbthreads );

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        parsec_thread_mempool_construct(&mempool->thread_mempools[tid], mempool);
}

static void parsec_mempool_destruct_impl( parsec_mempool_t *mempool,
                                          uint64_t *nb_allocated, uint64_t *nb_cached )
{
    uint32_t tid;

    *nb_allocated = 0;
    *nb_cached = 0;
    for(tid = 0; tid < mempool->nb_thread_mempools; tid++) {
        *nb_allocated += (uint64_t)mempool->thread_mempools[tid].nb_elt;
        *nb_cached += parsec_thread_mempool_destruct(&mempool->thread_mempools[tid]);
    }

    free(mempool->thread_mempools);
    mempool->thread_mempools = NULL;
    mempool->nb_thread_mempools = 0;
}

uint64_t parsec_mempool_destruct( parsec_mempool_t *mempool )
{
    uint64_t nb_allocated, nb_cached;
    parsec_mempool_destruct_impl(mempool, &nb_allocated, &nb_cached);
    return nb_allocated;
}

uint64_t parsec_mempool_destruct_in_use( parsec_mempool_t *mempool )
{
    uint64_t nb_allocated, nb_cached;
    parsec_mempool_destruct_impl(mempool, &nb_allocated, &nb_cached);
    return nb_allocated - nb_cached;
}

void parsec_mempool_set_elts_per_chunk( parsec_mempool_t *mempool, unsigned int elts_per_chunk )
{
    for(uint32_t tid = 0; tid < mempool->nb_thread_mempools; tid++) {
        assert(0 == mempool->thread_mempools[tid].nb_elt);  /* nothing allocated yet */
    }
    mempool->elts_per_chunk = (0 == elts_per_chunk) ? 1 : elts_per_chunk;
}

/* Cache line size used to align and pad chunked elements */
#define PARSEC_MEMPOOL_CHUNK_ALIGNMENT 64

/*
 * Allocate a chunk of elts_per_chunk elements with a single allocation. The
 * first slot of the chunk is its header, which links the chunk into the
 * thread mempool's list of chunks; every element starts on a cache line.
 * One element is returned, the others are pushed into the thread mempool
 * with a single atomic operation.
 */
static void *parsec_thread_mempool_allocate_chunk( parsec_thread_mempool_t *thread_mempool )
{
    parsec_mempool_t *mempool = thread_mempool->parent;
    const size_t align = PARSEC_LIFO_ALIGNMENT(&thread_mempool->mempool) > PARSEC_MEMPOOL_CHUNK_ALIGNMENT
                       ? PARSEC_LIFO_ALIGNMENT(&thread_mempool->mempool) : PARSEC_MEMPOOL_CHUNK_ALIGNMENT;
    const size_t stride = ((mempool->elt_size + align - 1) / align) * align;
    const unsigned int nb = mempool->elts_per_chunk;
    parsec_list_item_t *ring = NULL;
    char *chunk = NULL;
    void *ret;

    /* the header needs a list item: elt_size >= sizeof(parsec_list_item_t), see parsec_mempool_construct */
    if( 0 != posix_memalign((void**)&chunk, align, stride * (nb + 1)) || NULL == chunk ) {
        return NULL;
    }
    parsec_lifo_push(&thread_mempool->chunks, parsec_list_item_from_storage(chunk));

    for( unsigned int i = 1; i <= nb; i++ ) {
        char *elt = chunk + i * stride;
        *(parsec_thread_mempool_t **)(elt + mempool->pool_owner_offset) = thread_mempool;
        if( 1 == i ) continue;  /* returned to the caller */
        parsec_list_item_t *item = parsec_list_item_from_storage(elt);
        if( NULL == ring ) ring = item;
        else parsec_list_item_ring_push(ring, item);
    }
    ret = chunk + stride;
    if( NULL != ring ) {
        parsec_lifo_chain(&thread_mempool->mempool, ring);
    }
    parsec_atomic_fetch_add_int32(&thread_mempool->nb_elt, (int32_t)nb);
    return ret;
}

void *parsec_thread_mempool_allocate_when_empty( parsec_thread_mempool_t *thread_mempool )
{
    /*
     * Simple heuristic: don't try to balance things between threads,
     * just allocate within this thread
     */
    void *elt;
    parsec_thread_mempool_t **owner;

    if( thread_mempool->parent->elts_per_chunk > 1 ) {
        return parsec_thread_mempool_allocate_chunk(thread_mempool);
    }

    elt = parsec_lifo_item_alloc(&thread_mempool->mempool, thread_mempool->parent->elt_size );
    owner = (parsec_thread_mempool_t **)((char*)elt + thread_mempool->parent->pool_owner_offset);
    *owner = thread_mempool;
    parsec_atomic_fetch_inc_int32(&thread_mempool->nb_elt);
    return elt;
}
