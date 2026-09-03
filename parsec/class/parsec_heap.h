/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

#ifndef PARSEC_HEAP_H_HAS_BEEN_INCLUDED
#define PARSEC_HEAP_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/list_item.h"
#include "parsec/constants.h"

#include <stdint.h>
#include <stddef.h>

BEGIN_C_DECLS

/** Pass as seq_offset to parsec_heap_init() to disable FIFO tie-breaking
 * between elements of equal priority (legacy, arbitrary-tie behavior). */
#define PARSEC_HEAP_NO_SEQ ((size_t)-1)

/**
 * @brief Intrusive pointer-based max-heap with an int32_t priority key.
 *
 * @details Elements are linked directly through their parsec_list_item_t
 * list_prev (left child) and list_next (right child) pointers, forming a
 * complete binary tree — the same technique used by parsec/maxheap.c for
 * CPU scheduler task heaps.  No separate backing array is allocated.
 *
 * The priority of each element is read as *(int32_t*)((char*)element +
 * comp_offset), matching the COMPARISON_VAL macro convention used by the
 * rbtree and list sort.
 *
 * All operations are O(log N).  Sift-up uses a small on-stack path array
 * (max 64 entries; supports up to 2^64 elements).
 *
 * An element's list_prev/list_next are used as tree child pointers while it
 * is in the heap.  parsec_heap_pop() restores them to singleton state before
 * returning, so the caller can pass the result directly to
 * parsec_list_push_back() / parsec_gpu_stream_push_pending() etc.
 *
 * Ties in priority are broken FIFO-style using a monotonic enqueue sequence
 * number, so that a steady stream of equal-priority insertions cannot starve
 * elements already present in the heap: see parsec_heap_init(). The sequence
 * number is stamped into the element by parsec_heap_push()/push_chain() at
 * byte offset seq_offset, so the element type must reserve an unused
 * uint64_t there (unless tie-breaking is disabled with PARSEC_HEAP_NO_SEQ).
 */
typedef struct parsec_heap_s {
    parsec_list_item_t  *top;         /**< root of the complete binary tree */
    size_t               size;        /**< current element count */
    size_t               comp_offset; /**< byte offset of int32_t priority key */
    size_t               seq_offset;  /**< byte offset of uint64_t FIFO sequence key, or PARSEC_HEAP_NO_SEQ */
    uint64_t             next_seq;    /**< monotonic counter used to stamp the sequence key on push */
} parsec_binheap_t;

/**
 * Initialize an empty heap with the given priority-key offset.
 * @param[in] seq_offset byte offset of a uint64_t field reserved in the
 *            element type, used to break priority ties in FIFO order; pass
 *            PARSEC_HEAP_NO_SEQ to disable tie-breaking.
 */
static inline void parsec_heap_init(parsec_binheap_t *heap, size_t comp_offset, size_t seq_offset) {
    heap->top = NULL;
    heap->size = 0;
    heap->comp_offset = comp_offset;
    heap->seq_offset = seq_offset;
    heap->next_seq = 0;
}

/** Finalize heap (no-op: no allocation to free). */
static inline void parsec_heap_fini(parsec_binheap_t *heap) {
    (void)heap;
}

/** Return non-zero if the heap is empty. */
static inline int parsec_heap_is_empty(const parsec_binheap_t *heap) {
    return (heap->size == 0);
}

/** Return the number of elements. */
static inline size_t parsec_heap_size(const parsec_binheap_t *heap) {
    return heap->size;
}

/** View the maximum element without removing it. O(1). */
static inline parsec_list_item_t *parsec_heap_peek(const parsec_binheap_t *heap) {
    return heap->top;
}

/**
 * Insert one element. O(log N).
 * @return PARSEC_SUCCESS (cannot fail; no allocation is performed).
 */
int parsec_heap_push(parsec_binheap_t *heap, parsec_list_item_t *item);

/**
 * Remove and return the maximum element, or NULL if empty. O(log N).
 * The returned item's list_prev and list_next are reset to singleton state.
 */
parsec_list_item_t *parsec_heap_pop(parsec_binheap_t *heap);

/**
 * Batch-insert all elements from a chain or ring.
 * @return PARSEC_SUCCESS (cannot fail).
 */
int parsec_heap_push_chain(parsec_binheap_t *heap, parsec_list_item_t *chain);

END_C_DECLS

#endif /* PARSEC_HEAP_H_HAS_BEEN_INCLUDED */
