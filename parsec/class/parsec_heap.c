/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/class/parsec_heap.h"
#include "parsec/constants.h"

#include <assert.h>

/* COMPARISON_VAL(ptr, offset) — reads *(int*) at byte offset inside ptr.
 * Defined in parsec_config_bottom.h, included transitively via parsec_config.h. */

/* left child  = list_prev
 * right child = list_next
 * Same convention as parsec/maxheap.c and parsec_rbtree.c. */
#define HLEFT(item)       ((parsec_list_item_t *)(item)->list_prev)
#define HRIGHT(item)      ((parsec_list_item_t *)(item)->list_next)
#define HSET_LEFT(it, v)  ((it)->list_prev = (volatile struct parsec_list_item_s *)(v))
#define HSET_RIGHT(it, v) ((it)->list_next = (volatile struct parsec_list_item_s *)(v))

static inline int heap_cmp(const parsec_heap_t *h,
                            const parsec_list_item_t *a,
                            const parsec_list_item_t *b)
{
    int va = COMPARISON_VAL(a, h->comp_offset);
    int vb = COMPARISON_VAL(b, h->comp_offset);
    return (va > vb) - (va < vb);
}

/* Maximum depth of the path array.  64 supports heaps of up to 2^64 elements. */
#define HEAP_MAX_DEPTH 64

int parsec_heap_push(parsec_heap_t *heap, parsec_list_item_t *item)
{
    HSET_LEFT(item, NULL);
    HSET_RIGHT(item, NULL);
    heap->size++;

    if (heap->size == 1) {
        heap->top = item;
        return PARSEC_SUCCESS;
    }

    /* Find the insertion point by following the bit path of 'size'.
     * After the leading 1-bit, each subsequent bit chooses right (1) or left (0).
     * Save ancestors for the sift-up pass.
     * (Same bit-navigation used by heap_insert in parsec/maxheap.c.) */
    parsec_list_item_t *path[HEAP_MAX_DEPTH];
    int depth = 0;

    size_t size = heap->size;
    size_t bitmask = 1;
    while (bitmask <= size) bitmask <<= 1;
    bitmask >>= 2;  /* position at bit just below the leading 1 */

    parsec_list_item_t *node = heap->top;
    path[depth++] = node;
    while (bitmask > 1) {
        node = (bitmask & size) ? HRIGHT(node) : HLEFT(node);
        path[depth++] = node;
        bitmask >>= 1;
    }
    /* Attach item as left (0) or right (1) child */
    if (bitmask & size) HSET_RIGHT(node, item);
    else                HSET_LEFT(node,  item);

    /* Sift up: walk from immediate parent (path[depth-1]) toward root */
    int level = depth - 1;
    while (level >= 0) {
        parsec_list_item_t *parent = path[level];
        if (heap_cmp(heap, item, parent) <= 0) break;

        /* Fix grandparent to point to item instead of parent */
        if (level > 0) {
            parsec_list_item_t *gp = path[level - 1];
            if (HLEFT(gp) == parent) HSET_LEFT(gp,  item);
            else                     HSET_RIGHT(gp, item);
        } else {
            heap->top = item;
        }

        /* Swap item and parent: item takes parent's position, parent takes item's */
        parsec_list_item_t *pl = HLEFT(parent);
        parsec_list_item_t *pr = HRIGHT(parent);
        HSET_LEFT(parent,  HLEFT(item));
        HSET_RIGHT(parent, HRIGHT(item));
        if (pl == item) {
            HSET_LEFT(item,  parent);
            HSET_RIGHT(item, pr);
        } else {
            HSET_LEFT(item,  pl);
            HSET_RIGHT(item, parent);
        }
        level--;
    }
    return PARSEC_SUCCESS;
}

parsec_list_item_t *parsec_heap_pop(parsec_heap_t *heap)
{
    if (0 == heap->size) return NULL;

    parsec_list_item_t *root = heap->top;

    if (heap->size == 1) {
        heap->top = NULL;
        heap->size = 0;
        PARSEC_LIST_ITEM_SINGLETON(root);
        return root;
    }

    /* Navigate to the parent of the 'last' node (rightmost node in the
     * bottom level), then detach it.  Track which side it was on so we
     * can correctly wire it into root's position even after clearing the
     * pointer. */
    size_t size = heap->size;
    size_t bitmask = 1;
    while (bitmask <= size) bitmask <<= 1;
    bitmask >>= 2;

    parsec_list_item_t *parent = heap->top;
    while (bitmask > 1) {
        parent = (bitmask & size) ? HRIGHT(parent) : HLEFT(parent);
        bitmask >>= 1;
    }

    parsec_list_item_t *last;
    int last_was_right = (int)(bitmask & size);
    if (last_was_right) {
        last = HRIGHT(parent);
        HSET_RIGHT(parent, NULL);
    } else {
        last = HLEFT(parent);
        HSET_LEFT(parent, NULL);
    }
    assert(last != NULL);

    /* Wire 'last' into root's place, inheriting root's children. */
    if (parent != root) {
        HSET_LEFT(last,  HLEFT(root));
        HSET_RIGHT(last, HRIGHT(root));
    } else {
        /* last was a direct child of root; one pointer was already cleared above */
        if (last_was_right) {
            HSET_LEFT(last,  HLEFT(root));  /* root's left is intact */
            HSET_RIGHT(last, NULL);         /* last is a leaf */
        } else {
            HSET_LEFT(last,  NULL);         /* last is a leaf */
            HSET_RIGHT(last, HRIGHT(root)); /* root's right is intact */
        }
    }
    heap->top = last;
    heap->size--;

    /* Sift down: swap last with the larger child until heap order is restored.
     * Track parent and which side we came from (no extra allocation needed). */
    parsec_list_item_t *bubbler = last;
    parsec_list_item_t *par = NULL;
    int from_right = 0;
    while (1) {
        parsec_list_item_t *left  = HLEFT(bubbler);
        parsec_list_item_t *right = HRIGHT(bubbler);
        int go_left  = (left  && heap_cmp(heap, left,  bubbler) > 0 &&
                        (!right || heap_cmp(heap, left,  right) >= 0));
        int go_right = (!go_left && right && heap_cmp(heap, right, bubbler) > 0);
        if (!go_left && !go_right) break;

        parsec_list_item_t *swap = go_left ? left : right;
        if (par) {
            if (from_right) HSET_RIGHT(par, swap);
            else            HSET_LEFT(par,  swap);
        } else {
            heap->top = swap;
        }
        HSET_LEFT(bubbler,  HLEFT(swap));
        HSET_RIGHT(bubbler, HRIGHT(swap));
        if (go_left) {
            HSET_LEFT(swap,  bubbler);
            HSET_RIGHT(swap, right);
            from_right = 0;
        } else {
            HSET_LEFT(swap,  left);
            HSET_RIGHT(swap, bubbler);
            from_right = 1;
        }
        par = swap;
    }

    PARSEC_LIST_ITEM_SINGLETON(root);
    return root;
}

int parsec_heap_push_chain(parsec_heap_t *heap, parsec_list_item_t *chain)
{
    parsec_list_item_t *item = chain;
    do {
        /* Capture list_next before parsec_heap_push repurposes it as right-child */
        parsec_list_item_t *next = (parsec_list_item_t *)item->list_next;
        parsec_heap_push(heap, item);
        item = next;
    } while (item != chain && item != NULL);
    return PARSEC_SUCCESS;
}
