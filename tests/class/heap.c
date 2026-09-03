/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

/* Regression tests for parsec/class/parsec_heap.c, covering the two P1
 * issues raised in review of PR#784 (device task priority heap):
 *   - parsec_lifo_detach_chain() must hand back a chain whose list_next
 *     links are still walkable (not poisoned) so parsec_heap_push_chain()
 *     can traverse it, in both PARANOID and non-PARANOID builds.
 *   - the heap must break priority ties in FIFO order so that a continuous
 *     stream of equal-priority insertions cannot starve elements already
 *     queued.
 */

#include "parsec/runtime.h"
#undef NDEBUG
#include <stdarg.h>
#include <signal.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

#include "parsec/class/lifo.h"
#include "parsec/class/parsec_heap.h"

static void fatal(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    vprintf(format, va);
    va_end(va);
    raise(SIGABRT);
}

typedef struct {
    parsec_list_item_t list;
    int32_t             priority;
    uint64_t            seq;
    int                 id;
} elt_t;

static elt_t *make_elt(int id, int32_t priority)
{
    elt_t *e = (elt_t *)malloc(sizeof(elt_t));
    PARSEC_OBJ_CONSTRUCT(e, parsec_list_item_t);
    e->priority = priority;
    e->seq = 0;
    e->id = id;
    return e;
}

/* Push a steady stream of equal-priority elements while popping, and check
 * that the heap drains in strict arrival (FIFO) order: no element is ever
 * overtaken by one that arrived later. */
static void test_fifo_fairness(void)
{
    parsec_binheap_t heap;
    int next_push = 0, next_pop = 0;
    int i;

    printf(" - equal-priority insertions must drain in FIFO order (no starvation)\n");
    parsec_heap_init(&heap, offsetof(elt_t, priority), offsetof(elt_t, seq));

    /* seed a backlog */
    for (i = 0; i < 64; i++) {
        parsec_heap_push(&heap, (parsec_list_item_t *)make_elt(next_push++, 0));
    }

    /* continuously interleave new arrivals with pops (at least one arrival
     * per pop, matching the reported starvation scenario): the backlog
     * must still drain in arrival order instead of being perpetually
     * overtaken by fresh, same-priority work. */
    for (i = 0; i < 100000; i++) {
        parsec_heap_push(&heap, (parsec_list_item_t *)make_elt(next_push++, 0));
        if (0 == (i % 5)) {
            parsec_heap_push(&heap, (parsec_list_item_t *)make_elt(next_push++, 0));
        }
        elt_t *popped = (elt_t *)parsec_heap_pop(&heap);
        if (NULL == popped)
            fatal(" ! Error: heap unexpectedly empty at iteration %d\n", i);
        if (popped->id != next_pop)
            fatal(" ! Error: FIFO order violated: expected id %d, got %d (starvation)\n",
                  next_pop, popped->id);
        next_pop++;
        free(popped);
    }

    /* drain the rest */
    while (!parsec_heap_is_empty(&heap)) {
        elt_t *popped = (elt_t *)parsec_heap_pop(&heap);
        if (popped->id != next_pop)
            fatal(" ! Error: FIFO order violated on drain: expected id %d, got %d\n",
                  next_pop, popped->id);
        next_pop++;
        free(popped);
    }
    if (next_pop != next_push)
        fatal(" ! Error: expected to pop %d elements, popped %d\n", next_push, next_pop);
    parsec_heap_fini(&heap);
    printf("   ok (%d elements drained in order)\n", next_pop);
}

/* Basic max-heap sanity check with distinct priorities pushed in random
 * order: pop order must be non-increasing in priority. */
static void test_priority_order(void)
{
    parsec_binheap_t heap;
    const int N = 4096;
    int32_t last_priority = INT32_MAX;
    int i, count = 0;

    printf(" - distinct priorities must be popped in non-increasing order\n");
    parsec_heap_init(&heap, offsetof(elt_t, priority), offsetof(elt_t, seq));

    for (i = 0; i < N; i++) {
        int32_t priority = (int32_t)(rand() % (2 * N));
        parsec_heap_push(&heap, (parsec_list_item_t *)make_elt(i, priority));
    }

    while (!parsec_heap_is_empty(&heap)) {
        elt_t *popped = (elt_t *)parsec_heap_pop(&heap);
        if (popped->priority > last_priority)
            fatal(" ! Error: priority order violated: %d popped after %d\n",
                  popped->priority, last_priority);
        last_priority = popped->priority;
        count++;
        free(popped);
    }
    if (count != N)
        fatal(" ! Error: expected %d elements, popped %d\n", N, count);
    parsec_heap_fini(&heap);
    printf("   ok (%d elements)\n", count);
}

/* Regression test for the detach_chain / push_chain pipeline used by the
 * device management thread: push a batch of elements into a LIFO, detach
 * the whole chain at once, and hand it to parsec_heap_push_chain(). This
 * traversal must see valid list_next links end-to-end (they must not be
 * poisoned by the paranoid detach bookkeeping), and every element must
 * come out of the heap exactly once. */
static void test_lifo_detach_to_heap_chain(void)
{
    parsec_lifo_t lifo;
    parsec_binheap_t heap;
    const int N = 2048;
    unsigned char *seen;
    int i, popped_count = 0;

    printf(" - parsec_lifo_detach_chain() -> parsec_heap_push_chain() must preserve every element\n");
    PARSEC_OBJ_CONSTRUCT(&lifo, parsec_lifo_t);
    parsec_heap_init(&heap, offsetof(elt_t, priority), offsetof(elt_t, seq));

    for (i = 0; i < N; i++) {
        elt_t *e = (elt_t *)parsec_lifo_item_alloc(&lifo, sizeof(elt_t));
        e->priority = 0;
        e->seq = 0;
        e->id = i;
        parsec_lifo_push(&lifo, (parsec_list_item_t *)e);
    }

    parsec_list_item_t *chain = parsec_lifo_detach_chain(&lifo);
    if (NULL == chain)
        fatal(" ! Error: parsec_lifo_detach_chain() returned NULL for a non-empty LIFO\n");
    parsec_heap_push_chain(&heap, chain);

    seen = (unsigned char *)calloc(1, N);
    while (!parsec_heap_is_empty(&heap)) {
        elt_t *e = (elt_t *)parsec_heap_pop(&heap);
        if (e->id < 0 || e->id >= N)
            fatal(" ! Error: popped element with corrupt id %d\n", e->id);
        if (seen[e->id])
            fatal(" ! Error: element %d popped twice (chain traversal corruption)\n", e->id);
        seen[e->id] = 1;
        popped_count++;
        parsec_lifo_item_free((parsec_list_item_t *)e);
    }
    free(seen);
    if (popped_count != N)
        fatal(" ! Error: expected %d elements out of the heap, got %d\n", N, popped_count);
    parsec_heap_fini(&heap);
    PARSEC_OBJ_DESTRUCT(&lifo);
    printf("   ok (%d elements survived detach_chain -> push_chain)\n", popped_count);
}


int main(int argc, char *argv[])
{
#if defined(PARSEC_HAVE_MPI)
    MPI_Init(&argc, &argv);
#else
    (void)argc; (void)argv;
#endif

    test_priority_order();
    test_fifo_fairness();
    test_lifo_detach_to_heap_chain();

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return 0;
}
