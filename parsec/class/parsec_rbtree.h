/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */


#ifndef PARSEC_RBTREE_H
#define PARSEC_RBTREE_H


#include "parsec/class/list_item.h"


BEGIN_C_DECLS

typedef enum parsec_rbtree_color_e { PARSEC_RBTREE_RED, PARSEC_RBTREE_BLACK } parsec_rbtree_color_e;

typedef struct parsec_rbtree_node_t {
    parsec_list_item_t super; // use prev/next for left/right
    parsec_rbtree_color_e color;
    struct parsec_rbtree_node_t *parent;
} parsec_rbtree_node_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_rbtree_node_t);

typedef struct parsec_rbtree_t {
    parsec_rbtree_node_t nil_element;
    parsec_rbtree_node_t *root;
    parsec_rbtree_node_t *nil;
    size_t comp_offset;
    size_t count;   /**< number of nodes currently in the tree */
} parsec_rbtree_t;

typedef void (parsec_rbtree_visitor_cb)(parsec_rbtree_node_t*, void*);

void parsec_rbtree_init(parsec_rbtree_t *tree, size_t compare_offset);

void parsec_rbtree_fini(parsec_rbtree_t* tree);

void parsec_rbtree_insert(parsec_rbtree_t *tree, parsec_rbtree_node_t *node);

parsec_rbtree_node_t* parsec_rbtree_minimum(parsec_rbtree_t *tree, parsec_rbtree_node_t *x);

parsec_rbtree_node_t* parsec_rbtree_maximum(parsec_rbtree_t *tree, parsec_rbtree_node_t *x);

void parsec_rbtree_remove(parsec_rbtree_t *tree, parsec_rbtree_node_t *z);

parsec_rbtree_node_t* parsec_rbtree_find(parsec_rbtree_t *tree, int data);

parsec_rbtree_node_t* parsec_rbtree_find_or_larger(parsec_rbtree_t *tree, int data);

int parsec_rbtree_update_node(parsec_rbtree_t *tree, parsec_rbtree_node_t *node, int newdata);

void parsec_rbtree_foreach(parsec_rbtree_t *tree, parsec_rbtree_visitor_cb *fn, void *cbdata);

/**
 * Bulk-insert a doubly-linked ring of rbtree nodes into the tree.
 *
 * When the ring is small relative to the existing tree (ring count <
 * log2(tree count + 1)) the nodes are inserted individually via
 * parsec_rbtree_insert.  Otherwise the tree is rebuilt from a merged
 * sorted sequence in O(N log N + M) time with zero per-node rotations.
 *
 * Every node in the ring must have its comparison key (at comp_offset)
 * already set before this call.  The ring's list_prev/list_next links
 * are consumed and overwritten; do not use them after this call.
 */
void parsec_rbtree_insert_ring(parsec_rbtree_t *tree, parsec_rbtree_node_t *ring);

END_C_DECLS

#endif // PARSEC_RBTREE_H