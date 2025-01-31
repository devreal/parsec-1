#include <stdbool.h>

#include "parsec/parsec_config.h"

#include "parsec/class/parsec_rbtree.h"
#include "parsec/constants.h"

/* left and right children, using list pointers */
#define LEFT(node)  (*(parsec_rbtree_node_t**)&node->super.list_prev)
#define RIGHT(node) (*(parsec_rbtree_node_t**)&node->super.list_next)


/**
 * The list_item object instance.
 */
static inline void
parsec_rbtree_node_construct( parsec_rbtree_node_t* item )
{
    item->color = PARSEC_RBTREE_BLACK;
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_rbtree_node_t, parsec_list_item_t,
                          parsec_rbtree_node_construct, NULL);

void parsec_rbtree_init(parsec_rbtree_t* tree, size_t offset) {
    PARSEC_OBJ_CONSTRUCT(&tree->nil_element, parsec_rbtree_node_t);
    tree->nil = &tree->nil_element;
    tree->root = tree->nil;
    tree->comp_offset = offset;
}

void parsec_rbtree_fini(parsec_rbtree_t* tree) {
    PARSEC_OBJ_DESTRUCT(&tree->nil_element);
    tree->nil = NULL;
    tree->root = NULL;
    tree->comp_offset = 0;
}

static inline
void parsec_rbtree_left_rotate(parsec_rbtree_t *tree, parsec_rbtree_node_t *x) {
    parsec_rbtree_node_t *y = RIGHT(x);
    RIGHT(x) = LEFT(y);
    if (LEFT(y) != tree->nil) {
        LEFT(y)->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == tree->nil) {
        tree->root = y;
    } else if (x == LEFT(x->parent)) {
        LEFT(x->parent) = y;
    } else {
        RIGHT(x->parent) = y;
    }
    LEFT(y) = x;
    x->parent = y;
}

static inline
void parsec_rbtree_right_rotate(parsec_rbtree_t *tree, parsec_rbtree_node_t *y) {
    parsec_rbtree_node_t *x = LEFT(y);
    LEFT(y) = RIGHT(x);
    if (RIGHT(x) != tree->nil) {
        RIGHT(x)->parent = y;
    }
    x->parent = y->parent;
    if (y->parent == tree->nil) {
        tree->root = x;
    } else if (y == LEFT(y->parent)) {
        LEFT(y->parent) = x;
    } else {
        RIGHT(y->parent) = x;
    }
    RIGHT(x) = y;
    y->parent = x;
}

static void parsec_rbtree_insert_fixup(parsec_rbtree_t *tree, parsec_rbtree_node_t *z) {
    while (z->parent->color == PARSEC_RBTREE_RED) {
        if (z->parent == LEFT(z->parent->parent)) {
            parsec_rbtree_node_t *y = RIGHT(z->parent->parent);
            if (y->color == PARSEC_RBTREE_RED) {
                z->parent->color = PARSEC_RBTREE_BLACK;
                y->color = PARSEC_RBTREE_BLACK;
                z->parent->parent->color = PARSEC_RBTREE_RED;
                z = z->parent->parent;
            } else {
                if (z == RIGHT(z->parent)) {
                    z = z->parent;
                    parsec_rbtree_left_rotate(tree, z);
                }
                z->parent->color = PARSEC_RBTREE_BLACK;
                z->parent->parent->color = PARSEC_RBTREE_RED;
                parsec_rbtree_right_rotate(tree, z->parent->parent);
            }
        } else {
            parsec_rbtree_node_t *y = LEFT(z->parent->parent);
            if (y->color == PARSEC_RBTREE_RED) {
                z->parent->color = PARSEC_RBTREE_BLACK;
                y->color = PARSEC_RBTREE_BLACK;
                z->parent->parent->color = PARSEC_RBTREE_RED;
                z = z->parent->parent;
            } else {
                if (z == LEFT(z->parent)) {
                    z = z->parent;
                    parsec_rbtree_right_rotate(tree, z);
                }
                z->parent->color = PARSEC_RBTREE_BLACK;
                z->parent->parent->color = PARSEC_RBTREE_RED;
                parsec_rbtree_left_rotate(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = PARSEC_RBTREE_BLACK;
}

void parsec_rbtree_insert(parsec_rbtree_t *tree, parsec_rbtree_node_t *node) {
    node->color = PARSEC_RBTREE_RED;
    node->parent = tree->nil;
    parsec_rbtree_node_t *z = node;
    parsec_rbtree_node_t *y = tree->nil;
    parsec_rbtree_node_t *x = tree->root;
    while (x != tree->nil) {
        y = x;
        if (A_LOWER_PRIORITY_THAN_B(z, x, tree->comp_offset)) {
            x = LEFT(x);
        } else {
            x = RIGHT(x);
        }
    }
    z->parent = y;
    if (y == tree->nil) {
        tree->root = z;
    } else if (A_LOWER_PRIORITY_THAN_B(z, y, tree->comp_offset)) {
        LEFT(y) = z;
    } else {
        RIGHT(y) = z;
    }
    LEFT(z) = tree->nil;
    RIGHT(z) = tree->nil;
    z->color = PARSEC_RBTREE_RED;
    parsec_rbtree_insert_fixup(tree, z);
}

static void parsec_rbtree_delete_fixup(parsec_rbtree_t *tree, parsec_rbtree_node_t *x) {
    while (x != tree->root && x->color == PARSEC_RBTREE_BLACK) {
        if (x == LEFT(x->parent)) {
            parsec_rbtree_node_t *w = RIGHT(x->parent);
            if (w->color == PARSEC_RBTREE_RED) {
                w->color = PARSEC_RBTREE_BLACK;
                x->parent->color = PARSEC_RBTREE_RED;
                parsec_rbtree_left_rotate(tree, x->parent);
                w = RIGHT(x->parent);
            }
            if (LEFT(w)->color == PARSEC_RBTREE_BLACK && RIGHT(w)->color == PARSEC_RBTREE_BLACK) {
                w->color = PARSEC_RBTREE_RED;
                x = x->parent;
            } else {
                if (RIGHT(w)->color == PARSEC_RBTREE_BLACK) {
                    LEFT(w)->color = PARSEC_RBTREE_BLACK;
                    w->color = PARSEC_RBTREE_RED;
                    parsec_rbtree_right_rotate(tree, w);
                    w = RIGHT(x->parent);
                }
                w->color = x->parent->color;
                x->parent->color = PARSEC_RBTREE_BLACK;
                RIGHT(w)->color = PARSEC_RBTREE_BLACK;
                parsec_rbtree_left_rotate(tree, x->parent);
                x = tree->root;
            }
        } else {
            parsec_rbtree_node_t *w = LEFT(x->parent);
            if (w->color == PARSEC_RBTREE_RED) {
                w->color = PARSEC_RBTREE_BLACK;
                x->parent->color = PARSEC_RBTREE_RED;
                parsec_rbtree_right_rotate(tree, x->parent);
                w = LEFT(x->parent);
            }
            if (RIGHT(w)->color == PARSEC_RBTREE_BLACK && LEFT(w)->color == PARSEC_RBTREE_BLACK) {
                w->color = PARSEC_RBTREE_RED;
                x = x->parent;
            } else {
                if (LEFT(w)->color == PARSEC_RBTREE_BLACK) {
                    RIGHT(w)->color = PARSEC_RBTREE_BLACK;
                    w->color = PARSEC_RBTREE_RED;
                    parsec_rbtree_left_rotate(tree, w);
                    w = LEFT(x->parent);
                }
                w->color = x->parent->color;
                x->parent->color = PARSEC_RBTREE_BLACK;
                LEFT(w)->color = PARSEC_RBTREE_BLACK;
                parsec_rbtree_right_rotate(tree, x->parent);
                x = tree->root;
            }
        }
    }
    x->color = PARSEC_RBTREE_BLACK;
}

static inline void parsec_rbtree_transplant(parsec_rbtree_t *tree, parsec_rbtree_node_t *u, parsec_rbtree_node_t *v) {
    if (u->parent == tree->nil) {
        tree->root = v;
    } else if (u == LEFT(u->parent)) {
        LEFT(u->parent) = v;
    } else {
        RIGHT(u->parent) = v;
    }
    v->parent = u->parent;
}

parsec_rbtree_node_t* parsec_rbtree_minimum(parsec_rbtree_t *tree, parsec_rbtree_node_t *x) {
    while (LEFT(x) != tree->nil) {
        x = LEFT(x);
    }
    return x;
}

void parsec_rbtree_remove(parsec_rbtree_t *tree, parsec_rbtree_node_t *z) {
    parsec_rbtree_node_t *y = z;
    parsec_rbtree_node_t *x;
    parsec_rbtree_color_e y_original_color = y->color;
    if (LEFT(z) == tree->nil) {
        x = RIGHT(z);
        parsec_rbtree_transplant(tree, z, RIGHT(z));
    } else if (RIGHT(z) == tree->nil) {
        x = LEFT(z);
        parsec_rbtree_transplant(tree, z, LEFT(z));
    } else {
        y = parsec_rbtree_minimum(tree, RIGHT(z));
        y_original_color = y->color;
        x = RIGHT(y);
        if (y->parent == z) {
            x->parent = y;
        } else {
            parsec_rbtree_transplant(tree, y, RIGHT(y));
            RIGHT(y) = RIGHT(z);
            RIGHT(y)->parent = y;
        }
        parsec_rbtree_transplant(tree, z, y);
        LEFT(y) = LEFT(z);
        LEFT(y)->parent = y;
        y->color = z->color;
    }
    if (y_original_color == PARSEC_RBTREE_BLACK) {
        parsec_rbtree_delete_fixup(tree, x);
    }
}

parsec_rbtree_node_t* parsec_rbtree_find(parsec_rbtree_t *tree, int data) {
    parsec_rbtree_node_t *current = tree->root;
    while (current != tree->nil) {
        int compval = COMPARISON_VAL(current, tree->comp_offset);
        if (compval == data) {
            return current;
        } else if (compval < data) {
            current = RIGHT(current);
        } else {
            current = LEFT(current);
        }
    }
    return NULL; // data not found
}

parsec_rbtree_node_t* parsec_rbtree_find_or_larger(parsec_rbtree_t *tree, int data) {
    parsec_rbtree_node_t *current = tree->root;
    parsec_rbtree_node_t *larger  = tree->nil;
    while (current != tree->nil) {
        int compval = COMPARISON_VAL(current, tree->comp_offset);
        if (compval == data) {
            return current;
        } else if (compval < data) {
            current = RIGHT(current);
        } else {
            larger  = current;
            current = LEFT(current);
        }
    }
    if (larger == tree->nil) return NULL;
    return larger; // data not found
}

int parsec_rbtree_update_node(parsec_rbtree_t *tree, parsec_rbtree_node_t *node, int newdata) {
    parsec_rbtree_node_t *left = LEFT(node);
    parsec_rbtree_node_t *right = RIGHT(node);
    bool needs_reinsert = true;
    do {
        if (left != tree->nil && COMPARISON_VAL(left, tree->comp_offset) >= newdata) {
            /* left child is now larger or equal so reinsert */
            break;
        }
        if (right != tree->nil && COMPARISON_VAL(right, tree->comp_offset) <= newdata) {
            /* right child is now larger or equal so reinsert */
            break;
        }
        if (node->parent != tree->nil) {
            /**
             * Walk up the tree until we find one of the following:
             * 1) An ancestor has the same value as the new data. We need to reinsert.
             * 2) The node was a left child, an ancestor is smaller than the new data,
             *    and we're hanging off its right side (i.e., ordering is still correct).
             * 3) The node was a right child, an ancestor is larger than the new data,
             *    and we're hanging off its left side (i.e., ordering is still correct).
             */
            bool was_left_child = (LEFT(node->parent) == node);
            parsec_rbtree_node_t *tmp_node = node;
            parsec_rbtree_node_t *tmp_parent;
            do {
                tmp_parent = tmp_node->parent;
                int parent_data = COMPARISON_VAL(tmp_parent, tree->comp_offset);
                if (parent_data == newdata) {
                    /* the new data equals that of a parent node so we need to rebalance */
                    break;
                } else if (was_left_child &&
                        RIGHT(tmp_parent) == tmp_node &&
                        parent_data < newdata) {
                    /* The node value was smaller than its parent, we found an ancestor
                    * that is smaller than the new value, and we're on the right side
                    * of the ancestor. No reinserting needed. */
                    needs_reinsert = false;
                    break;
                } else if (LEFT(tmp_parent) == tmp_node &&
                        parent_data > newdata) {
                    /* The node value was larger than its parent, we found an ancestor
                    * that is larger than the new value, and we're on the left side
                    * of the ancestor. No reinserting needed. */
                    needs_reinsert = false;
                    break;
                }
                tmp_node = tmp_parent;
            } while (tmp_parent != tree->root);
        }
    } while (0);
    if (needs_reinsert) {
        parsec_rbtree_remove(tree, node);
        COMPARISON_VAL(node, tree->comp_offset) = newdata;
        parsec_rbtree_insert(tree, node);
    } else {
        COMPARISON_VAL(node, tree->comp_offset) = newdata;
    }

    return PARSEC_SUCCESS;
}

static void parsec_rbtree_foreach_node(
    parsec_rbtree_t* tree,
    parsec_rbtree_node_t* node,
    parsec_rbtree_visitor_cb *fn,
    void *cbdata)
{
    if (tree->nil != node) {
        parsec_rbtree_foreach_node(tree, LEFT(node), fn, cbdata);
        fn(node, cbdata);
        parsec_rbtree_foreach_node(tree, RIGHT(node), fn, cbdata);
    }
}

void parsec_rbtree_foreach(parsec_rbtree_t *tree, parsec_rbtree_visitor_cb *fn, void *cbdata) {
    if (NULL != tree) {
        parsec_rbtree_foreach_node(tree, tree->root, fn, cbdata);
    }
}
