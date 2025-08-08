import my_phylo as mp
import numpy as np

# --- Yule tree builder -------------------------------------------------------
def yule_tree(n_leaves: int, branch_length: float, rng) -> mp.Node:
    """
    Pure‐birth (Yule) tree: start with one lineage, then
  repeatedly pick one random leaf to speciate until we
    have n_leaves. All branch‐lengths are set to branch_length.
    """
    if n_leaves < 1:
        raise ValueError("n_leaves must be at least 1")
    # initialize root as the only lineage
    root = mp.Node("L0")
    root.bl = [branch_length]       # length from its (nonexistent) parent
    leaves = [root]                 # current extant lineages
    parent_map = {root: None}       # to rewire when we split

    next_idx = 1
    while len(leaves) < n_leaves:
        # pick a random extant leaf to speciate
        leaf = rng.choice(leaves)
        # create its new sibling
        new_leaf = mp.Node(f"L{next_idx}")
        new_leaf.bl = [branch_length]
        next_idx += 1

        # create new internal node in place of `leaf`
        new_internal = mp.Node()
        new_internal.children = [leaf, new_leaf]
        new_internal.bl = [branch_length, branch_length]

        # rewire into the existing tree
        parent = parent_map[leaf]
        parent_map[leaf]      = new_internal
        parent_map[new_leaf]  = new_internal
        parent_map[new_internal] = parent

        if parent is None:
            # leaf was root, so new_internal becomes new root
            root = new_internal
        else:
            # replace `leaf` by `new_internal` in its parent
            for i, ch in enumerate(parent.children):
                if ch is leaf:
                    parent.children[i] = new_internal
                    parent.bl[i]       = branch_length
                    break

        # update the extant‐lineages list
        leaves.remove(leaf)
        leaves.append(leaf)      # leaf remains a tip under new_internal
        leaves.append(new_leaf)

    return root

