import my_phylo as mp
import numpy as np

# --- Caterpillar builder ---------------------------------------------------
def build_caterpillar_tree(n_leaves: int, theta: float) -> mp.Node:
    """
    Build a discrete‐CFN “caterpillar” tree on n_leaves,
    where every edge has correlation parameter theta.
    """
    if n_leaves < 2:
        raise ValueError("n_leaves must be at least 2 for a caterpillar tree.")

    # create and label leaves
    leaves = [mp.Node(f"L{i}") for i in range(n_leaves)]

    for leaf in leaves:
        leaf.bl = [theta]

    if n_leaves == 2: # edge case: just 2 leaves under root
        root = mp.Node()
        root.children = [leaves[0], leaves[1]]
        root.bl = [theta, theta]
        return root

    # start spine
    root = mp.Node()
    first_internal = mp.Node()
    root.children = [leaves[0], first_internal]
    root.bl = [theta, theta]

    # walk down the spine, attaching 1 leaf and 1 internal
    current = first_internal
    for i in range(1, n_leaves - 1):
        if i < n_leaves - 2:
            nxt = mp.Node()
            current.children = [leaves[i], nxt]
            current.bl = [theta, theta]
            current = nxt
        else:
            # final internal: attach the last two leaves
            current.children = [leaves[i], leaves[i+1]]
            current.bl = [theta, theta]
    return root

