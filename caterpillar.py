import my_phylo as mp
import numpy as np

# --- Caterpillar builder ---------------------------------------------------
def build_caterpillar_tree(n_leaves: int, branch_length: float) -> mp.Node:
    """
    Build a rooted binary “caterpillar” tree with n_leaves leaves:
    the root has one leaf child and one internal child, which in turn
    has one leaf and one internal child, and so on, until the last
    internal node has two leaf children.
    """
    if n_leaves < 2:
        raise ValueError("n_leaves must be at least 2 for a caterpillar tree.")
    # create and label leaves
    leaves = [mp.Node(f"L{i}") for i in range(n_leaves)]
    for leaf in leaves:
        leaf.bl = [branch_length]
    # special case: just two leaves under root
    if n_leaves == 2:
        root = mp.Node()
        root.children = [leaves[0], leaves[1]]
        root.bl = [branch_length, branch_length]
        return root
    # build the spine
    root = mp.Node()
    first_internal = mp.Node()
    root.children = [leaves[0], first_internal]
    root.bl = [branch_length, branch_length]
    current = first_internal
    # attach leaves along the spine
    for i in range(1, n_leaves - 1):
        if i < n_leaves - 2:
            nxt = mp.Node()
            current.children = [leaves[i], nxt]
            current.bl = [branch_length, branch_length]
            current = nxt
        else:
            # final internal: attach the last two leaves
            current.children = [leaves[i], leaves[i+1]]
            current.bl = [branch_length, branch_length]
    return root

# --- Main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    theta = 0.25
    n_leaves = 8
    L1_vals = np.logspace(0.5, 2.5, 30, base=10, dtype=int).tolist()
    L2 = 100
    reps = 500

    mp.run_my_phylo(build_caterpillar_tree, reps=reps, L1_vals=L1_vals, rng=rng)

if __name__ == '__main__':
    main()
