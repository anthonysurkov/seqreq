import my_phylo as mp
import numpy as np

# --- Pref. Attachment tree builder ------------------------------------------
def build_pa_tree(n_leaves: int, branch_length: float, rng) -> mp.Node:
    """
    Pure‑birth preferential‑attachment shape:
    new leaves attach to existing nodes with probability ∝ (num_children + 1).
    """
    # start with root
    root = mp.Node("L0")
    root.bl = [branch_length]
    nodes = [root]
    next_i = 1

    while len(nodes) < n_leaves:
        # weight every node by (degree + 1) so no zeros
        weights = np.array([len(nd.children) + 1 for nd in nodes], dtype=float)
        probs   = weights / weights.sum()      # sum>0 so no NaN
        idx     = rng.choice(len(nodes), p=probs)
        attach  = nodes[idx]

        # create the new leaf
        new_leaf = mp.Node(f"L{next_i}")
        new_leaf.bl = [branch_length]
        next_i += 1

        # hang it off `attach`
        attach.children.append(new_leaf)
        attach.bl.append(branch_length)

        # add to the pool
        nodes.append(new_leaf)

    return root

# --- Main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    theta = 0.25
    n_leaves = 16
    L1_vals = np.logspace(1.0, 2.5, 30, base=10, dtype=int).tolist()
    L2 = 100
    reps = 500

    mp.run_my_phylo(build_pa_tree, reps=reps, L1_vals=L1_vals, rng=rng)


if __name__ == '__main__':
    main()
