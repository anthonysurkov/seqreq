import my_phylo as mp
import numpy as np

# --- Balanced tree builder --------------------------------------------------
def build_balanced_tree(n_leaves: int, branch_length: float) -> mp.Node:
    if n_leaves & (n_leaves - 1) != 0:
        raise ValueError("n_leaves must be a power of two for a perfect balanced tree.")
    leaves = [mp.Node(f"L{i}") for i in range(n_leaves)]
    level = leaves[:]
    while len(level) > 1:
        next_level: List[Node] = []
        for i in range(0, len(level), 2):
            left, right = level[i], level[i+1]
            parent = mp.Node()
            parent.children = [left, right]
            parent.bl = [branch_length, branch_length]
            next_level.append(parent)
        level = next_level
    return level[0]

# --- Main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    theta = 0.8
    n_leaves = 8
    L1_vals = np.logspace(1, 3, 30, base=10, dtype=int).tolist()
    L2 = 100
    reps = 100

    mp.run_my_phylo(build_balanced_tree, reps=reps, L1_vals=L1_vals, rng=rng)


if __name__ == '__main__':
    main()
