import numpy as np
import my_phylo as mp
import balanced_binary as binary
import caterpillar
import yule

# Theta params
START_THETA = 0.7
END_THETA   = 0.8
NUM_THETAS  = 2

# Other
REPS        = 10
NUM_LEAVES  = 8
L1_VALS     = np.logspace(1, 3, 30, base=10, dtype=int).tolist()
L2_VAL      = 100
RNG         = np.random.default_rng(42)

def main():
    thetas = np.linspace(START_THETA, END_THETA, NUM_THETAS, dtype=float).tolist()

    for theta in thetas:
        print(theta)
        mp.run_my_phylo(
            build_tree = binary.build_balanced_tree,
            reps       = REPS,
            n_leaves   = NUM_LEAVES,
            theta      = theta,
            L1_vals    = L1_VALS,
            L2         = L2_VAL,
            rng        = RNG
        )


if __name__ == "__main__":
    main()
