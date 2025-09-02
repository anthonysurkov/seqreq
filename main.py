import numpy as np
import my_phylo as mp
import balanced_binary as binary
import caterpillar
import yule

# Theta params
START_THETA = 0.10
END_THETA   = 0.2
NUM_THETA   = 5

# Functions
FXN_LIST    = [
    binary.binary_tree,
    #caterpillar.caterpillar_tree,
    #yule.yule_tree
]

# Other
REPS        = 200
NUM_LEAVES  = 4 # <-- CONST
L1_VALS     = np.logspace(1.5, 4, 30, base=10, dtype=int).tolist()
L2_VAL      = 100
RNG         = np.random.default_rng(42)

def main():
    thetas = np.linspace(
        START_THETA, END_THETA, NUM_THETA, dtype=float
    ).tolist()

    for fxn in FXN_LIST:
        for theta in thetas:

            print(f"\n    FXN   = {fxn.__name__}")
            print(f"--- theta = {theta:.2f} ------------------------")

            mp.run_my_phylo(
                build_tree = fxn,
                reps       = REPS,
                n_leaves   = NUM_LEAVES,
                theta      = theta,
                L1_vals    = L1_VALS,
                L2         = L2_VAL,
                rng        = RNG
            )

if __name__ == "__main__":
    main()

