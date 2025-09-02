import os, math, tempfile, shutil, subprocess, inspect, re, json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Tuple, Set
from numpy.random import Generator

# --- Julia bridge -------------------------------------------------------------
JULIA = os.environ.get("JULIA", "julia")
ENV_PATH = os.path.expanduser("~/.julia_envs/fourleaf_env")
SERVER = os.path.join(os.path.dirname(__file__), "fourleaf_server.jl")

def start_fourleaf_server():
    env = os.environ.copy()
    env.setdefault("JULIA_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("JULIA_PKG_SERVER", "")
    p = subprocess.Popen(
        [JULIA, f"--project={ENV_PATH}", "--startup-file=no", SERVER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1
    )
    return p

class FourLeafClient:
    def __init__(self):
        self.p = start_fourleaf_server()
    def infer(self, vec8):
        line = json.dumps([int(x) for x in vec8])
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().strip()
        return json.loads(out)
    def close(self):
        try:
            self.p.stdin.close()
        finally:
            self.p.terminate()

# --- Node def'n ---------------------------------------------------------------
class Node:
    __slots__ = ("children", "bl", "seq", "name")
    def __init__(self, name: str | None = None):
        self.children: List[Node] = []
        self.bl: List[float] = []           # store continuous-time branch lengths θ
        self.seq: np.ndarray | None = None  # 0/1 sequence
        self.name: str | None = name

    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> List["Node"]:
        if self.is_leaf():
            return [self]
        out: List["Node"] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

# --- Helper: apply θ̂ onto a topology ----------------------------------------
def _apply_theta_hat(true_tree: Node, order: list[str], theta_hat: list[float]) -> None:
    # pendant edges: map name -> θ
    theta_pendant = {name: float(theta_hat[i]) for i, name in enumerate(order)}

    def dfs_assign_pendants(n: Node):
        for i, c in enumerate(n.children):
            if c.is_leaf():
                n.bl[i] = theta_pendant[c.name]
            dfs_assign_pendants(c)
    dfs_assign_pendants(true_tree)

    # internal edge: set the unique internal->internal edge
    def is_leaf_name_set(root: Node):
        return {lf.name for lf in root.leaves()}
    leaves = is_leaf_name_set(true_tree)

    theta_internal = float(theta_hat[4])
    found = False
    def dfs_assign_internal(n: Node):
        nonlocal found
        for i, c in enumerate(n.children):
            if found: return
            u_is_leaf = n.name in leaves
            v_is_leaf = c.name in leaves
            if (not u_is_leaf) and (not v_is_leaf):
                n.bl[i] = theta_internal
                found = True
                return
            dfs_assign_internal(c)
    dfs_assign_internal(true_tree)
    if not found:
        raise ValueError("Quartet internal edge not found; is this a quartet?")

# --- CFN simulation (continuous time) -----------------------------------------
def _cfn_probs(theta: float) -> tuple[float, float]:
    """Return (stay_prob, flip_prob) for CFN at branch length θ."""
    e = math.exp(-2.0 * float(theta))
    stay = 0.5 * (1.0 + e)
    flip = 0.5 * (1.0 - e)
    return stay, flip

def simulate_cfn_ct(node: Node, rng: Generator) -> None:
    """Simulate down the tree using continuous-time CFN with per-edge θ in node.bl."""
    if node.is_leaf():
        return
    L = len(node.seq)
    for child, theta in zip(node.children, node.bl):
        stay, flip = _cfn_probs(theta)
        u = rng.random(L)
        seq = node.seq.copy()
        flips = np.where(u < flip)[0]
        seq[flips] = 1 - seq[flips]
        child.seq = seq
        simulate_cfn_ct(child, rng)

# --- 8-bin (complement-only) counter -----------------------------------------
def vec8_from_matrix(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.int64, copy=False)
    ints = (M[0] << 3) | (M[1] << 2) | (M[2] << 1) | M[3]
    reps = np.minimum(ints, ints ^ 0xF)
    v8 = np.bincount(reps % 8, minlength=8)
    return v8.astype(int)

def get_pattern_vector(leaves) -> np.ndarray:
    order, M = quartet_matrix(leaves)
    return vec8_from_matrix(M)

def quartet_matrix(leaves):
    order = sorted([lf.name for lf in leaves])
    name_to_leaf = {lf.name: lf for lf in leaves}
    M = np.vstack([name_to_leaf[name].seq.astype(int) for name in order])
    return order, M

# --- Splits utilities ---------------------------------------------------------
def splits(tree: Node) -> Set[frozenset[str]]:
    leaves = {leaf.name for leaf in tree.leaves()}
    raw: Set[frozenset[str]] = set()
    def collect(n: Node, p: Node | None) -> Set[str]:
        if n.is_leaf(): return {n.name}
        s: Set[str] = set()
        for c in n.children:
            s |= collect(c, n)
        if p is not None and 1 < len(s) < len(leaves) - 1:
            raw.add(frozenset(s))
        return s
    collect(tree, None)
    sym: Set[frozenset[str]] = set()
    for s in raw:
        sym.update([frozenset(s), frozenset(leaves - set(s))])
    return sym

def canonical_split(groupA, groupB) -> frozenset[frozenset[str]]:
    A = frozenset(groupA); B = frozenset(groupB)
    return frozenset([A, B])

def find_true_split(tree: Node) -> frozenset[frozenset[str]]:
    all_leaves = {lf.name for lf in tree.leaves()}
    for s in splits(tree):
        if len(s) == 2:
            return canonical_split(s, all_leaves - set(s))
    raise ValueError("Did not find a 2|2 split - is the tree a quartet?")

# --- Parse FourLeafMLE's output into a split ----------------------------------
def parse_fourleaf(desc: str, order: list[str]) -> frozenset[frozenset[str]]:
    m = re.search(r'(?:τ|tau)\s*=\s*([1234]+)\s*\|\s*([1234]+)', desc, re.IGNORECASE)
    if m:
        left = [order[int(d) - 1] for d in m.group(1)]
        right = [order[int(d) - 1] for d in m.group(2)]
        return canonical_split(left, right)
    m = re.search(r'\(\s*([1-4])\s*--\s*([1-4])\s*--<\s*([1-4])\s*,\s*([1-4])\s*\)', desc)
    if m:
        a,b,c,d = (int(m.group(i)) for i in range(1,5))
        left  = [order[a-1], order[b-1]]
        right = [order[c-1], order[d-1]]
        return canonical_split(left, right)
    m = re.search(r'\b([1234]{2})\|([1234]{2})\b', desc)
    if m:
        left = [order[int(d) - 1] for d in m.group(1)]
        right = [order[int(d) - 1] for d in m.group(2)]
        return canonical_split(left, right)
    raise ValueError(f"Could not parse split from FourLeafMLE desc: {desc!r}")

# --- Rooting: Majority Rule ---------------------------------------------------
def root_by_MR(tree: Node, rng: Generator) -> np.ndarray:
    leaves = tree.leaves()
    seqs = np.array([leaf.seq for leaf in leaves])
    L = seqs.shape[1]
    root_pred = np.empty(L, dtype=int)
    for i in range(L):
        column = seqs[:, i]
        values, counts = np.unique(column, return_counts=True)
        max_count = counts.max()
        modes = values[counts == max_count]
        root_pred[i] = rng.choice(modes)
    return root_pred

# --- Helpers for θ error ------------------------------------------------------
def _edges_with_theta(root: Node):
    edges = []
    def dfs(n: Node):
        for i,c in enumerate(n.children):
            edges.append((n, c, float(n.bl[i])))
            dfs(c)
    dfs(root); return edges

def _pendant_theta_by_leaf(root: Node) -> dict[str, float]:
    is_leaf = {lf.name for lf in root.leaves()}
    out = {}
    for u,v,th in _edges_with_theta(root):
        if v.name in is_leaf: out[v.name] = th
        if u.name in is_leaf: out[u.name] = th
    return out

def _internal_edge_theta(root: Node) -> float:
    is_leaf = {lf.name for lf in root.leaves()}
    for u,v,th in _edges_with_theta(root):
        if u.name not in is_leaf and v.name not in is_leaf:
            return th
    raise ValueError("Internal edge not found.")

def agg_theta_err(true_tree: Node, order: list[str], theta_hat: list[float]) -> float:
    pend = _pendant_theta_by_leaf(true_tree)
    theta_true_vec = [pend[name] for name in order] + [_internal_edge_theta(true_tree)]
    return float(np.mean(np.abs(np.array(theta_true_vec) - np.array(theta_hat, dtype=float))))

# --- Build inferred quartet from (split, θ̂) ----------------------------------
def build_inferred_quartet(order: list[str],
                           split_hat: frozenset[frozenset[str]],
                           theta_hat: list[float]) -> Node:
    left, right = [list(s) for s in split_hat]
    theta_pendant = {name: float(theta_hat[i]) for i, name in enumerate(order)}
    theta_int = float(theta_hat[4])

    root = Node("ROOT")
    lint = Node("LINT"); rint = Node("RINT")
    root.children = [lint, rint]; root.bl = [theta_int, theta_int]

    lint.children = [Node(left[0]), Node(left[1])]
    lint.bl = [theta_pendant[left[0]], theta_pendant[left[1]]]

    rint.children = [Node(right[0]), Node(right[1])]
    rint.bl = [theta_pendant[right[0]], theta_pendant[right[1]]]
    return root

# --- Pruning likelihood (continuous-time CFN) ---------------------------------
def _prune_lik_edgewise_ct(node: Node) -> np.ndarray:
    if node.is_leaf():
        L = len(node.seq)
        M = np.empty((2, L), dtype=float)
        M[0, :] = (node.seq == 0)
        M[1, :] = (node.seq == 1)
        return M

    child_mats = [_prune_lik_edgewise_ct(c) for c in node.children]
    L = child_mats[0].shape[1]
    out = np.ones((2, L), dtype=float)

    for mat, theta in zip(child_mats, node.bl):
        stay, flip = _cfn_probs(theta)
        # transitions: 0->0 and 1->1 use 'stay', off-diagonals use 'flip'
        out0 = stay * mat[0, :] + flip * mat[1, :]
        out1 = stay * mat[1, :] + flip * mat[0, :]
        out[0, :] *= out0
        out[1, :] *= out1

    return out

def root_by_ML_ct(tree: Node) -> np.ndarray:
    Lm = _prune_lik_edgewise_ct(tree)
    return np.argmax(Lm, axis=0)

# --- Single experimental replicate --------------------------------------------
def run_rep(build_tree, L1: int, L2: int,
            n_leaves: int, theta_true: float, rng: Generator, fl: FourLeafClient
           ) -> Tuple[bool, float, float, float, float] | str:

    # Simulate T1 for inference (builder should set per-edge θ in .bl)
    if 'rng' in inspect.signature(build_tree).parameters:
        T_true = build_tree(n_leaves, theta_true, rng=rng)
    else:
        T_true = build_tree(n_leaves, theta_true)
    T_true.seq = rng.integers(0, 2, size=L1)
    simulate_cfn_ct(T_true, rng)

    # estimate params and split from FourLeafMLE
    order, M = quartet_matrix(T_true.leaves())
    vec8 = vec8_from_matrix(M)
    jl_out = fl.infer(vec8)
    desc = jl_out.get("desc", "")
    raw_a = jl_out.get("params", [])
    if not isinstance(raw_a, (list, tuple)) or len(raw_a) != 5:
        return "SKIP"

    def a_to_theta(a):
        a = float(max(1e-12, min(1-1e-12, a)))
        return -0.5 * math.log(a)

    theta_hat = [a_to_theta(a) for a in raw_a]
    split_hat = parse_fourleaf(desc, order)

    # Fresh T2 for evaluation
    T_eval = deepcopy(T_true)
    T_eval.seq = rng.integers(0, 2, size=L2)
    simulate_cfn_ct(T_eval, rng)

    # Split match + aggregate θ error
    split_true = find_true_split(T_true)
    split_match = (split_true == split_hat)
    theta_err = agg_theta_err(T_true, order, theta_hat)

    # Root ML on TRUE topology with θ̂
    seqmap = {lf.name: lf.seq for lf in T_eval.leaves()}
    T_true_w_thetahat = deepcopy(T_true)
    _apply_theta_hat(T_true_w_thetahat, order, theta_hat)
    for lf in T_true_w_thetahat.leaves():
        lf.seq = seqmap[lf.name]
    root_true_thetahat = root_by_ML_ct(T_true_w_thetahat)
    acc_ml_true_thetahat = float((root_true_thetahat == T_eval.seq).mean())

    # Root ML on INFERRED topology with θ̂
    T_hat = build_inferred_quartet(order, split_hat, theta_hat)
    for lf in T_hat.leaves():
        lf.seq = seqmap[lf.name]
    root_inf = root_by_ML_ct(T_hat)
    acc_ml_inf = float((root_inf == T_eval.seq).mean())

    # MR baseline
    root_mr = root_by_MR(T_eval, rng)
    acc_mr = float((root_mr == T_eval.seq).mean())

    return split_match, theta_err, acc_ml_true_thetahat, acc_ml_inf, acc_mr

# --- Main logic ---------------------------------------------------------------
def run_my_phylo(build_tree, reps=500, n_leaves=4, theta=0.1,
                 L1_vals=None, L2=100, rng=None):
    if L1_vals is None:
        L1_vals = np.logspace(1, 3, 30, base=10, dtype=int).tolist()
    L1_vals = [L for L in L1_vals if L != 4]

    fl = FourLeafClient()

    def mean_or_nan(x): 
        return float(np.mean(x)) if x else float('nan')

    header = (
        f"{'L1':>6} | {'split_P':>7} | {'agg_θ_err':>10} | "
        f"{'ML_true(θ̂)':>12} | {'ML_inf':>7} | {'MR':>7} | {'skips':>8}"
    )
    print(header); print("-" * len(header))

    splitP_vals, agg_theta_err_vals = [], []
    ml_true_thetahat_vals, ml_inf_vals, mr_vals = [], [], []
    skip_rate_vals, acc_if_good_vals, acc_if_bad_vals = [], [], []

    try:
        for L1 in L1_vals:
            split_hits, theta_errs, ml_true_accs, ml_inf_accs, mr_accs = [], [], [], [], []
            inf_when_good, inf_when_bad = [], []
            skipped = 0

            for _ in range(reps):
                res = run_rep(build_tree, L1, L2, n_leaves, theta, rng, fl)
                if res == "SKIP":
                    skipped += 1
                    continue
                ok, terr, a_true, a_inf, a_mr = res
                split_hits.append(ok)
                theta_errs.append(terr)
                ml_true_accs.append(a_true)
                ml_inf_accs.append(a_inf)
                mr_accs.append(a_mr)
                (inf_when_good if ok else inf_when_bad).append(a_inf)

            print(
                f"{L1:6d} | "
                f"{mean_or_nan(split_hits):7.3f} | "
                f"{mean_or_nan(theta_errs):10.4f} | "
                f"{mean_or_nan(ml_true_accs):12.3f} | "
                f"{mean_or_nan(ml_inf_accs):7.3f} | "
                f"{mean_or_nan(mr_accs):7.3f} | "
                f"{skipped:3d}/{reps:<3d} ({skipped/reps:>4.1%})"
            )

            splitP_vals.append(mean_or_nan(split_hits))
            agg_theta_err_vals.append(mean_or_nan(theta_errs))
            ml_true_thetahat_vals.append(mean_or_nan(ml_true_accs))
            ml_inf_vals.append(mean_or_nan(ml_inf_accs))
            mr_vals.append(mean_or_nan(mr_accs))
            skip_rate_vals.append(skipped / reps if reps else float('nan'))
            acc_if_good_vals.append(mean_or_nan(inf_when_good))
            acc_if_bad_vals.append(mean_or_nan(inf_when_bad))

        # -------------------- PLOTTING --------------------
        def _savefig(path):
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        this_dir = f"plots/{build_tree.__name__}/theta{theta:.3f}"
        os.makedirs(this_dir, exist_ok=True)
        try:
            shutil.copyfile("utils/open_all.py", f"{this_dir}/open_all.py")
        except Exception:
            pass

        plt.figure(figsize=(6,4))
        plt.plot(L1_vals, splitP_vals, "o-")
        plt.xscale("log"); plt.ylim(0, 1.01)
        plt.xlabel("Sequence length for inference L1")
        plt.ylabel("P(correct split)")
        plt.title("Probability of Correct Quartet Split vs L1")
        plt.grid(True, alpha=0.3)
        _savefig(f"{this_dir}/split_probability.png")

        plt.figure(figsize=(6,4))
        plt.plot(L1_vals, agg_theta_err_vals, "o-")
        plt.xscale("log")
        plt.xlabel("L1")
        plt.ylabel("Mean |θ̂ − θ| (5-edge avg)")
        plt.title("Aggregate Branch-Length Error vs L1")
        plt.grid(True, alpha=0.3)
        _savefig(f"{this_dir}/agg_theta_err.png")

        plt.figure(figsize=(6,4))
        plt.plot(L1_vals, ml_true_thetahat_vals, "o-", label="ML (true topo, θ̂)")
        plt.plot(L1_vals, ml_inf_vals,          "s-", label="ML (inferred topo, θ̂)")
        plt.plot(L1_vals, mr_vals,              "x--", label="Majority rule")
        plt.xscale("log"); plt.ylim(0, 1.01)
        plt.xlabel("L1"); plt.ylabel("Root accuracy")
        plt.title("Overall Root Reconstruction Accuracy vs L1")
        plt.legend(); plt.grid(True, alpha=0.3)
        _savefig(f"{this_dir}/overall_root_accuracy.png")

        fig, axes = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.plot(L1_vals, ml_true_thetahat_vals, "o-", label="ML (true topo, θ̂)")
        ax1.plot(L1_vals, ml_inf_vals,           "s-", label="ML (inferred, θ̂)")
        ax1.plot(L1_vals, mr_vals,               "x--", label="MR")
        ax1.set_xscale("log"); ax1.set_ylim(0,1.01)
        ax1.set_title("Root Accuracy vs L1"); ax1.set_xlabel("L1"); ax1.set_ylabel("Accuracy")
        ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(L1_vals, agg_theta_err_vals, "o-")
        ax2.set_xscale("log")
        ax2.set_title("Aggregate θ-error"); ax2.set_xlabel("L1"); ax2.set_ylabel("mean |θ̂−θ|")
        ax2.grid(True, alpha=0.3)

        delta = np.array(ml_true_thetahat_vals) - np.array(ml_inf_vals)
        ax3.plot(L1_vals, delta, "o-")
        ax3.set_xscale("log")
        ax3.set_title("Loss from Topology Error")
        ax3.set_xlabel("L1"); ax3.set_ylabel("ML(true,θ̂) − ML(inferred,θ̂)")
        ax3.grid(True, alpha=0.3)

        ax4.plot(L1_vals, np.array(skip_rate_vals)*100.0, "o-")
        ax4.set_xscale("log")
        ax4.set_title("Reduced / Non-generic Cases")
        ax4.set_xlabel("L1"); ax4.set_ylabel("skip rate (%)")
        ax4.grid(True, alpha=0.3)

        _savefig(f"{this_dir}/summary.png")

        plt.figure(figsize=(6,4))
        plt.plot(L1_vals, acc_if_good_vals, "o-", label="ML inf | split correct")
        plt.plot(L1_vals, acc_if_bad_vals,  "s-", label="ML inf | split wrong")
        plt.xscale("log"); plt.ylim(0,1.01)
        plt.xlabel("L1"); plt.ylabel("Root accuracy")
        plt.title("ML (inferred) Accuracy Conditional on Split Recovery")
        plt.legend(); plt.grid(True, alpha=0.3)
        _savefig(f"{this_dir}/conditional_accuracy.png")

        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(L1_vals, splitP_vals, "o-", label="P(correct split)")
        ax1.set_xscale("log"); ax1.set_ylim(0,1.01)
        ax1.set_xlabel("L1"); ax1.set_ylabel("P(correct split)")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(L1_vals, np.array(skip_rate_vals)*100.0, "x--", label="skip rate (%)")
        ax2.set_ylabel("skip rate (%)")
        lns = ax1.get_lines() + ax2.get_lines()
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="best")
        plt.title("Split Recovery & Skip Rate vs L1")
        _savefig(f"{this_dir}/split_vs_skips.png")

    finally:
        try:
            fl.close()
        except Exception:
            pass
