from __future__ import annotations
import os
import sys
import math
import tempfile
import shutil
import subprocess
import inspect
from copy import deepcopy
from typing import List, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng, Generator
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix

# --- Node definition --------------------------------------------------------
class Node:
    __slots__ = ("children", "bl", "seq", "name")
    def __init__(self, name: str | None = None):
        self.children: List[Node] = []
        self.bl: List[float] = []
        self.seq: np.ndarray | None = None
        self.name: str | None = name

    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> List[Node]:
        if self.is_leaf():
            return [self]
        out: List[Node] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

# --- Locate IQ‑TREE binary --------------------------------------------------
_IQTREE = shutil.which("iqtree2") or shutil.which("iqtree")
if _IQTREE is None:
    sys.stderr.write("ERROR: Cannot find IQ‑TREE on PATH.\n")
    sys.exit(1)

# --- ML tree inference via IQ‑TREE -----------------------------------------
def infer_tree_ml(phylip_path: str, workdir: str) -> Phylo.BaseTree.Tree:
    """
    Run IQ‑TREE (JC model) on the given PHYLIP file in a temporary workdir,
    returning a Biopython Tree object of the inferred topology.
    """
    prefix = os.path.join(workdir, "run")
    cmd = [
        _IQTREE,
        "-s", phylip_path,
        "-m", "JC",
        "-nt", "4",
        "-fast",
        "-quiet",
        "-pre", prefix
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"IQ‑TREE failed:\n{proc.stderr}")
    # IQ‑TREE writes the best‐tree to <prefix>.treefile
    return Phylo.read(prefix + ".treefile", "newick")

# --- Jukes–Cantor simulation ------------------------------------------------
EXP_COEF = -4.0/3.0

def simulate_jc(node: Node, rng: Generator) -> None:
    if node.is_leaf():
        return
    L = len(node.seq)
    for child, bl in zip(node.children, node.bl):
        p_same = 0.25 + 0.75 * math.exp(EXP_COEF * bl)
        u = rng.random(L)
        seq = node.seq.copy()
        muts = np.where(u >= p_same)[0]
        for pos in muts:
            choices = [b for b in range(4) if b != seq[pos]]
            seq[pos] = rng.choice(choices)
        child.seq = seq
        simulate_jc(child, rng)

# --- Distance & θ estimation -----------------------------------------------
def jc_dist(x: np.ndarray, y: np.ndarray) -> float:
    p = min(max(np.mean(x != y), 0.0), 0.75 - 1e-8)
    return -0.75 * math.log(1 - (4.0/3.0) * p)

def estimate_theta_hat(leaves: List[Node]) -> float:
    seqs = [leaf.seq for leaf in leaves]
    d_sum = 0.0
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            d_sum += jc_dist(seqs[i], seqs[j])
    pairs = len(seqs)*(len(seqs)-1)/2
    d_bar = d_sum / pairs
    depth = int(math.log2(len(seqs)))
    return d_bar / (2 * depth)

# --- Biopython -> Node conversion -------------------------------------------
def convert_clade(clade) -> Node:
    node = Node(clade.name if clade.is_terminal() else None)
    node.bl = []
    for child in clade.clades:
        cn = convert_clade(child)
        bl = child.branch_length or 0.0
        cn.bl = [bl]
        node.children.append(cn)
    return node

# --- NJ inference with JC correction ----------------------------------------
def infer_nj_tree(leaves: List[Node]) -> Node:
    seqs = [leaf.seq for leaf in leaves]
    names = [leaf.name for leaf in leaves]
    n = len(seqs)
    pdist_mat = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i+1, n):
            p = np.mean(seqs[i] != seqs[j])
            p = min(max(p, 0.0), 0.75 - 1e-8)
            pdist_mat[i, j] = pdist_mat[j, i] = -0.75 * math.log(1 - (4.0/3.0) * p)
    mat: List[List[float]] = []
    for i in range(n):
        row = [float(pdist_mat[i, j]) for j in range(i)]
        row.append(0.0)
        mat.append(row)
    dm = DistanceMatrix(names, mat)
    nj = DistanceTreeConstructor().nj(dm)
    return convert_clade(nj.root)

# --- Splits & comparison ---------------------------------------------------
def splits(tree: Node) -> Set[frozenset[str]]:
    leaves = {leaf.name for leaf in tree.leaves()}
    raw: Set[frozenset[str]] = set()
    def collect(n: Node, p: Node|None) -> Set[str]:
        if n.is_leaf(): return {n.name}
        s: Set[str] = set()
        for c in n.children:
            s |= collect(c, n)
        if p is not None and 1 < len(s) < len(leaves)-1:
            raw.add(frozenset(s))
        return s
    collect(tree, None)
    sym: Set[frozenset[str]] = set()
    for s in raw:
        sym.update([frozenset(s), frozenset(leaves - set(s))])
    return sym

def unrooted_equal(t1: Node, t2: Node) -> bool:
    return splits(t1) == splits(t2)

# --- Root reconstruction via ML JR (pruning) --------------------------------
def prune_lik(node: Node, theta: float) -> np.ndarray:
    if node.is_leaf():
        L = len(node.seq)
        M = np.zeros((4, L))
        for b in range(4): M[b] = (node.seq == b).astype(float)
        return M
    mats = [prune_lik(c, theta) for c in node.children]
    ps = 0.25 + 0.75 * math.exp(EXP_COEF * theta)
    pd = 0.25 * (1 - math.exp(EXP_COEF * theta))
    Lm = np.ones_like(mats[0])
    for M in mats:
        Lm *= (ps*M + pd*(M.sum(axis=0)-M))
    return Lm

def reconstruct_root_sequence(tree: Node, theta: float) -> np.ndarray:
    return np.argmax(prune_lik(tree, theta), axis=0)

# --- Root reconstruction via Majority Rule (MR) -----------------------------
def reconstruct_root_majority(tree: Node, rng: Generator) -> np.ndarray:
    leaves = tree.leaves()
    seqs = np.array([leaf.seq for leaf in leaves])  # shape (n_leaves, L)
    L = seqs.shape[1]
    root_pred = np.empty(L, dtype=int)
    for i in range(L):
        column = seqs[:, i]
        values, counts = np.unique(column, return_counts=True)
        max_count = counts.max()
        modes = values[counts == max_count]
        # break ties randomly
        root_pred[i] = rng.choice(modes)
    return root_pred

def write_phylip(leaves: List[Node], path: str) -> None:
    int2char = {0:'A',1:'C',2:'G',3:'T'}
    n, L = len(leaves), len(leaves[0].seq)
    with open(path, 'w') as fh:
        fh.write(f"{n} {L}\n")
        for lf in leaves:
            name = lf.name[:10].ljust(10)
            seq  = ''.join(int2char[int(b)] for b in lf.seq)
            fh.write(f"{name}{seq}\n")

# --- Single replicate -------------------------------------------------------
def run_rep(build_tree,
            L1: int,
            L2: int,
            n_leaves: int,
            theta: float,
            rng
           ) -> Tuple[bool, float, float, float, float]:
    """
    1) Simulate T1 with L1 → estimate θ̂.
    2) Attach ghost “OG” at T1’s root and infer its ML topology (rooted on OG).
    3) Deep-copy that augmented tree → simulate L2 fresh sites → strip OG → T2.
    4) Reconstruct root on T2 via:
         - ML_true: true shape + θ̂
         - ML_inf:  inferred shape + θ̂
         - MR:      majority rule
    Returns (ok, |θ̂−θ|, acc_ml_true, acc_ml_inf, acc_mr).
    """

    # (A) simulate T1 → θ̂
    sig = inspect.signature(build_tree)
    if 'rng' in sig.parameters:
        T1 = build_tree(n_leaves, theta, rng=rng)
    else:
        T1 = build_tree(n_leaves, theta)
    T1.seq = rng.integers(0,4,size=L1)
    simulate_jc(T1, rng)
    theta_hat = estimate_theta_hat(T1.leaves())

    # (B) attach ghost outgroup at T1’s root
    ghost = Node("OG")
    ghost.bl  = [0.0]
    ghost.seq = T1.seq.copy()
    aug_root = Node()
    aug_root.children = [T1, ghost]
    aug_root.bl       = [theta, 0.0]

    # (C) infer augmented tree via IQ‑TREE, then reroot on OG
    with tempfile.TemporaryDirectory() as tmpdir:
        aln = os.path.join(tmpdir, "T1_aug.phy")
        write_phylip(aug_root.leaves(), aln)
        ml_tree = infer_tree_ml(aln, tmpdir)    # returns a Bio.Phylo Tree
    ml_tree.root_with_outgroup("OG")
    aug_T1hat = convert_clade(ml_tree.root)
    # strip OG
    left, right = aug_T1hat.children
    T1hat = right if any(l.name=="OG" for l in left.leaves()) else left

    # (D) simulate T2 on the same augmented shape with L2, then strip OG
    T2aug = deepcopy(aug_root)
    T2aug.seq = rng.integers(0,4,size=L2)
    simulate_jc(T2aug, rng)
    l2, r2 = T2aug.children
    T2 = r2 if any(l.name=="OG" for l in l2.leaves()) else l2

    # (E) ML_true on T2
    root_ml_true = reconstruct_root_sequence(T2, theta_hat)
    acc_ml_true  = float((root_ml_true == T2.seq).mean())

    # (F) ML_inf on T1hat
    seqmap = {lf.name: lf.seq for lf in T2.leaves()}
    for lf in T1hat.leaves():
        lf.seq = seqmap[lf.name]
    root_ml_inf = reconstruct_root_sequence(T1hat, theta_hat)
    acc_ml_inf  = float((root_ml_inf == T2.seq).mean())

    # (G) MR baseline
    root_mr = reconstruct_root_majority(T2, rng)
    acc_mr   = float((root_mr == T2.seq).mean())

    # (H) did we recover T2’s splits?
    ok = unrooted_equal(T2, T1hat)

    return ok, abs(theta_hat - theta), acc_ml_true, acc_ml_inf, acc_mr

# --- Main logic -----------------------------------------------------------
def run_my_phylo(build_tree, reps=1000, n_leaves=8, theta=0.25,
                 L1_vals=np.logspace(0.5, 3, 30, base=10, dtype=int).tolist(),
                 L2=100, rng=None):
    ok_vals = []
    th_err_vals = []
    ml_true_vals = []
    ml_inf_vals = []
    mr_vals = []
    acc_if_good_vals = []
    acc_if_bad_vals = []

    print(f"  L1 | θ_err  | ML_true | ML_inf  | MR")
    for L1 in L1_vals:
        ok_list = []
        e_list = []
        t_true = []
        t_inf = []
        t_mr = []
        inf_acc_list = []

        for _ in range(reps):
            ok, e, a_true, a_inf, a_mr = run_rep(build_tree, L1, L2, n_leaves, theta, rng)
            ok_list.append(ok)
            e_list.append(e)
            t_true.append(a_true)
            t_inf.append(a_inf)
            t_mr.append(a_mr)
            inf_acc_list.append(a_inf)

        ok_vals.append(np.mean(ok_list))
        th_err_vals.append(np.mean(e_list))
        ml_true_vals.append(np.mean(t_true))
        ml_inf_vals.append(np.mean(t_inf))
        mr_vals.append(np.mean(t_mr))

        ok_arr = np.array(ok_list)
        inf_arr = np.array(inf_acc_list)
        acc_if_good_vals.append(inf_arr[ok_arr].mean() if ok_arr.any() else np.nan)
        acc_if_bad_vals.append(inf_arr[~ok_arr].mean() if (~ok_arr).any() else np.nan)

        print(f"{L1:4d} | {th_err_vals[-1]:6.3f} | {ml_true_vals[-1]:7.3f}"
              f" | {ml_inf_vals[-1]:7.3f} | {mr_vals[-1]:5.3f}")

    # Probability of correct tree reconstruction
    plt.figure(figsize=(6,4))
    plt.plot(L1_vals, ok_vals, 'o-')
    plt.xscale('log')
    plt.xlabel('Sequence length used for θ estimation L1')
    plt.ylabel('P(tree recovered)')
    plt.title('Probability of Correct Tree Reconstruction Over L1 (ML inferred)')
    plt.grid(True)

    # Overall inferred‑ML root accuracy
    plt.figure(figsize=(6,4))
    plt.plot(L1_vals, ml_inf_vals, 'o-')
    plt.xscale('log')
    plt.xlabel('Sequence length used for θ estimation L1')
    plt.ylabel('Overall root accuracy (ML inferred)')
    plt.title('Overall Root Reconstruction Accuracy Over L1')
    plt.grid(True)

    # 2×2 summary
    fig, axes = plt.subplots(2,2,figsize=(10,8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(L1_vals, ml_true_vals, 'o-', label='ML (true topo)')
    ax1.plot(L1_vals, ml_inf_vals,  's-', label='ML (inferred)')
    ax1.plot(L1_vals, mr_vals,      'x--', label='Majority')
    ax1.set_xscale('log'); ax1.set_title('Root Accuracy vs L1')
    ax1.set_xlabel('L1'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True)

    ax2.plot(L1_vals, th_err_vals, 'o-')
    ax2.set_xscale('log'); ax2.set_title('θ‑estimation Error')
    ax2.set_xlabel('L1'); ax2.set_ylabel('|θ̂−θ|'); ax2.grid(True)

    delta = np.array(ml_true_vals) - np.array(ml_inf_vals)
    ax3.plot(L1_vals, delta, 'o-')
    ax3.set_xscale('log'); ax3.set_title('Loss from Topology Error')
    ax3.set_xlabel('L1'); ax3.set_ylabel('ML (true) −ML (inf)'); ax3.grid(True)

    err_inf  = 1 - np.array(ml_inf_vals)
    err_true = 1 - np.array(ml_true_vals)
    err_mr   = 1 - np.array(mr_vals)
    ax4.plot(L1_vals, err_inf,  's-',  label='Err ML (inferred tree)')
    ax4.plot(L1_vals, err_true, 'o-',  label='Err ML (true tree)')
    ax4.plot(L1_vals, err_mr,   'x--', label='Err MR')
    ax4.set_xscale('log'); ax4.set_title('Error Rates vs L1')
    ax4.set_xlabel('L1'); ax4.set_ylabel('Error rate')
    ax4.legend(); ax4.grid(True)

    # Conditional‑accuracy (only inferred ML)
    plt.figure(figsize=(6,4))
    plt.plot(L1_vals, acc_if_good_vals, 'o-', label='Correct topography')
    plt.plot(L1_vals, acc_if_bad_vals,  's-', label='Incorrect topography')
    plt.xscale('log')
    plt.xlabel('Sequence length used for θ estimation L1')
    plt.ylabel('Root reconstruction accuracy')
    plt.title('ML Accuracy Conditional on Tree Recovery')
    plt.legend(); plt.grid(True)

    plt.show()

