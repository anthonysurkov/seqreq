# Deprecated functions from my_phylo.py

# --- Jukes–Cantor simulation (Deprecated) ----------------------------------
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

# --- NJ inference -----------------------------------------------------------
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
