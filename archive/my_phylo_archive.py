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

