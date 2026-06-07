"""
Pure-Python validation of the L2S algorithmic core (no numpy/torch).
Goal: prove correctness of instance gen, CPM EST/LST, critical path/blocks,
N5 neighborhood, FDD/MWKR dispatch, and that random local search reduces makespan.
This same logic is mirrored (with numpy) in the Colab notebook.
"""
import random
from collections import deque


# ----------------------------- Instance -----------------------------
def gen_instance(J, M, seed=0, low=1, high=99):
    rng = random.Random(seed)
    times = [[rng.randint(low, high) for _ in range(M)] for _ in range(J)]
    machines = []
    for _ in range(J):
        perm = list(range(M))
        rng.shuffle(perm)
        machines.append(perm)
    return {"J": J, "M": M, "times": times, "machines": machines}


def op_id(j, i, M):
    return j * M + i


# ----------------------------- Solution graph -----------------------------
def build_adj(inst, machine_seq):
    """Return predecessors[v] = list of u with arc u->v, plus proc times."""
    J, M = inst["J"], inst["M"]
    n_ops = J * M
    S, T = n_ops, n_ops + 1
    N = n_ops + 2
    p = [0.0] * N
    for j in range(J):
        for i in range(M):
            p[op_id(j, i, M)] = inst["times"][j][i]
    preds = [[] for _ in range(N)]
    succs = [[] for _ in range(N)]

    def add_arc(u, v):
        succs[u].append(v)
        preds[v].append(u)

    # conjunctive (job precedence) + source/sink
    for j in range(J):
        add_arc(S, op_id(j, 0, M))
        for i in range(M - 1):
            add_arc(op_id(j, i, M), op_id(j, i + 1, M))
        add_arc(op_id(j, M - 1, M), T)
    # disjunctive (machine order)
    for m in range(M):
        seq = machine_seq[m]
        for a in range(len(seq) - 1):
            add_arc(seq[a], seq[a + 1])
    return p, preds, succs, S, T, N


def topo_order(preds, succs, N):
    indeg = [len(preds[v]) for v in range(N)]
    q = deque([v for v in range(N) if indeg[v] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != N:
        raise ValueError("Graph has a cycle (infeasible solution)")
    return order


def cpm(p, preds, succs, S, T, N):
    order = topo_order(preds, succs, N)
    est = [0.0] * N
    for u in order:
        for v in succs[u]:
            if est[u] + p[u] > est[v]:
                est[v] = est[u] + p[u]
    makespan = est[T]
    lst = [makespan] * N
    for u in reversed(order):
        # lft[u] = min over successors v of lst[v]; lst[u] = lft[u]-p[u]
        if succs[u]:
            lft = min(lst[v] for v in succs[u])
        else:
            lft = makespan
        lst[u] = lft - p[u]
    return est, lst, makespan, order


def mp_est(p, preds, succs, S, T, N):
    """Message-passing EST operator from the paper (max-pooling). Cross-check vs CPM."""
    d = [0.0] * N
    c = [1.0] * N
    c[S] = 0.0
    for _ in range(N):  # at most H<=N iterations
        changed = False
        for v in range(N):
            if not preds[v]:
                continue
            nd = max(p[u] + (1.0 - c[u]) * d[u] for u in preds[v])
            nc = max(c[u] for u in preds[v])
            if nd != d[v] or nc != c[v]:
                d[v] = nd
                c[v] = nc
                changed = True
        if not changed:
            break
    return d


# ----------------------------- Critical path / blocks -----------------------------
def critical_path(p, est, lst, succs, S, T, rng):
    """Walk a single critical path S->T over tight & critical edges."""
    path = [S]
    u = S
    while u != T:
        cands = [v for v in succs[u]
                 if abs(est[v] - lst[v]) < 1e-9 and abs(est[u] + p[u] - est[v]) < 1e-9]
        if not cands:
            # fallback: any tight edge
            cands = [v for v in succs[u] if abs(est[u] + p[u] - est[v]) < 1e-9]
        v = rng.choice(cands)
        path.append(v)
        u = v
    return path  # includes S ... T


def critical_blocks(path, op_machine, S, T):
    """Group consecutive ops on the path by same machine (exclude S,T)."""
    core = [o for o in path if o != S and o != T]
    blocks = []
    cur = []
    cur_m = None
    for o in core:
        m = op_machine[o]
        if m == cur_m:
            cur.append(o)
        else:
            if cur:
                blocks.append(cur)
            cur = [o]
            cur_m = m
    if cur:
        blocks.append(cur)
    return blocks


def n5_moves(blocks):
    """Return list of (u,v) adjacent pairs to swap per N5 rule."""
    moves = []
    nb = len(blocks)
    for bi, block in enumerate(blocks):
        L = len(block)
        if L < 2:
            continue
        first_pair = (block[0], block[1])
        last_pair = (block[L - 2], block[L - 1])
        if nb == 1:
            # only one block: both ends valid (paper: at most 2N-2; single block edge handling)
            moves.append(first_pair)
            if L > 2:
                moves.append(last_pair)
        elif bi == 0:
            moves.append(last_pair)        # first CB: only last pair
        elif bi == nb - 1:
            moves.append(first_pair)       # last CB: only first pair
        else:
            moves.append(first_pair)
            if L > 2:
                moves.append(last_pair)
    # dedup
    seen = set()
    uniq = []
    for mv in moves:
        if mv not in seen:
            seen.add(mv)
            uniq.append(mv)
    return uniq


def apply_move(machine_seq, op_machine, u, v):
    """Swap adjacent ops u,v on their (shared) machine sequence."""
    m = op_machine[u]
    seq = machine_seq[m]
    iu, iv = seq.index(u), seq.index(v)
    seq[iu], seq[iv] = seq[iv], seq[iu]


# ----------------------------- FDD/MWKR initial solution -----------------------------
def fdd_mwkr_dispatch(inst):
    J, M = inst["J"], inst["M"]
    times, machines = inst["times"], inst["machines"]
    machine_seq = [[] for _ in range(M)]
    next_op = [0] * J            # next op index per job
    cum = [[0.0] * M for _ in range(J)]   # FDD: cumulative proc up to op i
    for j in range(J):
        s = 0.0
        for i in range(M):
            s += times[j][i]
            cum[j][i] = s
    total = [sum(times[j]) for j in range(J)]

    n_ops = J * M
    for _ in range(n_ops):
        best, best_pri = None, None
        for j in range(J):
            i = next_op[j]
            if i >= M:
                continue
            fdd = cum[j][i]
            mwkr = total[j] - (cum[j][i] - times[j][i])  # remaining work incl current
            pri = fdd / max(mwkr, 1e-9)
            if best_pri is None or pri < best_pri:
                best_pri = pri
                best = j
        j = best
        i = next_op[j]
        m = machines[j][i]
        machine_seq[m].append(op_id(j, i, M))
        next_op[j] += 1
    return machine_seq


def op_machine_map(inst):
    J, M = inst["J"], inst["M"]
    om = {}
    for j in range(J):
        for i in range(M):
            om[op_id(j, i, M)] = inst["machines"][j][i]
    return om


# ----------------------------- Validation runs -----------------------------
def evaluate(inst, machine_seq):
    p, preds, succs, S, T, N = build_adj(inst, machine_seq)
    est, lst, mk, order = cpm(p, preds, succs, S, T, N)
    return p, preds, succs, S, T, N, est, lst, mk


def random_local_search(inst, steps=300, seed=1):
    rng = random.Random(seed)
    om = op_machine_map(inst)
    machine_seq = fdd_mwkr_dispatch(inst)
    p, preds, succs, S, T, N, est, lst, mk = evaluate(inst, machine_seq)
    init_mk = mk
    best_mk = mk
    best_seq = [list(s) for s in machine_seq]
    for _ in range(steps):
        path = critical_path(p, est, lst, succs, S, T, rng)
        blocks = critical_blocks(path, om, S, T)
        moves = n5_moves(blocks)
        if not moves:
            break  # absorbing state
        u, v = rng.choice(moves)
        apply_move(machine_seq, om, u, v)
        p, preds, succs, S, T, N, est, lst, mk = evaluate(inst, machine_seq)
        if mk < best_mk:
            best_mk = mk
            best_seq = [list(s) for s in machine_seq]
    return init_mk, best_mk


def main():
    print("=== L2S core validation (pure Python) ===")
    inst = gen_instance(3, 3, seed=42)
    om = op_machine_map(inst)
    seq = fdd_mwkr_dispatch(inst)
    p, preds, succs, S, T, N, est, lst, mk = evaluate(inst, seq)
    d_mp = mp_est(p, preds, succs, S, T, N)
    ok = all(abs(est[v] - d_mp[v]) < 1e-9 for v in range(N))
    print(f"3x3 makespan(FDD/MWKR)={mk:.0f}; MP-EST == CPM-EST: {ok}")
    rng = random.Random(0)
    path = critical_path(p, est, lst, succs, S, T, rng)
    blocks = critical_blocks(path, om, S, T)
    print(f"critical path (ops)={[o for o in path if o not in (S,T)]}")
    print(f"critical blocks={blocks}")
    print(f"N5 moves={n5_moves(blocks)}")

    print("\n=== Random local search (N5) reduces makespan ===")
    for (J, M, sd) in [(3, 3, 42), (6, 6, 1), (10, 10, 7), (15, 10, 3)]:
        inst = gen_instance(J, M, seed=sd)
        init_mk, best_mk = random_local_search(inst, steps=500, seed=sd)
        impr = 100.0 * (init_mk - best_mk) / init_mk
        print(f"{J}x{M}: init={init_mk:.0f} -> best={best_mk:.0f}  improvement={impr:.1f}%")


if __name__ == "__main__":
    main()
