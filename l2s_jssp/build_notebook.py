"""
Generator for L2S_JSSP_Colab.ipynb.

Defines every notebook cell as a string and assembles a valid nbformat-4
notebook. Run:  python3 build_notebook.py
"""
import json
import os

CELLS = []  # list of ("markdown"|"code", source_str)


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src))


# =====================================================================
# CELL 1 - Title (markdown)
# =====================================================================
md(r"""# L2S for Job-Shop Scheduling — Colab Notebook

A from-scratch, **single-file PyTorch** re-implementation of the *Learning-to-Search* (L2S)
improvement-heuristic for the Job-Shop Scheduling Problem (JSSP), inspired by:

> Cong Zhang et al., *Deep Reinforcement Learning Guided Improvement Heuristic for
> Job Shop Scheduling*, ICLR 2024.

**What L2S does.** Instead of *constructing* a schedule operation-by-operation, L2S starts
from a **complete** solution and *learns to search*: at every step a GNN policy picks a local
move (an operation-pair swap inside a *critical block*), the makespan is recomputed, and the
process repeats for `T` steps. The agent is rewarded for improving the best solution found.

**The 5 pillars (and where they live in this notebook):**
1. **Local-search loop** → `Environment` (Cells 8-9)
2. **N5 neighbourhood** (critical blocks) → Cell 6
3. **Graph embedding** (GIN-based TPM + GAT-based CAM) → Cell 10
4. **MDP** (state = complete-solution graph, reward = step improvement) → Cell 8
5. **Message-passing schedule evaluator** (EST/LST) → Cell 5

> This notebook uses only `torch`, `numpy`, `matplotlib` (all pre-installed on Colab).
> No `torch_geometric` needed — the GNN uses dense adjacency matrices, which is perfect
> for the small/medium instances we train on. Set Runtime → GPU for a speed-up (optional).
""")

# =====================================================================
# CELL 2 - environment check
# =====================================================================
code(r"""# Cell 2 — Environment check (Colab has torch/numpy/matplotlib pre-installed)
import sys, torch, numpy as np
print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("NumPy  :", np.__version__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
""")

# =====================================================================
# CELL 3 - imports + config
# =====================================================================
code(r"""# Cell 3 — Imports, seeding, and global configuration
import math, random, time
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # ---- problem ----
    n_jobs: int = 6           # |J|
    n_machines: int = 6       # |M|
    p_low: int = 1
    p_high: int = 99
    # ---- feature normalisation (paper: p/99, est/1000, lst/1000) ----
    p_norm: float = 99.0
    t_norm: float = 1000.0
    # ---- model (paper uses embed=128 hidden=128, K=4; smaller defaults train fast) ----
    embed_dim: int = 64
    gin_hidden: int = 64
    gat_hidden: int = 64
    n_layers: int = 3         # K (paper: 4)
    act_hidden: int = 64
    act_dim: int = 32         # q in the score matrix
    leaky_slope: float = 0.15
    # ---- RL / training ----
    horizon_T: int = 64       # improvement steps per episode (paper: 500; raise after it works)
    n_step: int = 8           # n in n-step REINFORCE (paper: 10)
    gamma: float = 1.0
    batch_size: int = 8       # instances generated on the fly per iteration (paper: 64)
    n_iterations: int = 120   # outer iterations
    lr: float = 5e-4          # paper: 5e-5 (larger here because the net is smaller)
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0

cfg = Config()
set_seed(cfg.seed)
print(cfg)
""")

# =====================================================================
# CELL 4 - instance generation & parsing
# =====================================================================
code(r"""# Cell 4 — JSSP instance: random Taillard-style generator + standard-format parser
#
# An instance is stored as:
#   times[j][i]    : processing time of the i-th operation of job j
#   machines[j][i] : machine that processes that operation (each job visits every machine once)
# Global operation id:  op = j * M + i ;  dummy source S = J*M ; dummy sink T = J*M + 1.

class JSSPInstance:
    def __init__(self, times, machines):
        self.times = np.asarray(times, dtype=np.float64)
        self.machines = np.asarray(machines, dtype=np.int64)
        self.J, self.M = self.times.shape
        self.n_ops = self.J * self.M
        self.S = self.n_ops
        self.T = self.n_ops + 1
        self.N = self.n_ops + 2
        # per-operation flat arrays
        self.p = np.zeros(self.N, dtype=np.float64)
        self.op_machine = np.full(self.N, -1, dtype=np.int64)
        for j in range(self.J):
            for i in range(self.M):
                o = self.op_id(j, i)
                self.p[o] = self.times[j, i]
                self.op_machine[o] = self.machines[j, i]

    def op_id(self, j, i):
        return j * self.M + i


def generate_instance(J, M, seed=0, low=1, high=99):
    rng = np.random.RandomState(seed)
    times = rng.randint(low, high + 1, size=(J, M))
    machines = np.stack([rng.permutation(M) for _ in range(J)])
    return JSSPInstance(times, machines)


def parse_standard_jssp(text):
    '''Parse the classic JSSP text format:
        first line:  <num_jobs> <num_machines>
        next J lines: machine time machine time ...  (one job per line)
    '''
    toks = [t for t in text.strip().split("\n") if t.strip() and not t.strip().startswith("#")]
    J, M = map(int, toks[0].split()[:2])
    times = np.zeros((J, M), dtype=np.int64)
    machines = np.zeros((J, M), dtype=np.int64)
    for j in range(J):
        nums = list(map(int, toks[1 + j].split()))
        for i in range(M):
            machines[j, i] = nums[2 * i]
            times[j, i] = nums[2 * i + 1]
    return JSSPInstance(times, machines)


# quick demo
_inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=42)
print(f"Generated {_inst.J}x{_inst.M} instance | n_ops={_inst.n_ops} | N(with S,T)={_inst.N}")
print("machine order of job 0:", _inst.machines[0])
print("proc times of job 0  :", _inst.times[0])
""")

# =====================================================================
# CELL 5 - graph + CPM + message passing
# =====================================================================
code(r"""# Cell 5 — Disjunctive graph, CPM schedule (EST/LST/makespan) + message-passing evaluator
#
# A *solution* is the processing order of operations on each machine:
#   machine_seq[m] = ordered list of op ids on machine m.
# From it we build the DAG (conjunctive job arcs + disjunctive machine arcs) and compute the schedule.

def build_graph(inst, machine_seq):
    '''Return predecessor/successor adjacency lists for the solution DAG.'''
    N, S, T, M = inst.N, inst.S, inst.T, inst.M
    preds = [[] for _ in range(N)]
    succs = [[] for _ in range(N)]
    def arc(u, v):
        succs[u].append(v); preds[v].append(u)
    # conjunctive arcs (job precedence) + source/sink
    for j in range(inst.J):
        arc(S, inst.op_id(j, 0))
        for i in range(inst.M - 1):
            arc(inst.op_id(j, i), inst.op_id(j, i + 1))
        arc(inst.op_id(j, inst.M - 1), T)
    # disjunctive arcs (machine order)
    for m in range(M):
        seq = machine_seq[m]
        for a in range(len(seq) - 1):
            arc(seq[a], seq[a + 1])
    return preds, succs


def topo_order(preds, succs, N):
    indeg = [len(preds[v]) for v in range(N)]
    dq = deque([v for v in range(N) if indeg[v] == 0])
    order = []
    while dq:
        u = dq.popleft(); order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    if len(order) != N:
        raise ValueError("Infeasible solution: the disjunctive graph contains a cycle.")
    return order


def cpm_schedule(inst, preds, succs):
    '''Critical Path Method: earliest/latest start times and makespan.'''
    N, S, T, p = inst.N, inst.S, inst.T, inst.p
    order = topo_order(preds, succs, N)
    est = np.zeros(N)
    for u in order:
        eu = est[u] + p[u]
        for v in succs[u]:
            if eu > est[v]:
                est[v] = eu
    makespan = est[T]
    lst = np.full(N, makespan)
    for u in reversed(order):
        if succs[u]:
            lft = min(lst[v] for v in succs[u])
        else:
            lft = makespan
        lst[u] = lft - p[u]
    return est, lst, makespan, order


def message_passing_est(inst, preds, succs):
    '''Paper pillar 5: EST via a max-pooling message-passing operator (GPU-batchable in spirit).
    Provided to mirror the paper; CPM above is the fast reference used by the env.'''
    N, S, p = inst.N, inst.S, inst.p
    d = np.zeros(N); c = np.ones(N); c[S] = 0.0
    for _ in range(N):
        changed = False
        for v in range(N):
            if not preds[v]:
                continue
            nd = max(p[u] + (1.0 - c[u]) * d[u] for u in preds[v])
            nc = max(c[u] for u in preds[v])
            if nd != d[v] or nc != c[v]:
                d[v], c[v] = nd, nc; changed = True
        if not changed:
            break
    return d


# sanity: message-passing EST must equal CPM EST
_seq = [[] for _ in range(_inst.M)]
for j in range(_inst.J):
    for i in range(_inst.M):
        _seq[_inst.machines[j, i]].append(_inst.op_id(j, i))
_pr, _su = build_graph(_inst, _seq)
_est, _lst, _mk, _ = cpm_schedule(_inst, _pr, _su)
_dmp = message_passing_est(_inst, _pr, _su)
print("MP-EST == CPM-EST :", np.allclose(_est, _dmp), "| makespan =", _mk)
""")

# =====================================================================
# CELL 6 - critical path / blocks / N5
# =====================================================================
code(r"""# Cell 6 — Critical path, critical blocks, and the N5 neighbourhood
EPS = 1e-7

def find_critical_path(inst, preds, succs, est, lst, rng):
    '''Walk one S->T path over edges that are tight AND critical (est==lst).'''
    S, T, p = inst.S, inst.T, inst.p
    path = [S]; u = S
    while u != T:
        cands = [v for v in succs[u]
                 if abs(est[v] - lst[v]) < EPS and abs(est[u] + p[u] - est[v]) < EPS]
        if not cands:
            cands = [v for v in succs[u] if abs(est[u] + p[u] - est[v]) < EPS]
        u = cands[rng.randrange(len(cands))]
        path.append(u)
    return path


def critical_blocks(inst, path):
    '''Consecutive operations on the path processed by the same machine.'''
    core = [o for o in path if o != inst.S and o != inst.T]
    blocks, cur, cur_m = [], [], None
    for o in core:
        m = inst.op_machine[o]
        if m == cur_m:
            cur.append(o)
        else:
            if cur: blocks.append(cur)
            cur, cur_m = [o], m
    if cur: blocks.append(cur)
    return blocks


def n5_candidate_moves(blocks):
    '''N5: swap first/last adjacent pair in each critical block.
    First block -> only last pair; last block -> only first pair; |N5| <= 2N(s)-2.'''
    moves, nb = [], len(blocks)
    for bi, b in enumerate(blocks):
        L = len(b)
        if L < 2:
            continue
        first_pair, last_pair = (b[0], b[1]), (b[L - 2], b[L - 1])
        if nb == 1:
            moves.append(first_pair)
            if L > 2: moves.append(last_pair)
        elif bi == 0:
            moves.append(last_pair)
        elif bi == nb - 1:
            moves.append(first_pair)
        else:
            moves.append(first_pair)
            if L > 2: moves.append(last_pair)
    seen, uniq = set(), []
    for mv in moves:
        if mv not in seen:
            seen.add(mv); uniq.append(mv)
    return uniq
""")

# =====================================================================
# CELL 7 - FDD/MWKR dispatch
# =====================================================================
code(r"""# Cell 7 — FDD/MWKR dispatching rule -> initial complete solution
def fdd_mwkr_initial(inst):
    '''Build an initial machine ordering using the minimum ratio of
    Flow-Due-Date (cumulative processing time up to the op) to
    Most-Work-Remaining (remaining work of the job).'''
    J, M, times = inst.J, inst.M, inst.times
    machine_seq = [[] for _ in range(M)]
    next_op = [0] * J
    cum = np.cumsum(times, axis=1)            # FDD per op
    total = times.sum(axis=1)                 # job total work
    for _ in range(inst.n_ops):
        best_j, best_pri = -1, None
        for j in range(J):
            i = next_op[j]
            if i >= M:
                continue
            fdd = cum[j, i]
            mwkr = total[j] - (cum[j, i] - times[j, i])   # remaining incl. current
            pri = fdd / max(mwkr, 1e-9)
            if best_pri is None or pri < best_pri:
                best_pri, best_j = pri, j
        j = best_j; i = next_op[j]
        machine_seq[inst.machines[j, i]].append(inst.op_id(j, i))
        next_op[j] += 1
    return machine_seq
""")

# =====================================================================
# CELL 8 - environment (MDP)
# =====================================================================
code(r"""# Cell 8 — The L2S local-search environment (the MDP)
#
# State   : current complete solution (its disjunctive graph) + node features (p, est, lst).
# Action  : an N5 operation-pair (u, v) to swap.
# Reward  : max(incumbent_makespan - new_makespan, 0)  (improvement over best-so-far).
# Done    : horizon reached, or absorbing state (no feasible N5 move).

class Environment:
    def __init__(self, inst, cfg, rng):
        self.inst = inst
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        self.machine_seq = fdd_mwkr_initial(self.inst)
        self._recompute()
        self.init_makespan = self.makespan
        self.best_makespan = self.makespan
        self.steps = 0
        self.done = False
        st = self._state()
        if len(st["cands"]) == 0:      # already optimal / no critical-block move
            self.done = True
        return st

    def _recompute(self):
        self.preds, self.succs = build_graph(self.inst, self.machine_seq)
        self.est, self.lst, self.makespan, _ = cpm_schedule(self.inst, self.preds, self.succs)

    def _candidates(self):
        path = find_critical_path(self.inst, self.preds, self.succs, self.est, self.lst, self.rng)
        blocks = critical_blocks(self.inst, path)
        return n5_candidate_moves(blocks)

    def _state(self):
        inst, cfg = self.inst, self.cfg
        N = inst.N
        feat = np.zeros((N, 3), dtype=np.float32)
        feat[:, 0] = inst.p / cfg.p_norm
        feat[:, 1] = self.est / cfg.t_norm
        feat[:, 2] = self.lst / cfg.t_norm
        # dense predecessor adjacencies: A[i, j] = 1 if j -> i (j is a predecessor of i)
        A_all = np.zeros((N, N), dtype=np.float32)
        A_J = np.zeros((N, N), dtype=np.float32)   # conjunctive only (job precedence)
        A_M = np.zeros((N, N), dtype=np.float32)   # disjunctive only (machine order)
        for v in range(N):
            for u in self.preds[v]:
                A_all[v, u] = 1.0
                same_job = (u // inst.M == v // inst.M) and u < inst.n_ops and v < inst.n_ops
                is_src_sink = (u == inst.S or v == inst.T)
                if same_job or is_src_sink:
                    A_J[v, u] = 1.0
                else:
                    A_M[v, u] = 1.0
        cands = self._candidates()
        return {"feat": feat, "A_all": A_all, "A_J": A_J, "A_M": A_M,
                "cands": cands, "makespan": self.makespan,
                "best": self.best_makespan, "init": self.init_makespan}

    def step(self, move):
        '''Apply swap; return (state, reward, done).'''
        u, v = move
        m = self.inst.op_machine[u]
        seq = self.machine_seq[m]
        iu, iv = seq.index(u), seq.index(v)
        seq[iu], seq[iv] = seq[iv], seq[iu]
        self._recompute()
        reward = max(self.best_makespan - self.makespan, 0.0)
        if self.makespan < self.best_makespan:
            self.best_makespan = self.makespan
        self.steps += 1
        st = self._state()
        if self.steps >= self.cfg.horizon_T or len(st["cands"]) == 0:
            self.done = True
        return st, reward, self.done
""")

# =====================================================================
# CELL 9 - env sanity check (random policy)
# =====================================================================
code(r"""# Cell 9 — Sanity check: a RANDOM policy should already reduce the makespan via N5
def random_rollout(inst, cfg, steps=200, seed=0):
    rng = random.Random(seed)
    env = Environment(inst, cfg, rng)
    st = env.reset()
    curve = [env.makespan]
    for _ in range(steps):
        if env.done or len(st["cands"]) == 0:
            break
        mv = st["cands"][rng.randrange(len(st["cands"]))]
        st, r, done = env.step(mv)
        curve.append(env.best_makespan)
    return env.init_makespan, env.best_makespan, curve

_inst = generate_instance(10, 10, seed=7)
init_mk, best_mk, curve = random_rollout(_inst, Config(horizon_T=300), steps=300, seed=7)
print(f"10x10 random N5 search: init={init_mk:.0f} -> best={best_mk:.0f} "
      f"({100*(init_mk-best_mk)/init_mk:.1f}% better)")
plt.figure(figsize=(6,3)); plt.plot(curve); plt.xlabel("step"); plt.ylabel("best makespan")
plt.title("Random N5 local search"); plt.grid(alpha=.3); plt.show()
""")

# =====================================================================
# CELL 10 - model
# =====================================================================
code(r"""# Cell 10 — The L2S policy network: TPM (GIN) + CAM (GAT) + action head
#
# Works on a SINGLE graph at a time using dense adjacency (great for small/medium JSSP).

def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GINLayer(nn.Module):
    '''Topological Embedding Module block (Graph Isomorphism Network).
       mu' = MLP( (1+eps) * mu + A_pred @ mu ).'''
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.net = mlp([in_dim, hid, hid, out_dim])   # 2 hidden layers (paper)

    def forward(self, h, A):
        agg = A @ h                      # sum of predecessors
        return self.net((1.0 + self.eps) * h + agg)


class GATLayer(nn.Module):
    '''Single-head graph-attention block over a masked dense adjacency (with self-loops).'''
    def __init__(self, in_dim, out_dim, slope=0.15):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(slope)

    def forward(self, h, A):
        N = h.size(0)
        Wh = self.W(h)                                   # (N, out)
        mask = A + torch.eye(N, device=h.device)         # add self-loops
        e = self.leaky(self.a_src(Wh) + self.a_dst(Wh).T)  # (N, N): e[i,j]
        e = e.masked_fill(mask <= 0, float("-inf"))
        alpha = torch.softmax(e, dim=1)                  # attend over neighbours j of i
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return F.elu(alpha @ Wh)


class L2SPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        # --- TPM: stack of GIN layers ---
        self.gin = nn.ModuleList()
        for k in range(cfg.n_layers):
            self.gin.append(GINLayer(3 if k == 0 else d, cfg.gin_hidden, d))
        # --- CAM: two GAT stacks (G_J and G_M) merged each layer ---
        self.gat_J = nn.ModuleList()
        self.gat_M = nn.ModuleList()
        for k in range(cfg.n_layers):
            self.gat_J.append(GATLayer(3 if k == 0 else d, d, cfg.leaky_slope))
            self.gat_M.append(GATLayer(3 if k == 0 else d, d, cfg.leaky_slope))
        # --- action head: node+graph embedding -> q-dim score vector ---
        self.act_mlp = mlp([4 * d, cfg.act_hidden, cfg.act_hidden,
                            cfg.act_hidden, cfg.act_dim])

    def embed(self, feat, A_all, A_J, A_M):
        # TPM (sum embeddings of all layers)
        h = feat; mu_sum = 0.0
        for layer in self.gin:
            h = layer(h, A_all); mu_sum = mu_sum + h
        mu = mu_sum
        # CAM
        h = feat
        for lj, lm in zip(self.gat_J, self.gat_M):
            h = 0.5 * (lj(h, A_J) + lm(h, A_M))
        nu = h
        hV = torch.cat([mu, nu], dim=1)                  # (N, 2d)
        hG = torch.cat([mu.mean(0, keepdim=True),
                        nu.mean(0, keepdim=True)], dim=1) # (1, 2d)
        return hV, hG

    def action_logits(self, hV, hG, cands):
        N = hV.size(0)
        hcat = torch.cat([hV, hG.expand(N, -1)], dim=1)  # (N, 4d)
        hp = self.act_mlp(hcat)                          # (N, q)
        SC = hp @ hp.T                                   # (N, N) pair scores
        idx = torch.tensor(cands, dtype=torch.long, device=hV.device)  # (C, 2)
        logits = SC[idx[:, 0], idx[:, 1]]                # (C,)
        return logits

    def forward(self, state, device):
        feat = torch.from_numpy(state["feat"]).to(device)
        A_all = torch.from_numpy(state["A_all"]).to(device)
        A_J = torch.from_numpy(state["A_J"]).to(device)
        A_M = torch.from_numpy(state["A_M"]).to(device)
        hV, hG = self.embed(feat, A_all, A_J, A_M)
        return self.action_logits(hV, hG, state["cands"])


print("Model defined. Param count (with cfg):",
      sum(p.numel() for p in L2SPolicy(cfg).parameters()))
""")

# =====================================================================
# CELL 11 - action sampling helpers
# =====================================================================
code(r"""# Cell 11 — Action selection: sample a move + log-prob + entropy
def select_action(policy, state, device, greedy=False):
    logits = policy(state, device)                       # (C,)
    dist = torch.distributions.Categorical(logits=logits)
    if greedy:
        a = torch.argmax(logits)
    else:
        a = dist.sample()
    logp = dist.log_prob(a)
    entropy = dist.entropy()
    move = state["cands"][a.item()]
    return move, logp, entropy
""")

# =====================================================================
# CELL 12 - trainer
# =====================================================================
code(r"""# Cell 12 — n-step REINFORCE trainer (with baseline + entropy regularisation)
#
# Faithful to Algorithm 1 of L2S: update the policy every n steps along the trajectory.
# We additionally subtract a per-window mean-return baseline and add an entropy bonus
# (the latter in the spirit of the TBGAT follow-up) to stabilise the small-net prototype.

def compute_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def train(cfg, log_every=10):
    set_seed(cfg.seed)
    policy = L2SPolicy(cfg).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history = {"iter": [], "init_mk": [], "final_mk": [], "impr%": [], "loss": []}

    for it in range(cfg.n_iterations):
        # fresh batch of instances generated on the fly
        insts = [generate_instance(cfg.n_jobs, cfg.n_machines,
                                   seed=cfg.seed + it * 10_000 + b)
                 for b in range(cfg.batch_size)]
        envs = [Environment(inst, cfg, random.Random(cfg.seed + it * 7 + b))
                for b, inst in enumerate(insts)]
        states = [e.reset() for e in envs]
        init_mks = [e.init_makespan for e in envs]

        win_logps = [[] for _ in range(cfg.batch_size)]
        win_rews = [[] for _ in range(cfg.batch_size)]
        win_ents = [[] for _ in range(cfg.batch_size)]
        iter_loss = 0.0; n_updates = 0

        for t in range(cfg.horizon_T):
            for b in range(cfg.batch_size):
                if envs[b].done or len(states[b]["cands"]) == 0:
                    continue
                move, logp, ent = select_action(policy, states[b], DEVICE)
                ns, r, done = envs[b].step(move)
                states[b] = ns
                # reward scaled by initial makespan (keeps gradients well-conditioned)
                win_logps[b].append(logp)
                win_ents[b].append(ent)
                win_rews[b].append(r / max(init_mks[b], 1.0))

            if (t + 1) % cfg.n_step == 0 or t == cfg.horizon_T - 1:
                loss = 0.0; count = 0
                for b in range(cfg.batch_size):
                    if not win_logps[b]:
                        continue
                    returns = compute_returns(win_rews[b], cfg.gamma)
                    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
                    baseline = returns.mean()
                    adv = returns - baseline
                    logps = torch.stack(win_logps[b])
                    ents = torch.stack(win_ents[b])
                    loss = loss + (-(logps * adv).sum() - cfg.entropy_coef * ents.sum())
                    count += 1
                if count > 0:
                    loss = loss / count
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                    opt.step()
                    iter_loss += float(loss.item()); n_updates += 1
                win_logps = [[] for _ in range(cfg.batch_size)]
                win_rews = [[] for _ in range(cfg.batch_size)]
                win_ents = [[] for _ in range(cfg.batch_size)]

        final_mks = [e.best_makespan for e in envs]
        impr = float(np.mean([(i - f) / i * 100 for i, f in zip(init_mks, final_mks)]))
        history["iter"].append(it)
        history["init_mk"].append(float(np.mean(init_mks)))
        history["final_mk"].append(float(np.mean(final_mks)))
        history["impr%"].append(impr)
        history["loss"].append(iter_loss / max(n_updates, 1))
        if it % log_every == 0 or it == cfg.n_iterations - 1:
            print(f"iter {it:4d} | init {np.mean(init_mks):7.1f} | "
                  f"best {np.mean(final_mks):7.1f} | improvement {impr:5.2f}% | "
                  f"loss {history['loss'][-1]:+.4f}")
    return policy, history
""")

# =====================================================================
# CELL 13 - run training + plot
# =====================================================================
code(r"""# Cell 13 — Train (default 6x6; takes a few minutes on CPU, faster on GPU)
cfg = Config(n_jobs=6, n_machines=6, horizon_T=64, n_step=8,
             batch_size=8, n_iterations=120, lr=5e-4, embed_dim=64, n_layers=3)
set_seed(cfg.seed)
t0 = time.time()
policy, history = train(cfg, log_every=10)
print(f"Training done in {time.time()-t0:.1f}s")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(history["iter"], history["impr%"]); ax[0].set_title("Avg improvement over FDD/MWKR (%)")
ax[0].set_xlabel("iteration"); ax[0].grid(alpha=.3)
ax[1].plot(history["iter"], history["init_mk"], label="init (FDD/MWKR)")
ax[1].plot(history["iter"], history["final_mk"], label="learned L2S")
ax[1].set_title("Makespan"); ax[1].set_xlabel("iteration"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

# =====================================================================
# CELL 14 - evaluation
# =====================================================================
code(r"""# Cell 14 — Evaluate the trained policy vs. random search on fresh test instances
@torch.no_grad()
def solve(policy, inst, cfg, steps=200, seed=0, greedy=True):
    rng = random.Random(seed)
    env = Environment(inst, cfg, rng)
    st = env.reset()
    curve = [env.best_makespan]
    for _ in range(steps):
        if env.done or len(st["cands"]) == 0:
            break
        move, _, _ = select_action(policy, st, DEVICE, greedy=greedy)
        st, r, done = env.step(move)
        curve.append(env.best_makespan)
    return env.init_makespan, env.best_makespan, curve

eval_cfg = Config(n_jobs=cfg.n_jobs, n_machines=cfg.n_machines, horizon_T=200,
                  embed_dim=cfg.embed_dim, n_layers=cfg.n_layers)
rows = []
for s in range(5):
    inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=10_000 + s)
    i_mk, l_mk, l_curve = solve(policy, inst, eval_cfg, steps=200, seed=s, greedy=True)
    _, r_mk, _ = random_rollout(inst, eval_cfg, steps=200, seed=s)
    rows.append((s, i_mk, r_mk, l_mk))
    print(f"test {s}: FDD/MWKR={i_mk:6.0f} | random-N5={r_mk:6.0f} | L2S(greedy)={l_mk:6.0f}")

avg_init = np.mean([r[1] for r in rows]); avg_rand = np.mean([r[2] for r in rows]); avg_l2s = np.mean([r[3] for r in rows])
print(f"\nAVG  init={avg_init:.1f}  random={avg_rand:.1f}  L2S={avg_l2s:.1f}  "
      f"(L2S improves init by {100*(avg_init-avg_l2s)/avg_init:.1f}%)")

# improvement curve on one instance
inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=12345)
_, _, lc = solve(policy, inst, eval_cfg, steps=200, seed=1, greedy=False)
plt.figure(figsize=(6,3)); plt.plot(lc); plt.xlabel("improvement step"); plt.ylabel("best makespan")
plt.title("Learned L2S search trajectory"); plt.grid(alpha=.3); plt.show()
""")

# =====================================================================
# CELL 15 - Gantt
# =====================================================================
code(r"""# Cell 15 — Gantt chart of the best schedule found by the trained policy
@torch.no_grad()
def best_schedule(policy, inst, cfg, steps=300, seed=0):
    rng = random.Random(seed)
    env = Environment(inst, cfg, rng)
    st = env.reset()
    best_seq = [list(s) for s in env.machine_seq]; best = env.best_makespan
    for _ in range(steps):
        if env.done or len(st["cands"]) == 0:
            break
        move, _, _ = select_action(policy, st, DEVICE, greedy=True)
        st, r, done = env.step(move)
        if env.makespan <= best:
            best = env.makespan; best_seq = [list(s) for s in env.machine_seq]
    return best_seq, best


def plot_gantt(inst, machine_seq, title=""):
    preds, succs = build_graph(inst, machine_seq)
    est, lst, mk, _ = cpm_schedule(inst, preds, succs)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 0.5 * inst.M + 1))
    for m in range(inst.M):
        for o in machine_seq[m]:
            j = o // inst.M
            ax.barh(m, inst.p[o], left=est[o], color=cmap(j % 20), edgecolor="black")
            ax.text(est[o] + inst.p[o] / 2, m, f"J{j}", va="center", ha="center", fontsize=7)
    ax.set_yticks(range(inst.M)); ax.set_yticklabels([f"M{m}" for m in range(inst.M)])
    ax.set_xlabel("time"); ax.set_title(f"{title} (makespan={mk:.0f})"); plt.tight_layout(); plt.show()

inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=2024)
seq, mk = best_schedule(policy, inst, eval_cfg, steps=300, seed=0)
plot_gantt(inst, seq, title="L2S best schedule")
""")

# =====================================================================
# CELL 16 - notes
# =====================================================================
md(r"""## Notes, tips & how to scale up

**Make it stronger / closer to the paper.** Increase capacity and search budget in `Config`:
```python
cfg = Config(n_jobs=10, n_machines=10,
             embed_dim=128, gin_hidden=128, gat_hidden=128, n_layers=4,
             horizon_T=500, n_step=10, batch_size=64,
             lr=5e-5, n_iterations=2000)
```
Then re-run Cells 12-14. Use a **GPU runtime** for the larger settings.

**Train on a fixed size, test on larger.** The GNN is size-agnostic, so a policy trained on
10×10 can be applied to 15×15, 20×20, … just call `solve(policy, big_instance, ...)`.

**Use real benchmarks.** Paste a classic JSSP instance (Taillard/ABZ/FT/LA format:
first line `J M`, then one `machine time machine time …` line per job) and parse it:
```python
inst = parse_standard_jssp(open("ta01.txt").read())
solve(policy, inst, eval_cfg, steps=2000)
```

**Differences vs. the paper (deliberate, for a runnable prototype):**
- Dense adjacency instead of `torch_geometric` sparse ops (fine for small/medium `N`).
- Smaller default network and horizon so it trains in minutes; bump them up as above.
- Added a mean-return baseline + small entropy bonus for stability.
- The paper's GPU-batched message-passing evaluator is included (`message_passing_est`)
  but the env uses the equivalent, faster CPM for the schedule.

**Where each L2S pillar lives:** loop=`Environment`, N5=Cell 6, GIN/GAT=Cell 10,
MDP/reward=Cell 8, message-passing evaluator=Cell 5.
""")


# ---------------------------------------------------------------------
def build():
    nb_cells = []
    for ctype, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": ctype, "metadata": {}, "source": lines}
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(__file__), "L2S_JSSP_Colab.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("Wrote", out, "with", len(nb_cells), "cells")


if __name__ == "__main__":
    build()
