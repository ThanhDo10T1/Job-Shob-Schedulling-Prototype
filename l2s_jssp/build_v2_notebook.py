"""
Generator for L2S_TBGAT_Colab_v2.ipynb — Improved L2S/TBGAT framework.

Key improvements over v1:
1. Higher defaults (10x10, embed=128, 4 heads) feasible on Colab T4
2. Multi-head GAT + torch_geometric for sparse graphs (scalable)
3. TBGAT bidirectional architecture: FEM (forward) + BEM (backward) with topological sort
4. Benchmark evaluation on classic datasets (FT, LA, Taillard, ABZ) from TBGAT repo
5. No comparison with random search — only vs. FDD/MWKR initial and known optimal
6. Entropy-regularized n-step REINFORCE (Algorithm 1 from TBGAT paper)
"""
import json, os, textwrap

CELLS = []

def md(src):
    CELLS.append(("markdown", textwrap.dedent(src).strip()))

def code(src):
    CELLS.append(("code", textwrap.dedent(src).strip()))


# ===========================================================================
# CELL 0 — Title
# ===========================================================================
md(r"""
# L2S + TBGAT for Job-Shop Scheduling — Colab Notebook v2

A **self-contained PyTorch + PyG** implementation of the L2S improvement-heuristic framework
(Zhang et al., ICLR 2024) enhanced with **TBGAT-style bidirectional multi-head graph attention**
(Zhang et al., UAI/ICML 2024) for solving the Job-Shop Scheduling Problem (JSSP).

## Key improvements over v1
| # | What | Detail |
|---|------|--------|
| 1 | **Higher default config** | 10x10, embed=128, 4 GAT heads — feasible on Colab T4 (16 GB) |
| 2 | **Multi-head GAT + `torch_geometric`** | Sparse message-passing; scales to large instances |
| 3 | **TBGAT bidirectional architecture** | FEM (forward view) + BEM (backward view) with topological sort features |
| 4 | **Benchmark evaluation** | FT 6x6/10x10, LA 10x5/15x5/20x5, Taillard 15x15 — with known optima |
| 5 | **No random-search comparison** | Evaluate against FDD/MWKR baseline and literature best-known solutions |
| 6 | **Entropy-regularized REINFORCE** | Faithful to Algorithm 1 in TBGAT paper |

### How to run
1. **Runtime → Change runtime type → T4 GPU**
2. **Run All** (Ctrl+F9)
3. Training takes ~10-15 min on T4 for default 10x10 config
""")


# ===========================================================================
# CELL 1 — Install torch_geometric
# ===========================================================================
code(r"""
# Cell 1 — Install torch_geometric (only needed on Colab; takes ~60s)
import subprocess, sys

def install_pyg():
    """Install PyG and its dependencies matching the current torch+CUDA versions."""
    import torch
    torch_ver = torch.__version__.split('+')[0]  # e.g. "2.5.1"
    cuda_tag = torch.version.cuda
    if cuda_tag is None:
        cuda_tag = "cpu"
    else:
        cuda_tag = "cu" + cuda_tag.replace(".", "")  # e.g. "cu121"
    print(f"PyTorch {torch_ver} | CUDA tag: {cuda_tag}")
    whl = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    pkgs = ["torch_scatter", "torch_sparse", "torch_geometric"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  {pkg} already installed")
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   pkg, "-f", whl])
    print("Done.")

install_pyg()
""")


# ===========================================================================
# CELL 2 — Environment check
# ===========================================================================
code(r"""
# Cell 2 — Environment check
import sys, torch, numpy as np
import torch_geometric
print(f"Python        : {sys.version.split()[0]}")
print(f"PyTorch       : {torch.__version__}")
print(f"PyG           : {torch_geometric.__version__}")
print(f"NumPy         : {np.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
""")


# ===========================================================================
# CELL 3 — Imports & Config
# ===========================================================================
code(r"""
# Cell 3 — Imports, seeding, and global configuration
import math, random, time, os, copy
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    # ---- problem ----
    n_jobs: int = 10
    n_machines: int = 10
    p_low: int = 1
    p_high: int = 99
    # ---- feature normalisation (following TBGAT paper) ----
    p_norm: float = 99.0
    t_norm: float = 1000.0
    # ---- model (TBGAT paper: embed=128, hidden=128, 4 heads, 3 layers) ----
    embed_dim: int = 128
    n_heads: int = 4          # multi-head attention (TBGAT default: 4)
    n_layers: int = 3         # L layers in FEM/BEM
    act_n_layers: int = 4     # L_A layers in action selection MLP
    leaky_slope: float = 0.2
    dropout: float = 0.0
    # ---- RL / training (TBGAT paper: lr=1e-5, EC=1e-5, n=10, T=500, B=64) ----
    horizon_T: int = 200       # improvement steps per episode (paper: 500)
    n_step: int = 10           # n in n-step REINFORCE
    gamma: float = 1.0
    batch_size: int = 32       # instances per iteration (paper: 64)
    n_iterations: int = 300    # training iterations (paper: 2000 batches)
    lr: float = 3e-5           # learning rate
    entropy_coef: float = 1e-5 # entropy regularization coefficient (paper: 1e-5)
    grad_clip: float = 1.0
    seed: int = 42
    # ---- evaluation ----
    eval_steps: int = 500      # improvement steps during evaluation

cfg = Config()
set_seed(cfg.seed)
print(cfg)
""")


# ===========================================================================
# CELL 4 — JSSP Instance
# ===========================================================================
code(r"""
# Cell 4 — JSSP instance: generation + parsers for standard format and TBGAT .npy format
class JSSPInstance:
    '''Represents a single JSSP instance.
    times[j][i]: processing time of j-th job's i-th operation.
    machines[j][i]: machine that processes j-th job's i-th operation.
    '''
    def __init__(self, times, machines):
        self.times = np.asarray(times, dtype=np.float64)
        self.machines = np.asarray(machines, dtype=np.int64)
        self.J, self.M = self.times.shape
        self.n_ops = self.J * self.M
        self.S = self.n_ops       # dummy source node
        self.T = self.n_ops + 1   # dummy sink node
        self.N = self.n_ops + 2   # total nodes
        # flat arrays
        self.p = np.zeros(self.N, dtype=np.float64)
        self.op_machine = np.full(self.N, -1, dtype=np.int64)
        self.op_job = np.full(self.N, -1, dtype=np.int64)
        for j in range(self.J):
            for i in range(self.M):
                o = j * self.M + i
                self.p[o] = self.times[j, i]
                self.op_machine[o] = self.machines[j, i]
                self.op_job[o] = j

    def op_id(self, j, i):
        return j * self.M + i


def generate_instance(J, M, seed=0, low=1, high=99):
    '''Generate a random JSSP instance (Taillard-style).'''
    rng = np.random.RandomState(seed)
    times = rng.randint(low, high + 1, size=(J, M))
    machines = np.stack([rng.permutation(M) for _ in range(J)])
    return JSSPInstance(times, machines)


def parse_standard_jssp(text):
    '''Parse classic JSSP format: first line "J M", then J lines of "machine time machine time ...".'''
    lines = [l for l in text.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    header = lines[0].split()
    J, M = int(header[0]), int(header[1])
    times = np.zeros((J, M), dtype=np.int64)
    machines = np.zeros((J, M), dtype=np.int64)
    for j in range(J):
        nums = list(map(int, lines[1 + j].split()))
        for i in range(M):
            machines[j, i] = nums[2 * i]
            times[j, i] = nums[2 * i + 1]
    return JSSPInstance(times, machines)


def load_tbgat_npy(filepath):
    '''Load TBGAT .npy benchmark file. Shape: (n_instances, 2, J, M).
    [k, 0] = machine assignments, [k, 1] = processing times.'''
    data = np.load(filepath)
    instances = []
    n, _, J, M = data.shape
    for k in range(n):
        machines = data[k, 0].astype(np.int64)
        times = data[k, 1].astype(np.int64)
        instances.append(JSSPInstance(times, machines))
    return instances


def load_tbgat_results(filepath):
    '''Load TBGAT _result.npy: best-known makespans. Shape: (n_instances,).'''
    return np.load(filepath).flatten()


# Quick demo
_inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=42)
print(f"Generated {_inst.J}x{_inst.M} instance | n_ops={_inst.n_ops} | N(with S,T)={_inst.N}")
print("Machine order of job 0:", _inst.machines[0])
print("Proc times of job 0  :", _inst.times[0])
""")


# ===========================================================================
# CELL 5 — Graph construction + CPM + MPTS
# ===========================================================================
code(r"""
# Cell 5 — Disjunctive graph, CPM (EST/LST), and MPTS (message-passing topological sort)

def build_graph(inst: JSSPInstance, machine_seq: list):
    '''Build the DAG from a complete solution. Returns predecessor/successor lists.'''
    N, S, T = inst.N, inst.S, inst.T
    preds = [[] for _ in range(N)]
    succs = [[] for _ in range(N)]
    def arc(u, v):
        succs[u].append(v)
        preds[v].append(u)
    # Conjunctive arcs (job precedence) + source/sink
    for j in range(inst.J):
        arc(S, inst.op_id(j, 0))
        for i in range(inst.M - 1):
            arc(inst.op_id(j, i), inst.op_id(j, i + 1))
        arc(inst.op_id(j, inst.M - 1), T)
    # Disjunctive arcs (machine order)
    for m in range(inst.M):
        seq = machine_seq[m]
        for a in range(len(seq) - 1):
            arc(seq[a], seq[a + 1])
    return preds, succs


def topo_order(preds, succs, N):
    '''Kahn's algorithm for topological sort. Raises if cycle detected.'''
    indeg = [len(preds[v]) for v in range(N)]
    dq = deque([v for v in range(N) if indeg[v] == 0])
    order = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    if len(order) != N:
        raise ValueError("Infeasible: cycle in disjunctive graph")
    return order


def cpm_schedule(inst: JSSPInstance, preds, succs):
    '''Critical Path Method: EST, LST, makespan.'''
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


def mpts_forward(preds, N, S):
    '''Message-Passing Topological Sort (forward). Returns rank per node.
    Implements Theorem 1 from TBGAT: iteratively apply MPO.'''
    msg = np.zeros(N)
    # Nodes with in-degree 0 get message 1
    for v in range(N):
        if len(preds[v]) == 0:
            msg[v] = 1.0
    ranks = np.zeros(N, dtype=np.float64)
    collected = set()
    rank_val = 0
    for iteration in range(N):
        # Collect nodes with msg == 1 that haven't been collected
        new_collected = []
        for v in range(N):
            if msg[v] >= 1.0 and v not in collected:
                new_collected.append(v)
                ranks[v] = rank_val
                collected.add(v)
        if not new_collected:
            break
        rank_val += 1
        # MPO: propagate max of predecessor messages
        new_msg = msg.copy()
        for v in range(N):
            if v in collected:
                continue
            if preds[v]:
                new_msg[v] = max(msg[u] for u in preds[v])
        msg = new_msg
    return ranks


def mpts_backward(succs, N, T):
    '''Message-Passing Topological Sort (backward). Returns rank per node.'''
    msg = np.zeros(N)
    for v in range(N):
        if len(succs[v]) == 0:
            msg[v] = 1.0
    ranks = np.zeros(N, dtype=np.float64)
    collected = set()
    rank_val = 0
    for iteration in range(N):
        new_collected = []
        for v in range(N):
            if msg[v] >= 1.0 and v not in collected:
                new_collected.append(v)
                ranks[v] = rank_val
                collected.add(v)
        if not new_collected:
            break
        rank_val += 1
        new_msg = msg.copy()
        for v in range(N):
            if v in collected:
                continue
            if succs[v]:
                new_msg[v] = max(msg[u] for u in succs[v])
        msg = new_msg
    return ranks


# Sanity check
_seq = [[] for _ in range(_inst.M)]
for j in range(_inst.J):
    for i in range(_inst.M):
        _seq[_inst.machines[j, i]].append(_inst.op_id(j, i))
_pr, _su = build_graph(_inst, _seq)
_est, _lst, _mk, _ord = cpm_schedule(_inst, _pr, _su)
_fwd = mpts_forward(_pr, _inst.N, _inst.S)
_bwd = mpts_backward(_su, _inst.N, _inst.T)
print(f"10x10 instance: makespan={_mk:.0f}")
print(f"MPTS forward ranks (first 5 ops): {_fwd[:5]}")
print(f"MPTS backward ranks (first 5 ops): {_bwd[:5]}")
""")


# ===========================================================================
# CELL 6 — Critical path + N5 neighborhood
# ===========================================================================
code(r"""
# Cell 6 — Critical path, critical blocks, and N5 neighbourhood
EPS = 1e-7

def find_critical_path(inst, preds, succs, est, lst, rng):
    '''Walk one S->T path over tight AND critical edges.'''
    S, T, p = inst.S, inst.T, inst.p
    path = [S]
    u = S
    while u != T:
        # Prefer critical nodes (est == lst) with tight arc
        cands = [v for v in succs[u]
                 if abs(est[v] - lst[v]) < EPS and abs(est[u] + p[u] - est[v]) < EPS]
        if not cands:
            cands = [v for v in succs[u] if abs(est[u] + p[u] - est[v]) < EPS]
        if not cands:
            # Fallback: any successor (shouldn't happen in a valid solution)
            cands = succs[u]
        u = cands[rng.randrange(len(cands))]
        path.append(u)
    return path


def critical_blocks(inst, path):
    '''Group consecutive ops on the critical path by machine.'''
    core = [o for o in path if o != inst.S and o != inst.T]
    blocks, cur, cur_m = [], [], None
    for o in core:
        m = inst.op_machine[o]
        if m == cur_m:
            cur.append(o)
        else:
            if cur:
                blocks.append(cur)
            cur, cur_m = [o], m
    if cur:
        blocks.append(cur)
    return blocks


def n5_candidate_moves(blocks):
    '''N5 neighbourhood: swap first/last adjacent pair in each critical block.
    First block -> only last pair; last block -> only first pair.
    |N5| <= 2*N_blocks - 2.  (Nowicki & Smutnicki, 1996)'''
    moves, nb = [], len(blocks)
    for bi, b in enumerate(blocks):
        L = len(b)
        if L < 2:
            continue
        first_pair = (b[0], b[1])
        last_pair = (b[L - 2], b[L - 1])
        if nb == 1:
            moves.append(first_pair)
            if L > 2:
                moves.append(last_pair)
        elif bi == 0:
            moves.append(last_pair)      # first block: only last pair
        elif bi == nb - 1:
            moves.append(first_pair)     # last block: only first pair
        else:
            moves.append(first_pair)
            if L > 2:
                moves.append(last_pair)
    # deduplicate
    seen = set()
    uniq = []
    for mv in moves:
        if mv not in seen:
            seen.add(mv)
            uniq.append(mv)
    return uniq

print("N5 neighbourhood defined.")
""")


# ===========================================================================
# CELL 7 — FDD/MWKR dispatch
# ===========================================================================
code(r"""
# Cell 7 — FDD/MWKR dispatching rule -> initial complete solution
# (Same rule used in L2S and TBGAT papers)

def fdd_mwkr_initial(inst: JSSPInstance):
    '''Build initial machine ordering using min FDD/MWKR priority.
    FDD = cumulative processing time up to current op (Flow Due Date).
    MWKR = remaining work of the job (Most Work Remaining).'''
    J, M, times = inst.J, inst.M, inst.times
    machine_seq = [[] for _ in range(M)]
    next_op = [0] * J
    cum = np.cumsum(times, axis=1)     # cumulative proc time
    total = times.sum(axis=1)          # total work per job
    for _ in range(inst.n_ops):
        best_j, best_pri = -1, float('inf')
        for j in range(J):
            i = next_op[j]
            if i >= M:
                continue
            fdd = float(cum[j, i])
            mwkr = float(total[j] - (cum[j, i] - times[j, i]))  # remaining incl current
            pri = fdd / max(mwkr, 1e-9)
            if pri < best_pri:
                best_pri, best_j = pri, j
        j = best_j
        i = next_op[j]
        machine_seq[inst.machines[j, i]].append(inst.op_id(j, i))
        next_op[j] += 1
    return machine_seq


# Quick test
_mseq = fdd_mwkr_initial(_inst)
_pr, _su = build_graph(_inst, _mseq)
_est, _lst, _mk, _ = cpm_schedule(_inst, _pr, _su)
print(f"FDD/MWKR initial makespan for 10x10 instance: {_mk:.0f}")
""")


# ===========================================================================
# CELL 8 — Environment (MDP)
# ===========================================================================
code(r"""
# Cell 8 — The L2S local-search environment (MDP)
# State  : complete solution graph + node features
# Action : N5 operation-pair (u, v) to swap
# Reward : max(incumbent_makespan - new_makespan, 0)

class Environment:
    def __init__(self, inst: JSSPInstance, cfg: Config, rng: random.Random):
        self.inst = inst
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        self.machine_seq = fdd_mwkr_initial(self.inst)
        self._recompute()
        self.init_makespan = self.makespan
        self.best_makespan = self.makespan
        self.best_seq = [list(s) for s in self.machine_seq]
        self.steps = 0
        self.done = False
        return self._state()

    def _recompute(self):
        self.preds, self.succs = build_graph(self.inst, self.machine_seq)
        self.est, self.lst, self.makespan, _ = cpm_schedule(
            self.inst, self.preds, self.succs)
        self.fwd_topo = mpts_forward(self.preds, self.inst.N, self.inst.S)
        self.bwd_topo = mpts_backward(self.succs, self.inst.N, self.inst.T)

    def _candidates(self):
        path = find_critical_path(self.inst, self.preds, self.succs,
                                  self.est, self.lst, self.rng)
        blocks = critical_blocks(self.inst, path)
        return n5_candidate_moves(blocks)

    def _state(self):
        '''Build PyG-compatible state with TBGAT features.'''
        inst, cfg = self.inst, self.cfg
        N = inst.N
        # --- TBGAT node features ---
        # Forward: (p, est, fwd_topo)   Backward: (p, lst, bwd_topo)
        fwd_topo_norm = self.fwd_topo / max(self.fwd_topo.max(), 1.0)
        bwd_topo_norm = self.bwd_topo / max(self.bwd_topo.max(), 1.0)
        feat_fwd = np.zeros((N, 3), dtype=np.float32)
        feat_fwd[:, 0] = inst.p / cfg.p_norm
        feat_fwd[:, 1] = self.est / cfg.t_norm
        feat_fwd[:, 2] = fwd_topo_norm
        feat_bwd = np.zeros((N, 3), dtype=np.float32)
        feat_bwd[:, 0] = inst.p / cfg.p_norm
        feat_bwd[:, 1] = self.lst / cfg.t_norm
        feat_bwd[:, 2] = bwd_topo_norm
        # --- Edge indices for PyG (sparse) ---
        # Forward edges: all arcs u -> v (predecessors point to node)
        fwd_src, fwd_dst = [], []
        bwd_src, bwd_dst = [], []
        for v in range(N):
            for u in self.preds[v]:
                fwd_src.append(u)
                fwd_dst.append(v)
        # Backward edges: reverse of all arcs
        for v in range(N):
            for u in self.succs[v]:
                bwd_src.append(u)
                bwd_dst.append(v)
        edge_fwd = np.array([fwd_src, fwd_dst], dtype=np.int64) if fwd_src else np.zeros((2, 0), dtype=np.int64)
        edge_bwd = np.array([bwd_src, bwd_dst], dtype=np.int64) if bwd_src else np.zeros((2, 0), dtype=np.int64)

        cands = self._candidates()
        return {
            "feat_fwd": feat_fwd, "feat_bwd": feat_bwd,
            "edge_fwd": edge_fwd, "edge_bwd": edge_bwd,
            "cands": cands,
            "makespan": self.makespan, "best": self.best_makespan, "init": self.init_makespan,
            "N": N,
        }

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
            self.best_seq = [list(s) for s in self.machine_seq]
        self.steps += 1
        st = self._state()
        if self.steps >= self.cfg.horizon_T or len(st["cands"]) == 0:
            self.done = True
        return st, reward, self.done


# Quick test
_env = Environment(_inst, cfg, random.Random(0))
_st = _env.reset()
print(f"Init makespan: {_env.init_makespan:.0f}")
print(f"N5 candidates: {len(_st['cands'])} moves")
print(f"Forward edge count: {_st['edge_fwd'].shape[1]}")
print(f"Backward edge count: {_st['edge_bwd'].shape[1]}")
print(f"Forward features shape: {_st['feat_fwd'].shape}")
""")


# ===========================================================================
# CELL 9 — TBGAT Policy Network
# ===========================================================================
code(r"""
# Cell 9 — TBGAT Policy Network: FEM + BEM (multi-head GAT) + Action Selection
# Faithful to the architecture described in Section 4.2 of the TBGAT paper.

class TBGATEmbeddingModule(nn.Module):
    '''One direction of the TBGAT: either Forward (FEM) or Backward (BEM).
    Uses multi-head GATv2Conv from PyG for attention-based message passing.'''

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int,
                 n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        head_dim = hidden_dim // n_heads
        for l in range(n_layers):
            in_ch = in_dim if l == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(in_ch, head_dim, heads=n_heads,
                          concat=True, dropout=dropout, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        '''x: (N, in_dim), edge_index: (2, E). Returns: (N, hidden_dim).'''
        for l in range(self.n_layers):
            x = self.convs[l](x, edge_index)
            x = self.norms[l](x)
            if l < self.n_layers - 1:
                x = F.elu(x)
        return x


class ActionSelectionMLP(nn.Module):
    '''MLP for computing a scalar score for each candidate action pair.
    Input: concatenation of [h_u, h_v, h_G] for candidate pair (u, v).
    Paper: L_A = 4 hidden layers, each with half the dimension of parent layer.'''

    def __init__(self, in_dim: int, n_layers: int = 4):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(n_layers):
            out_d = max(d // 2, 16)
            layers.append(nn.Linear(d, out_d))
            layers.append(nn.ELU())
            d = out_d
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TBGATPolicy(nn.Module):
    '''Full TBGAT policy network.
    Architecture (Fig. 5 of paper):
    1. FEM: forward GAT on forward view edges with features (p, est, fwd_topo)
    2. BEM: backward GAT on backward view edges with features (p, lst, bwd_topo)
    3. Merge: h_x = [h_fwd_x || h_bwd_x]
    4. Graph embedding: h_G = mean_pool(h_x)
    5. For each candidate pair (u,v): score = MLP([h_u || h_v || h_G])
    '''

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.fem = TBGATEmbeddingModule(3, d, cfg.n_layers, cfg.n_heads, cfg.dropout)
        self.bem = TBGATEmbeddingModule(3, d, cfg.n_layers, cfg.n_heads, cfg.dropout)
        # Action MLP: input = h_u(2d) + h_v(2d) + h_G(2d) = 6d
        self.action_mlp = ActionSelectionMLP(6 * d, cfg.act_n_layers)

    def embed(self, feat_fwd, feat_bwd, edge_fwd, edge_bwd):
        '''Compute node embeddings from both views.'''
        h_fwd = self.fem(feat_fwd, edge_fwd)   # (N, d)
        h_bwd = self.bem(feat_bwd, edge_bwd)   # (N, d)
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (N, 2d)
        h_G = h.mean(dim=0, keepdim=True)       # (1, 2d)
        return h, h_G

    def action_logits(self, h, h_G, cands):
        '''Compute logits for each candidate action.'''
        if len(cands) == 0:
            return torch.zeros(0, device=h.device)
        idx = torch.tensor(cands, dtype=torch.long, device=h.device)
        h_u = h[idx[:, 0]]    # (C, 2d)
        h_v = h[idx[:, 1]]    # (C, 2d)
        h_g = h_G.expand(len(cands), -1)  # (C, 2d)
        inp = torch.cat([h_u, h_v, h_g], dim=-1)  # (C, 6d)
        return self.action_mlp(inp)  # (C,)

    def forward(self, state, device):
        feat_fwd = torch.from_numpy(state["feat_fwd"]).to(device)
        feat_bwd = torch.from_numpy(state["feat_bwd"]).to(device)
        edge_fwd = torch.from_numpy(state["edge_fwd"]).to(device)
        edge_bwd = torch.from_numpy(state["edge_bwd"]).to(device)
        h, h_G = self.embed(feat_fwd, feat_bwd, edge_fwd, edge_bwd)
        return self.action_logits(h, h_G, state["cands"])


# Test model
_policy = TBGATPolicy(cfg).to(DEVICE)
n_params = sum(p.numel() for p in _policy.parameters())
print(f"TBGAT Policy: {n_params:,} parameters")
with torch.no_grad():
    _logits = _policy(_st, DEVICE)
    print(f"Logits shape for {len(_st['cands'])} candidates: {_logits.shape}")
del _policy
""")


# ===========================================================================
# CELL 10 — Action sampling
# ===========================================================================
code(r"""
# Cell 10 — Action selection: sample move + log-prob + entropy

def select_action(policy, state, device, greedy=False):
    '''Select an action from the policy.'''
    logits = policy(state, device)
    if logits.numel() == 0:
        return None, None, None
    dist = torch.distributions.Categorical(logits=logits)
    if greedy:
        a = torch.argmax(logits)
    else:
        a = dist.sample()
    logp = dist.log_prob(a)
    entropy = dist.entropy()
    move = state["cands"][a.item()]
    return move, logp, entropy

print("Action selection defined.")
""")


# ===========================================================================
# CELL 11 — Trainer (entropy-regularized n-step REINFORCE)
# ===========================================================================
code(r"""
# Cell 11 — Entropy-regularized n-step REINFORCE trainer
# Faithful to Algorithm 1 from TBGAT paper (Appendix D)

def compute_returns(rewards, gamma):
    '''Discounted returns from a list of rewards.'''
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def train(cfg: Config, log_every=20):
    set_seed(cfg.seed)
    policy = TBGATPolicy(cfg).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history = {"iter": [], "init_mk": [], "final_mk": [], "impr_pct": [], "loss": []}

    t_start = time.time()
    for it in range(cfg.n_iterations):
        # Generate fresh batch of instances on the fly
        insts = [generate_instance(cfg.n_jobs, cfg.n_machines,
                                   seed=cfg.seed + it * 10000 + b)
                 for b in range(cfg.batch_size)]
        envs = [Environment(inst, cfg, random.Random(cfg.seed + it * 7 + b))
                for b, inst in enumerate(insts)]
        states = [e.reset() for e in envs]
        init_mks = [e.init_makespan for e in envs]

        # n-step windows
        win_logps = [[] for _ in range(cfg.batch_size)]
        win_rews = [[] for _ in range(cfg.batch_size)]
        win_ents = [[] for _ in range(cfg.batch_size)]
        iter_loss = 0.0
        n_updates = 0

        for t in range(cfg.horizon_T):
            for b in range(cfg.batch_size):
                if envs[b].done:
                    continue
                if len(states[b]["cands"]) == 0:
                    envs[b].done = True
                    continue
                move, logp, ent = select_action(policy, states[b], DEVICE)
                if move is None:
                    envs[b].done = True
                    continue
                ns, r, done = envs[b].step(move)
                states[b] = ns
                win_logps[b].append(logp)
                win_ents[b].append(ent)
                win_rews[b].append(r / max(init_mks[b], 1.0))

            # Update every n steps (Algorithm 1, line 8-14)
            if (t + 1) % cfg.n_step == 0 or t == cfg.horizon_T - 1:
                loss = torch.tensor(0.0, device=DEVICE)
                count = 0
                for b in range(cfg.batch_size):
                    if not win_logps[b]:
                        continue
                    returns = compute_returns(win_rews[b], cfg.gamma)
                    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
                    baseline = returns.mean()
                    adv = returns - baseline
                    logps = torch.stack(win_logps[b])
                    ents = torch.stack(win_ents[b])
                    # REINFORCE with entropy regularization (Eq. in Appendix D)
                    loss = loss + (-(logps * adv.detach()).sum()
                                   - cfg.entropy_coef * ents.sum())
                    count += 1
                if count > 0:
                    loss = loss / count
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                    opt.step()
                    iter_loss += loss.item()
                    n_updates += 1
                # Reset windows
                win_logps = [[] for _ in range(cfg.batch_size)]
                win_rews = [[] for _ in range(cfg.batch_size)]
                win_ents = [[] for _ in range(cfg.batch_size)]

        final_mks = [e.best_makespan for e in envs]
        impr = float(np.mean([(i - f) / i * 100 for i, f in zip(init_mks, final_mks)]))
        history["iter"].append(it)
        history["init_mk"].append(float(np.mean(init_mks)))
        history["final_mk"].append(float(np.mean(final_mks)))
        history["impr_pct"].append(impr)
        history["loss"].append(iter_loss / max(n_updates, 1))
        if it % log_every == 0 or it == cfg.n_iterations - 1:
            elapsed = time.time() - t_start
            print(f"iter {it:4d}/{cfg.n_iterations} | "
                  f"init {np.mean(init_mks):7.1f} | best {np.mean(final_mks):7.1f} | "
                  f"impr {impr:5.2f}% | loss {history['loss'][-1]:+.4f} | "
                  f"time {elapsed:.0f}s")

    print(f"\nTraining complete in {time.time() - t_start:.0f}s")
    return policy, history
""")


# ===========================================================================
# CELL 12 — Run training
# ===========================================================================
code(r"""
# Cell 12 — Train on 10x10 (default config; ~10-15 min on T4)
cfg = Config()   # 10x10, embed=128, 4 heads, 3 layers
set_seed(cfg.seed)
policy, history = train(cfg, log_every=20)

# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["iter"], history["impr_pct"])
axes[0].set_title("Improvement over FDD/MWKR (%)")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("%"); axes[0].grid(alpha=0.3)
axes[1].plot(history["iter"], history["init_mk"], label="FDD/MWKR (init)")
axes[1].plot(history["iter"], history["final_mk"], label="TBGAT (best)")
axes[1].set_title("Average Makespan"); axes[1].set_xlabel("Iteration")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")


# ===========================================================================
# CELL 13 — Evaluation function
# ===========================================================================
code(r"""
# Cell 13 — Evaluation: solve instances with the trained policy (greedy)

@torch.no_grad()
def solve(policy, inst: JSSPInstance, cfg: Config, steps=500, seed=0, greedy=True):
    '''Run the trained policy on a single instance.'''
    rng = random.Random(seed)
    eval_cfg = Config(n_jobs=inst.J, n_machines=inst.M, horizon_T=steps,
                      embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                      n_layers=cfg.n_layers, act_n_layers=cfg.act_n_layers)
    env = Environment(inst, eval_cfg, rng)
    st = env.reset()
    curve = [env.best_makespan]
    for _ in range(steps):
        if env.done or len(st["cands"]) == 0:
            break
        move, _, _ = select_action(policy, st, DEVICE, greedy=greedy)
        if move is None:
            break
        st, r, done = env.step(move)
        curve.append(env.best_makespan)
    return env.init_makespan, env.best_makespan, curve, env.best_seq


@torch.no_grad()
def evaluate_instances(policy, instances, cfg, best_known=None,
                       steps=500, label="Benchmark"):
    '''Evaluate on a list of instances, optionally comparing to best-known solutions.'''
    results = []
    for i, inst in enumerate(instances):
        init_mk, best_mk, curve, best_seq = solve(policy, inst, cfg, steps=steps, seed=i)
        gap_init = (best_mk - init_mk) / init_mk * 100
        row = {"idx": i, "init": init_mk, "best": best_mk, "gap_vs_init": gap_init}
        if best_known is not None and i < len(best_known):
            bk = best_known[i]
            gap_bk = (best_mk - bk) / bk * 100
            row["optimal"] = bk
            row["gap_vs_optimal"] = gap_bk
        results.append(row)

    # Summary
    avg_init = np.mean([r["init"] for r in results])
    avg_best = np.mean([r["best"] for r in results])
    avg_impr = np.mean([(r["init"] - r["best"]) / r["init"] * 100 for r in results])
    print(f"\n{'='*60}")
    print(f"{label}: {len(instances)} instances, {steps} steps")
    print(f"  Avg FDD/MWKR: {avg_init:.1f}")
    print(f"  Avg TBGAT:    {avg_best:.1f}")
    print(f"  Avg improvement over init: {avg_impr:.2f}%")
    if best_known is not None:
        avg_gap = np.mean([r.get("gap_vs_optimal", 0) for r in results])
        print(f"  Avg gap to best-known:    {avg_gap:.2f}%")
    print(f"{'='*60}")
    return results

print("Evaluation functions defined.")
""")


# ===========================================================================
# CELL 14 — Evaluate on generated test instances
# ===========================================================================
code(r"""
# Cell 14 — Evaluate on fresh random 10x10 instances
test_insts = [generate_instance(cfg.n_jobs, cfg.n_machines, seed=90000 + i)
              for i in range(20)]
results_rnd = evaluate_instances(policy, test_insts, cfg, steps=cfg.eval_steps,
                                 label="Random 10x10 (20 instances)")

# Show per-instance detail
print("\n  idx |   init  |  TBGAT  | improvement")
print("  ----|---------|---------|------------")
for r in results_rnd:
    impr = (r['init'] - r['best']) / r['init'] * 100
    print(f"  {r['idx']:3d} | {r['init']:7.0f} | {r['best']:7.0f} | {impr:+.1f}%")
""")


# ===========================================================================
# CELL 15 — Benchmark data setup instructions
# ===========================================================================
md(r"""
## Benchmark Evaluation on Classic JSSP Datasets

To evaluate on classic benchmarks (FT, LA, Taillard, ABZ, etc.) from the TBGAT repo, follow these steps:

### Step 1: Download benchmark data files

Download the `.npy` files from [https://github.com/zcaicaros/TBGAT/tree/main/test_data_jssp](https://github.com/zcaicaros/TBGAT/tree/main/test_data_jssp).

Recommended files for evaluation:
```
ft6x6.npy          ft6x6_result.npy
ft10x10.npy        ft10x10_result.npy
la10x5.npy         la10x5_result.npy
la15x5.npy         la15x5_result.npy
la20x5.npy         la20x5_result.npy
la10x10.npy        la10x10_result.npy
tai15x15.npy       tai15x15_result.npy
syn10x10.npy       syn10x10_result.npy
```

### Step 2: Upload to Colab

**Option A — Manual upload:**
1. In Colab, click the **folder icon** (Files panel) on the left sidebar
2. Create a folder: `benchmark_data/` (right-click → New folder)
3. Upload all `.npy` files into `benchmark_data/`

**Option B — Clone from GitHub:**
```python
!git clone --depth 1 --filter=blob:none --sparse https://github.com/zcaicaros/TBGAT.git /content/TBGAT_repo
!cd /content/TBGAT_repo && git sparse-checkout set test_data_jssp
!cp /content/TBGAT_repo/test_data_jssp/*.npy /content/benchmark_data/
```

### Expected directory structure on Colab:
```
/content/
├── benchmark_data/
│   ├── ft6x6.npy
│   ├── ft6x6_result.npy
│   ├── ft10x10.npy
│   ├── ft10x10_result.npy
│   ├── la10x5.npy
│   ├── la10x5_result.npy
│   ├── la15x5.npy
│   ├── la15x5_result.npy
│   ├── la20x5.npy
│   ├── la20x5_result.npy
│   ├── la10x10.npy
│   ├── la10x10_result.npy
│   ├── tai15x15.npy
│   ├── tai15x15_result.npy
│   ├── syn10x10.npy
│   └── syn10x10_result.npy
└── L2S_TBGAT_Colab_v2.ipynb   ← this notebook
```
""")


# ===========================================================================
# CELL 16 — Auto-download benchmark data
# ===========================================================================
code(r"""
# Cell 16 — Auto-download benchmark data from TBGAT repo (if not already present)
import urllib.request, pathlib

BENCH_DIR = pathlib.Path("/content/benchmark_data")
BENCH_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/zcaicaros/TBGAT/main/test_data_jssp"

BENCHMARKS = {
    "ft6x6":    {"J": 6,  "M": 6,  "label": "FT 6x6 (Fisher & Thompson)"},
    "ft10x10":  {"J": 10, "M": 10, "label": "FT 10x10 (Fisher & Thompson)"},
    "la10x5":   {"J": 10, "M": 5,  "label": "LA 10x5 (Lawrence)"},
    "la15x5":   {"J": 15, "M": 5,  "label": "LA 15x5 (Lawrence)"},
    "la20x5":   {"J": 20, "M": 5,  "label": "LA 20x5 (Lawrence)"},
    "la10x10":  {"J": 10, "M": 10, "label": "LA 10x10 (Lawrence)"},
    "tai15x15": {"J": 15, "M": 15, "label": "Taillard 15x15"},
    "syn10x10": {"J": 10, "M": 10, "label": "Synthetic 10x10 (TBGAT)"},
}

for name in BENCHMARKS:
    for suffix in [".npy", "_result.npy"]:
        fpath = BENCH_DIR / f"{name}{suffix}"
        if not fpath.exists():
            url = f"{BASE_URL}/{name}{suffix}"
            print(f"Downloading {name}{suffix}...")
            try:
                urllib.request.urlretrieve(url, str(fpath))
            except Exception as e:
                print(f"  WARNING: Could not download {url}: {e}")
                print(f"  Please upload {name}{suffix} manually to {BENCH_DIR}")

print(f"\nBenchmark files in {BENCH_DIR}:")
for f in sorted(BENCH_DIR.glob("*.npy")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
""")


# ===========================================================================
# CELL 17 — Run benchmark evaluation
# ===========================================================================
code(r"""
# Cell 17 — Evaluate trained policy on classic JSSP benchmarks
import pathlib
BENCH_DIR = pathlib.Path("/content/benchmark_data")

all_results = {}
for name, info in BENCHMARKS.items():
    data_path = BENCH_DIR / f"{name}.npy"
    result_path = BENCH_DIR / f"{name}_result.npy"
    if not data_path.exists():
        print(f"Skipping {name}: {data_path} not found. Upload it first.")
        continue
    instances = load_tbgat_npy(str(data_path))
    best_known = load_tbgat_results(str(result_path)) if result_path.exists() else None
    print(f"\n--- Evaluating: {info['label']} ({len(instances)} instances) ---")
    results = evaluate_instances(
        policy, instances, cfg, best_known=best_known,
        steps=cfg.eval_steps, label=info["label"]
    )
    all_results[name] = results

# Summary table
print("\n" + "=" * 70)
print(f"{'Benchmark':<25} | {'#Inst':>5} | {'Avg Gap%':>8} | {'Avg Impr%':>9}")
print("-" * 70)
for name, results in all_results.items():
    n = len(results)
    avg_impr = np.mean([(r["init"] - r["best"]) / r["init"] * 100 for r in results])
    if "gap_vs_optimal" in results[0]:
        avg_gap = np.mean([r["gap_vs_optimal"] for r in results])
        print(f"{BENCHMARKS[name]['label']:<25} | {n:5d} | {avg_gap:7.2f}% | {avg_impr:8.2f}%")
    else:
        print(f"{BENCHMARKS[name]['label']:<25} | {n:5d} | {'N/A':>8} | {avg_impr:8.2f}%")
print("=" * 70)
""")


# ===========================================================================
# CELL 18 — Gantt chart
# ===========================================================================
code(r"""
# Cell 18 — Gantt chart of the best schedule found by the policy

def plot_gantt(inst, machine_seq, title=""):
    '''Plot a Gantt chart for a schedule.'''
    preds, succs = build_graph(inst, machine_seq)
    est, lst, mk, _ = cpm_schedule(inst, preds, succs)
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, max(0.5 * inst.M + 1, 3)))
    for m in range(inst.M):
        for o in machine_seq[m]:
            j = o // inst.M
            color = cmap(j % 20)
            ax.barh(m, inst.p[o], left=est[o], color=color, edgecolor="black", linewidth=0.5)
            ax.text(est[o] + inst.p[o] / 2, m, f"J{j}", va="center", ha="center", fontsize=6)
    ax.set_yticks(range(inst.M))
    ax.set_yticklabels([f"M{m}" for m in range(inst.M)])
    ax.set_xlabel("Time")
    ax.set_title(f"{title} (makespan = {mk:.0f})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# Plot on a test instance
_test_inst = generate_instance(cfg.n_jobs, cfg.n_machines, seed=2024)
_init_mk, _best_mk, _curve, _best_seq = solve(policy, _test_inst, cfg, steps=cfg.eval_steps, seed=0)
print(f"Init: {_init_mk:.0f} -> Best: {_best_mk:.0f} ({(_init_mk - _best_mk) / _init_mk * 100:.1f}% improvement)")
plot_gantt(_test_inst, _best_seq, title="TBGAT best schedule (10x10)")

# Plot improvement curve
plt.figure(figsize=(8, 3))
plt.plot(_curve)
plt.xlabel("Improvement step"); plt.ylabel("Best makespan")
plt.title("TBGAT search trajectory"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")


# ===========================================================================
# CELL 19 — Save/Load model
# ===========================================================================
code(r"""
# Cell 19 — Save and load model checkpoints

def save_model(policy, cfg, path="tbgat_policy.pt"):
    '''Save model state dict and config.'''
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": cfg.__dict__,
    }, path)
    print(f"Model saved to {path}")


def load_model(path="tbgat_policy.pt"):
    '''Load model from checkpoint.'''
    ckpt = torch.load(path, map_location=DEVICE)
    cfg_loaded = Config(**ckpt["config"])
    model = TBGATPolicy(cfg_loaded).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from {path}")
    return model, cfg_loaded

save_model(policy, cfg, "tbgat_10x10.pt")
""")


# ===========================================================================
# CELL 20 — Scaling notes
# ===========================================================================
md(r"""
## Architecture Reference & Scaling Guide

### TBGAT Architecture (this notebook)
| Component | Detail |
|-----------|--------|
| **FEM** (Forward Embedding Module) | 3-layer GATv2, 4 heads, input=(p, EST, fwd_topo_sort) |
| **BEM** (Backward Embedding Module) | 3-layer GATv2, 4 heads, input=(p, LST, bwd_topo_sort) |
| **Merge** | Concatenate FEM + BEM outputs: h_x = [h_fwd ∥ h_bwd] |
| **Graph embedding** | Mean pooling: h_G = mean(h_x) |
| **Action selection** | MLP on [h_u ∥ h_v ∥ h_G] → scalar score per N5 candidate |
| **Training** | n-step REINFORCE + entropy regularization |
| **Initial solution** | FDD/MWKR dispatching rule |
| **Neighbourhood** | N5 (Nowicki & Smutnicki, 1996) |

### To scale toward paper results
```python
cfg = Config(
    n_jobs=15, n_machines=15,    # or 20x15, 30x15, etc.
    embed_dim=128, n_heads=4, n_layers=3,
    horizon_T=500, n_step=10,
    batch_size=64,
    n_iterations=2000,           # 128000 instances total (paper)
    lr=1e-5,
    entropy_coef=1e-5,
    eval_steps=5000,             # more search steps for harder instances
)
```

### Key references
- **L2S**: Zhang et al., *Deep RL Guided Improvement Heuristic for JSP*, ICLR 2024
- **TBGAT**: Zhang et al., *Learning Topological Representations with Bidirectional GAT for JSSP*, UAI 2024
- **MPTS**: Message-Passing Topological Sort (Theorem 1 in TBGAT paper)
- **N5**: Nowicki & Smutnicki, *A Fast Taboo Search Algorithm for the Job Shop Problem*, 1996
""")


# ===========================================================================
# Build notebook
# ===========================================================================
def build():
    nb_cells = []
    for ctype, src in CELLS:
        lines = (src + "\n").splitlines(keepends=True)
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
            "gpuClass": "standard",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(__file__) or ".", "L2S_TBGAT_Colab_v2.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out} with {len(nb_cells)} cells")
    return out


if __name__ == "__main__":
    build()
