# L2S + TBGAT for Job-Shop Scheduling

A self-contained **PyTorch + PyG (torch_geometric)** implementation of the L2S improvement-heuristic framework enhanced with TBGAT-style bidirectional multi-head graph attention for solving the Job-Shop Scheduling Problem (JSSP).

## References

- **L2S**: Cong Zhang et al., *Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling*, ICLR 2024
- **TBGAT**: Cong Zhang et al., *Learning Topological Representations with Bidirectional Graph Attention Network for Solving Job Shop Scheduling Problem*, UAI 2024

## Files

| File | Purpose |
|------|---------|
| `L2S_TBGAT_Colab_v2.ipynb` | **Main notebook.** 21 cells: install PyG -> data -> graph/CPM/MPTS -> N5 -> env -> TBGAT policy (FEM+BEM) -> n-step REINFORCE -> train -> benchmark eval -> Gantt. Open in Colab and Run All. |
| `L2S_JSSP_Colab.ipynb` | v1 notebook (dense adjacency, GIN+GAT, smaller defaults). Kept for reference. |
| `build_notebook.py` | v1 notebook generator. |
| `build_v2_notebook.py` | v2 notebook generator (edit here, then `python3 build_v2_notebook.py`). |
| `_validate_core.py` | Pure-Python validation of the algorithmic core (no deps). |

## Quick Start

1. Upload `L2S_TBGAT_Colab_v2.ipynb` to [Google Colab](https://colab.research.google.com/)
2. **Runtime -> Change runtime type -> T4 GPU**
3. **Runtime -> Run all** (Ctrl+F9)
4. Cell 1 installs `torch_geometric` (~60s)
5. Training on default 10x10 config takes ~10-15 min on T4

No manual `pip install` needed beyond what Cell 1 does automatically.

## Architecture (v2)

Implements TBGAT (Fig. 5 of the paper):

```
                    Forward view of DG
                          |
                    [FEM: 3-layer GATv2, 4 heads]
                    input: (p, EST, fwd_topo_sort)
                          |
        h_fwd -----+-----+
                    |
                [Concatenate]  -->  h_x = [h_fwd || h_bwd]
                    |
        h_bwd -----+-----+
                          |
                    [BEM: 3-layer GATv2, 4 heads]
                    input: (p, LST, bwd_topo_sort)
                          |
                    Backward view of DG

        h_G = mean_pool(h_x)

        For each N5 candidate pair (u,v):
            score = MLP([h_u || h_v || h_G])

        Action ~ Categorical(scores)
```

### Key components mapped to the notebook

| L2S/TBGAT Component | Notebook Cell |
|---------------------|---------------|
| JSSP instance + parsers + TBGAT `.npy` loader | Cell 4 |
| Disjunctive graph + CPM (EST/LST) | Cell 5 |
| MPTS (Message-Passing Topological Sort) | Cell 5 |
| Critical path + N5 neighbourhood | Cell 6 |
| FDD/MWKR initial solution | Cell 7 |
| MDP Environment | Cell 8 |
| TBGAT Policy (FEM + BEM + Action MLP) | Cell 9 |
| n-step REINFORCE + entropy regularization | Cell 11 |
| Training loop | Cell 12 |
| Benchmark evaluation (FT/LA/Taillard) | Cells 16-17 |
| Gantt chart visualization | Cell 18 |

## Benchmark Evaluation

The notebook evaluates on classic JSSP benchmarks from the [TBGAT repo](https://github.com/zcaicaros/TBGAT):

- **FT** (Fisher & Thompson): 6x6, 10x10
- **LA** (Lawrence): 10x5, 15x5, 20x5, 10x10
- **Taillard**: 15x15
- **Synthetic**: 10x10

Cell 16 auto-downloads `.npy` benchmark files. Results are compared against best-known optimal solutions.

## Default Configuration (feasible on Colab T4)

```python
Config(
    n_jobs=10, n_machines=10,       # problem size
    embed_dim=128, n_heads=4,       # TBGAT architecture
    n_layers=3, act_n_layers=4,
    horizon_T=200, n_step=10,       # RL
    batch_size=32, n_iterations=300,
    lr=3e-5, entropy_coef=1e-5,
    eval_steps=500,
)
```

### Scaling toward paper results

```python
cfg = Config(
    n_jobs=15, n_machines=15,
    embed_dim=128, n_heads=4, n_layers=3,
    horizon_T=500, n_step=10,
    batch_size=64, n_iterations=2000,
    lr=1e-5, entropy_coef=1e-5,
    eval_steps=5000,
)
```

## Improvements over v1

| # | Aspect | v1 | v2 |
|---|--------|----|----|
| 1 | Default size | 6x6, embed=64 | 10x10, embed=128 |
| 2 | GNN | Dense GIN+GAT (single-head) | Sparse multi-head GATv2 via PyG |
| 3 | Architecture | L2S (GIN TPM + GAT CAM) | TBGAT (FEM + BEM, bidirectional) |
| 4 | Features | (p, est, lst) | Forward: (p, est, fwd_topo), Backward: (p, lst, bwd_topo) |
| 5 | Evaluation | vs. random search | vs. FDD/MWKR + best-known optima on FT/LA/Taillard |
| 6 | Training | REINFORCE + baseline | Entropy-regularized REINFORCE (Algorithm 1, TBGAT) |

## License

MIT
