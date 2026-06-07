# L2S for Job-Shop Scheduling — Colab project

A compact, from-scratch **PyTorch** re-implementation of the *Learning-to-Search* (L2S)
improvement heuristic for the Job-Shop Scheduling Problem (JSSP), packaged as a
**Google Colab notebook** you can run cell by cell.

> Reference: Cong Zhang et al., *Deep Reinforcement Learning Guided Improvement Heuristic
> for Job Shop Scheduling*, ICLR 2024.

## Files

| File | Purpose |
|------|---------|
| `L2S_JSSP_Colab.ipynb` | **Main deliverable.** 16 cells: data → graph/CPM → N5 → env → GIN+GAT policy → n-step REINFORCE → train → evaluate → Gantt. Open in Colab and Run-All. |
| `build_notebook.py` | Generator that produces the `.ipynb` (edit here, then `python3 build_notebook.py`). |
| `_validate_core.py` | Pure-Python (no deps) check of the algorithmic core: confirms the message-passing EST evaluator equals CPM, and that N5 local search reduces makespan. |

## How to use

1. Upload `L2S_JSSP_Colab.ipynb` to [Google Colab](https://colab.research.google.com/)
   (or `File → Open notebook → GitHub`).
2. (Optional) `Runtime → Change runtime type → GPU`.
3. `Runtime → Run all`. Training on the default 6×6 setting takes a few minutes.

No `pip install` needed — it uses only `torch`, `numpy`, `matplotlib`, all pre-installed on Colab.

## How the L2S pillars map to the notebook

| L2S pillar | Where |
|------------|-------|
| Local-search loop | `Environment` (Cell 8) |
| N5 neighbourhood (critical blocks) | Cell 6 |
| Graph embedding: GIN (TPM) + GAT (CAM) | Cell 10 |
| MDP (state = complete-solution graph, reward = step improvement) | Cell 8 |
| Message-passing schedule evaluator (EST/LST) | Cell 5 |
| n-step REINFORCE | Cell 12 |

## Scaling toward the paper

Edit the `Config` in Cell 13:

```python
cfg = Config(n_jobs=10, n_machines=10,
             embed_dim=128, gin_hidden=128, gat_hidden=128, n_layers=4,
             horizon_T=500, n_step=10, batch_size=64,
             lr=5e-5, n_iterations=2000)
```

## Deliberate simplifications (for a runnable prototype)

- Dense adjacency matrices instead of `torch_geometric` sparse ops (ideal for small/medium `N`).
- Smaller default network + horizon so it trains in minutes; bump up as above.
- Added a mean-return baseline and a small entropy bonus for training stability.
