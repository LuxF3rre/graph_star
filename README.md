# graph_star

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LuxF3rre/graph_star)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type%20checker-ty-blue.svg)](https://github.com/astral-sh/ty)
[![Build](https://github.com/LuxF3rre/sejm_scraper/actions/workflows/test.yml/badge.svg)](https://github.com/LuxF3rre/sejm_scraper/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/LuxF3rre/graph_star/branch/main/graph/badge.svg)](https://codecov.io/gh/LuxF3rre/repo-graph_star)

Graph-based allocation optimizer for mapping hierarchical financial reports between accounting conventions (e.g. GAAP to IFRS).

Financial statements like balance sheets and P&L reports are hierarchical: leaf accounts roll up through intermediate groupings into top-level totals. When converting between conventions, the chart-of-accounts structure differs and leaf accounts don't map one-to-one. **graph_star** finds the allocation of source leaf accounts to target leaf accounts that minimizes the total distance across all levels of the hierarchy.

## How it works

Source and target reports are modelled as directed trees where edges point from child to parent. Each node carries a `Decimal` value. The optimizer allocates source leaves to target leaves, rolls up the allocated values through the target hierarchy, and measures the absolute difference at every node.

Four allocation strategies are provided, meant to be composed in a pipeline:

| Strategy | Purpose |
|---|---|
| `exact_walk` | Match source leaves (or small groups) whose values equal a target leaf exactly. |
| `greedy_walk` | Assign remaining sources one at a time to the target that reduces distance the most. |
| `greedy_optimization_walk` | Iteratively move, swap, delete, or add allocations until no single-step improvement exists. |
| `simulated_annealing_walk` | Escape local minima by accepting worse moves with decreasing probability. |

A typical pipeline runs exact matching first, feeds the result into the greedy walk, then refines with optimization or annealing.

## Installation

Requires Python 3.12+.

Install from source:

```bash
git clone https://github.com/LuxF3rre/graph_star
cd graph_star
uv sync
```

## Quick start

```python
from decimal import Decimal
from graph_star import create_graph, leaf_nodes, run_greedy_pipeline

# Source report (e.g. GAAP)
source = create_graph(
    nodes={
        "total":    Decimal("1000"),
        "revenue":  Decimal("600"),
        "services": Decimal("250"),
        "other":    Decimal("150"),
    },
    edges={"revenue": "total", "services": "total", "other": "total"},
)
source_leaves = leaf_nodes(graph=source, skip_zeros=True)

# Target report (e.g. IFRS)
target = create_graph(
    nodes={
        "total":        Decimal("1000"),
        "core_revenue": Decimal("800"),
        "non_core":     Decimal("200"),
    },
    edges={"core_revenue": "total", "non_core": "total"},
)
target_leaves = leaf_nodes(graph=target, skip_zeros=True)

# Run the full pipeline: exact matching -> greedy allocation -> local optimization
result = run_greedy_pipeline(
    target_graph=target,
    target_leaves=target_leaves,
    source_graph=source,
    source_leaves=source_leaves,
)

print(result.allocations)   # target leaf -> [source leaves]
print(result.distance)      # total hierarchical distance
```

For simulated annealing instead of greedy optimization, swap in `run_annealing_pipeline`:

```python
from graph_star import run_annealing_pipeline

result = run_annealing_pipeline(
    target_graph=target,
    target_leaves=target_leaves,
    source_graph=source,
    source_leaves=source_leaves,
    seed=42,
)
```

## Limitations

### Data quality

The optimizer takes node values at face value. If the input graphs contain incorrect balances, stale data, or rounding artefacts, the resulting allocation will faithfully minimise distance against those wrong numbers. Validate and reconcile source and target data before feeding it into the pipeline.

### Hierarchy granularity

The algorithm allocates whole source leaves — it cannot split a single source leaf across multiple target leaves.

- **Source more granular than target (many small source leaves, fewer large target leaves).** Works well. Multiple source leaves are grouped onto each target leaf to match its value.
- **Same granularity.** Straightforward; exact matching often resolves most leaves directly.
- **Source less granular than target (few large source leaves, many small target leaves).** The algorithm still produces a useful result. Because the distance function sums the absolute difference at every node in the hierarchy (leaves *and* intermediates), it rewards allocations that are correct at aggregate levels even when individual leaf assignments are off. A coarse source leaf placed under a target subtree whose total matches will yield zero distance at every intermediate node above it, even though the leaf-level distances are non-zero. In practice this means the optimiser finds the "directionally correct" placement of large source amounts, but leaf-level precision is inherently limited by the source granularity.

## Development

```bash
uv sync --all-extras

uv run ruff check --fix        # lint
uv run ruff format             # format
uv run ty check src tests      # type check
uv run pytest -v --cov --cov-branch --cov-fail-under=90
```

## License

See [LICENSE](LICENSE) for details.
