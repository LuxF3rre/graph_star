# graph_star

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LuxF3rre/graph_star)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type%20checker-ty-blue.svg)](https://github.com/astral-sh/ty)
[![Build](https://github.com/LuxF3rre/graph_star/actions/workflows/test.yml/badge.svg)](https://github.com/LuxF3rre/graph_star/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/LuxF3rre/graph_star/graph/badge.svg?token=L3vByKOARN)](https://codecov.io/gh/LuxF3rre/graph_star)

Graph-based allocation optimizer for mapping hierarchical financial reports between accounting conventions (e.g. GAAP to IFRS).

Organizations routinely need to understand how concepts in one financial report map to another — for instance, how line items in a local-GAAP filing correspond to an IFRS standard. This is especially valuable at scale, where dozens of entity-level reports with different charts of accounts all need to be mapped to a single standard taxonomy. Doing this manually is tedious and error-prone; **graph_star** automates it.

Financial statements like balance sheets and P&L reports are hierarchical: leaf accounts roll up through intermediate groupings into top-level totals. When converting between conventions, the chart-of-accounts structure differs and leaf accounts don't map one-to-one. **graph_star** finds the best mapping of source leaf accounts to target leaf accounts using two complementary approaches: **numerical allocation** minimizes the total value distance across all levels of the hierarchy, while **semantic allocation** matches leaves by account-name similarity using sentence embeddings. The search space is combinatorial — with 50 source accounts and 30 target accounts there are 30⁵⁰ possible assignments — so brute force is out of the question and the library uses heuristic pipelines instead.

## How it works

Source and target reports are modelled as directed trees where edges point from child to parent. Each node carries a `Decimal` value. The optimizer allocates source leaves to target leaves, rolls up the allocated values through the target hierarchy, and measures the absolute difference at every node.

Two families of allocation strategy are provided:

### Numerical strategies

Value-based heuristics meant to be composed in a pipeline:

| Strategy | Purpose |
| --- | --- |
| `exact_walk` | Match source leaves (or small groups) whose values equal a target leaf exactly. |
| `greedy_walk` | Assign remaining sources one at a time to the target that reduces distance the most. |
| `greedy_optimization_walk` | Iteratively move, swap, delete, or add allocations until no single-step improvement exists. |
| `simulated_annealing_walk` | Escape local minima by accepting worse moves with decreasing probability. |

A typical numerical pipeline runs exact matching first, feeds the result into the greedy walk, then refines with optimization or annealing.

### Semantic strategies

Name-based matching using sentence embeddings. Instead of optimizing by value, these strategies match source and target leaves by the similarity of their account names (e.g. "Revenue from operations" → "Operating income"). Sources below the similarity threshold remain unallocated.

| Strategy | Purpose |
| --- | --- |
| `semantic_walk` | Allocate each source to its best-matching target above the similarity threshold. |

A convenience pipeline `run_semantic_pipeline` computes embeddings and similarity in one call.

Semantic and numerical allocation are **alternatives** — pick one or the other depending on whether name similarity or value proximity matters more for your use case.

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
        "total": Decimal("1000"),
        "sales":Decimal("600"),
        "interest income (expense)": Decimal("250"),
        "income (loss) from sale of equipment": Decimal("150"),
    },
    edges={"sales": "total", "interest income (expense)": "total", "income (loss) from sale of equipment": "total"},
)
source_leaves = leaf_nodes(graph=source, skip_zeros=True)

# Target report (e.g. IFRS)
target = create_graph(
    nodes={
        "total": Decimal("1000"),
        "core revenue": Decimal("600"),
        "non core revenue": Decimal("400"),
    },
    edges={"core revenue": "total", "non core revenue": "total"},
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

print(result.allocations)
print(result.distance)
```

For name-based matching using sentence embeddings:

```python
from graph_star import run_semantic_pipeline

result = run_semantic_pipeline(
    target_graph=target,
    target_leaves=target_leaves,
    source_graph=source,
    source_leaves=source_leaves,
    similarity_threshold=0.5,
)

print(result.allocations)
print(result.distance)
```

## Limitations

### Data quality

The optimizer takes node values at face value. If the input graphs contain incorrect balances, stale data, or rounding artefacts, the resulting allocation will faithfully minimise distance against those wrong numbers. Validate and reconcile source and target data before feeding it into the pipeline.

### Non-distinguishable leaves

Source leaves whose value is zero cannot be meaningfully attributed: assigning them to any target leaf produces no distance change, so the optimiser's placement is arbitrary. Similarly, source leaves that share the same value are interchangeable — swapping them between targets produces identical distances — so the specific assignment among equal-valued leaves is not unique.

### Hierarchy granularity

The algorithm allocates whole source leaves — it cannot split a single source leaf across multiple target leaves.

- **Source more granular than target (many small source leaves, fewer large target leaves).** Works well. Multiple source leaves are grouped onto each target leaf to match its value.
- **Same granularity.** Straightforward; exact matching often resolves most leaves directly.
- **Source less granular than target (few large source leaves, many small target leaves).** The algorithm still produces a useful result. Because the distance function sums the absolute difference at every node in the hierarchy (leaves *and* intermediates), it rewards allocations that are correct at aggregate levels even when individual leaf assignments are off. A coarse source leaf placed under a target subtree whose total matches will yield zero distance at every intermediate node above it, even though the leaf-level distances are non-zero. In practice this means the optimiser finds the "directionally correct" placement of large source amounts, but leaf-level precision is inherently limited by the source granularity.

### Semantic allocation

- **Language.** The default model (`BAAI/bge-large-en-v1.5`) is English-centric. For other languages, pass a multilingual model name to the `model_name` parameter.
- **Threshold tuning.** The `similarity_threshold` parameter controls the minimum cosine similarity for a match to be accepted. With the default bi-encoder model, scores are in [0, 1] and the default threshold is `0.75`. If you swap to a different model, check its model card on Hugging Face for the output range before setting a threshold. A practical approach: plot a histogram of each source's best-match score — the distribution is often bimodal and the valley between clusters is a natural threshold candidate.
- **Name quality.** Semantic matching relies on account names being descriptive. Cryptic codes (e.g. "4110", "BS-A3") produce poor embeddings; consider enriching labels with descriptions before calling the pipeline.

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
