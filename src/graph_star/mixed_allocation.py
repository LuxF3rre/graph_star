"""Mixed allocation using semantic similarity to break ties in exact matching."""

from itertools import combinations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from graph_star.allocation import (
    VALUE,
    AllocationWithContext,
    TargetToSourceAllocations,
    distance,
    evaluate_target_rollups,
    get_unallocated_source_leaves,
    get_unallocated_target_leaves,
)
from graph_star.semantic_allocation import Embeddings

__all__ = [
    "mixed_exact_walk",
]


def _cosine_similarity(
    *,
    vec_a: NDArray[np.float32],
    vec_b: NDArray[np.float32],
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or -1.0 if either vector has zero norm.
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def mixed_exact_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    source_embeddings: Embeddings,
    target_embeddings: Embeddings,
    max_group_size: int | None = 4,
) -> AllocationWithContext:
    """Find exact-value allocations using semantic similarity to break ties.

    Replaces the arbitrary iteration-order matching of `exact_walk` with
    semantically-informed matching: all numerically exact matches — whether
    1:1 or group-to-one — compete in a single candidate pool sorted by
    cosine similarity, so the most semantically similar match always wins
    regardless of group size.

    For each group size from 1 up to ``max_group_size``, every combination
    of source leaves whose summed value equals a target leaf value is added
    to the candidate pool.  A 1:1 match is simply a group of size 1.  For
    groups of size >= 2 the average source embedding is re-normalized before
    computing similarity.  All candidates are then sorted by descending
    similarity and greedily assigned.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        source_embeddings: L2-normalized embeddings for `source_leaves`,
            shape `(len(source_leaves), dim)`.
        target_embeddings: L2-normalized embeddings for `target_leaves`,
            shape `(len(target_leaves), dim)`.
        max_group_size: Maximum number of sources to combine when searching
            for group matches. ``1`` disables groups. ``None`` removes the
            limit.

    Returns:
        Allocation result with exact matches found, preferring semantic
        similarity when values are equal.
    """
    source_idx = {leaf: i for i, leaf in enumerate(source_leaves)}
    target_idx = {leaf: i for i, leaf in enumerate(target_leaves)}

    n = len(source_leaves)
    upper_bound = n + 1 if max_group_size is None else min(n + 1, max_group_size + 1)

    # --- Unified candidate pool: 1:1 and group matches compete together ---
    candidates: list[tuple[tuple[str, ...], str, float]] = []
    for length in range(1, upper_bound):
        for group in combinations(source_leaves, length):
            total_value = sum(source_graph.nodes[leaf][VALUE] for leaf in group)
            for target_leaf in target_leaves:
                if total_value == target_graph.nodes[target_leaf][VALUE]:
                    group_emb = np.mean(
                        [source_embeddings[source_idx[s]] for s in group],
                        axis=0,
                    )
                    if length > 1:
                        norm = float(np.linalg.norm(group_emb))
                        if norm > 0.0:
                            group_emb = group_emb / norm
                    sim = _cosine_similarity(
                        vec_a=group_emb,
                        vec_b=target_embeddings[target_idx[target_leaf]],
                    )
                    candidates.append((group, target_leaf, sim))

    candidates.sort(key=lambda c: c[2], reverse=True)

    allocations: TargetToSourceAllocations = TargetToSourceAllocations({})
    used_sources: set[str] = set()
    used_targets: set[str] = set()

    for group, target_leaf, _sim in candidates:
        if target_leaf in used_targets or any(s in used_sources for s in group):
            continue
        allocations[target_leaf] = list(group)
        used_sources.update(group)
        used_targets.add(target_leaf)

    # --- Finalization ---
    for target_leaf in target_leaves:
        if target_leaf not in allocations:
            allocations[target_leaf] = []

    return AllocationWithContext(
        allocations=allocations,
        distance=distance(
            target_graph=evaluate_target_rollups(
                target_graph=target_graph,
                target_leaves=target_leaves,
                source_graph=source_graph,
                allocations=allocations,
            ),
        ),
        unallocated_target_leaves=get_unallocated_target_leaves(
            target_leaves=target_leaves, allocations=allocations
        ),
        unallocated_source_leaves=get_unallocated_source_leaves(
            source_leaves=source_leaves, allocations=allocations
        ),
    )
