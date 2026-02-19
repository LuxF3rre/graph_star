"""Semantic allocation using sentence embeddings for name-based matching."""

from collections import defaultdict
from functools import cache
from typing import Any, NewType

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from graph_star.allocation import (
    AllocationWithContext,
    TargetToSourceAllocations,
    distance,
    evaluate_target_rollups,
    get_unallocated_source_leaves,
    get_unallocated_target_leaves,
)


@cache
def _load_sentence_transformer(model_name: str) -> Any:
    """Load and cache a SentenceTransformer model by name."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(model_name)


__all__ = [
    "Embeddings",
    "SimilarityMatrix",
    "compute_embeddings",
    "compute_similarity_matrix",
    "semantic_walk",
]

Embeddings = NewType("Embeddings", NDArray[np.float32])
SimilarityMatrix = NewType("SimilarityMatrix", NDArray[np.float32])


def compute_embeddings(
    *,
    labels: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> Embeddings:
    """Compute sentence embeddings for a list of labels.

    Args:
        labels: Text labels to embed.
        model_name: HuggingFace model identifier for the bi-encoder.

    Returns:
        L2-normalized embedding matrix of shape `(len(labels), dim)`.

    Raises:
        ValueError: If `labels` is empty.
    """
    if not labels:
        msg = "labels must not be empty"
        raise ValueError(msg)

    model = _load_sentence_transformer(model_name)
    embeddings: NDArray[np.float32] = model.encode(
        labels,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return Embeddings(embeddings)


def compute_similarity_matrix(
    *,
    source_embeddings: Embeddings,
    target_embeddings: Embeddings,
) -> SimilarityMatrix:
    """Compute cosine similarity matrix between source and target embeddings.

    Both embedding matrices must be L2-normalized so the dot product equals
    cosine similarity.

    Args:
        source_embeddings: Source embedding matrix of shape
            `(n_source, dim)`.
        target_embeddings: Target embedding matrix of shape
            `(n_target, dim)`.

    Returns:
        Similarity matrix of shape `(n_source, n_target)`.

    Raises:
        ValueError: If embedding dimensions do not match.
    """
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        msg = (
            f"embedding dimensions do not match:"
            f" source={source_embeddings.shape[1]},"
            f" target={target_embeddings.shape[1]}"
        )
        raise ValueError(msg)

    matrix: NDArray[np.float32] = source_embeddings @ target_embeddings.T
    return SimilarityMatrix(matrix.astype(np.float32))


def semantic_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    similarity_matrix: SimilarityMatrix,
    similarity_threshold: float = 0.75,
) -> AllocationWithContext:
    """Allocate source leaves to target leaves by name similarity.

    For each source leaf, selects the target leaf with the highest
    similarity score. Sources whose best similarity falls below the
    threshold remain unallocated.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        similarity_matrix: Pre-computed similarity matrix of shape
            `(len(source_leaves), len(target_leaves))`.
        similarity_threshold: Minimum similarity for allocation.

    Returns:
        Allocation result with semantically matched leaves.

    Raises:
        ValueError: If matrix shape does not match leaf list lengths.
    """
    if similarity_matrix.shape != (len(source_leaves), len(target_leaves)):
        msg = (
            f"similarity_matrix shape {similarity_matrix.shape} does not"
            f" match (source={len(source_leaves)},"
            f" target={len(target_leaves)})"
        )
        raise ValueError(msg)

    allocations: TargetToSourceAllocations = TargetToSourceAllocations(
        defaultdict(list)
    )

    for i, source_leaf in enumerate(source_leaves):
        best_target_idx = int(np.argmax(similarity_matrix[i]))
        best_similarity = float(similarity_matrix[i, best_target_idx])

        if best_similarity >= similarity_threshold:
            allocations[target_leaves[best_target_idx]].append(source_leaf)

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
            target_leaves=target_leaves,
            allocations=allocations,
        ),
        unallocated_source_leaves=get_unallocated_source_leaves(
            source_leaves=source_leaves,
            allocations=allocations,
        ),
    )
