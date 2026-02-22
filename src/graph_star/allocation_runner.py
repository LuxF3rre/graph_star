"""High-level pipelines that chain allocation walk steps."""

from decimal import Decimal

import networkx as nx

from graph_star.allocation import (
    AllocationWithContext,
    exact_walk,
    greedy_optimization_walk,
    greedy_walk,
    simulated_annealing_walk,
)
from graph_star.mixed_allocation import mixed_exact_walk
from graph_star.semantic_allocation import (
    compute_embeddings,
    compute_similarity_matrix,
    semantic_walk,
)

__all__ = [
    "run_annealing_pipeline",
    "run_greedy_pipeline",
    "run_mixed_annealing_pipeline",
    "run_mixed_greedy_pipeline",
    "run_semantic_pipeline",
]


def run_greedy_pipeline(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    max_group_size: int | None = 4,
    exclude_source_leaves: list[str] | None = None,
    max_iterations: int | None = 1000,
) -> AllocationWithContext:
    """Run exact_walk -> greedy_walk -> greedy_optimization_walk.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        max_group_size: Maximum group size for exact matching.
        exclude_source_leaves: Source leaves to exclude from optimization.
        max_iterations: Maximum optimization iterations.

    Returns:
        Optimized allocation result.
    """
    exact_result = exact_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        max_group_size=max_group_size,
    )

    greedy_result = greedy_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        starting_allocations=exact_result.allocations,
    )

    return greedy_optimization_walk(
        previous_best_proposed_allocations=greedy_result.allocations,
        previous_best_distance=greedy_result.distance,
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        exclude_source_leaves=exclude_source_leaves,
        max_iterations=max_iterations,
    )


def run_annealing_pipeline(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    max_group_size: int | None = 4,
    temperature: Decimal = Decimal("1"),
    cooling_rate: Decimal = Decimal("0.95"),
    min_temperature: Decimal = Decimal("0.01"),
    iterations_per_temp: int = 100,
    seed: int | None = None,
) -> AllocationWithContext:
    """Run exact_walk -> greedy_walk -> simulated_annealing_walk.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        max_group_size: Maximum group size for exact matching.
        temperature: Initial SA temperature.
        cooling_rate: SA cooling rate.
        min_temperature: SA minimum temperature.
        iterations_per_temp: SA iterations per temperature step.
        seed: Random seed for reproducibility.

    Returns:
        Best allocation found during the annealing process.
    """
    exact_result = exact_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        max_group_size=max_group_size,
    )

    greedy_result = greedy_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        starting_allocations=exact_result.allocations,
    )

    return simulated_annealing_walk(
        starting_allocation=greedy_result.allocations,
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        temperature=temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        iterations_per_temp=iterations_per_temp,
        seed=seed,
    )


def run_semantic_pipeline(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    similarity_threshold: float = 0.75,
) -> AllocationWithContext:
    """Run embedding-only semantic allocation pipeline.

    Computes sentence embeddings for source and target leaf names, builds
    a cosine similarity matrix, and allocates by best match above the
    threshold.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        model_name: HuggingFace bi-encoder model identifier.
        similarity_threshold: Minimum cosine similarity for allocation.

    Returns:
        Allocation result with semantically matched leaves.
    """
    source_emb = compute_embeddings(
        labels=source_leaves,
        model_name=model_name,
    )
    target_emb = compute_embeddings(
        labels=target_leaves,
        model_name=model_name,
    )
    sim = compute_similarity_matrix(
        source_embeddings=source_emb,
        target_embeddings=target_emb,
    )
    return semantic_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        similarity_matrix=sim,
        similarity_threshold=similarity_threshold,
    )


def run_mixed_greedy_pipeline(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    max_group_size: int | None = 4,
    exclude_source_leaves: list[str] | None = None,
    max_iterations: int | None = 1000,
) -> AllocationWithContext:
    """Run mixed_exact_walk -> greedy_walk -> greedy_optimization_walk.

    Uses sentence embeddings to break ties when multiple source-target pairs
    share the same value during exact matching, then continues with the
    standard greedy pipeline.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        model_name: HuggingFace bi-encoder model identifier.
        max_group_size: Maximum group size for exact matching.
        exclude_source_leaves: Source leaves to exclude from optimization.
        max_iterations: Maximum optimization iterations.

    Returns:
        Optimized allocation result.
    """
    source_emb = compute_embeddings(labels=source_leaves, model_name=model_name)
    target_emb = compute_embeddings(labels=target_leaves, model_name=model_name)

    mixed_result = mixed_exact_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        source_embeddings=source_emb,
        target_embeddings=target_emb,
        max_group_size=max_group_size,
    )

    greedy_result = greedy_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        starting_allocations=mixed_result.allocations,
    )

    return greedy_optimization_walk(
        previous_best_proposed_allocations=greedy_result.allocations,
        previous_best_distance=greedy_result.distance,
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        exclude_source_leaves=exclude_source_leaves,
        max_iterations=max_iterations,
    )


def run_mixed_annealing_pipeline(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    max_group_size: int | None = 4,
    temperature: Decimal = Decimal("1"),
    cooling_rate: Decimal = Decimal("0.95"),
    min_temperature: Decimal = Decimal("0.01"),
    iterations_per_temp: int = 100,
    seed: int | None = None,
) -> AllocationWithContext:
    """Run mixed_exact_walk -> greedy_walk -> simulated_annealing_walk.

    Uses sentence embeddings to break ties when multiple source-target pairs
    share the same value during exact matching, then continues with the
    standard annealing pipeline.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        model_name: HuggingFace bi-encoder model identifier.
        max_group_size: Maximum group size for exact matching.
        temperature: Initial SA temperature.
        cooling_rate: SA cooling rate.
        min_temperature: SA minimum temperature.
        iterations_per_temp: SA iterations per temperature step.
        seed: Random seed for reproducibility.

    Returns:
        Best allocation found during the annealing process.
    """
    source_emb = compute_embeddings(labels=source_leaves, model_name=model_name)
    target_emb = compute_embeddings(labels=target_leaves, model_name=model_name)

    mixed_result = mixed_exact_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        source_embeddings=source_emb,
        target_embeddings=target_emb,
        max_group_size=max_group_size,
    )

    greedy_result = greedy_walk(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        starting_allocations=mixed_result.allocations,
    )

    return simulated_annealing_walk(
        starting_allocation=greedy_result.allocations,
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        temperature=temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        iterations_per_temp=iterations_per_temp,
        seed=seed,
    )
