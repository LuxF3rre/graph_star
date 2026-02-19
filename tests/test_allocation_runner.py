"""Tests for graph_star.allocation_runner module."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

import graph_star
from graph_star.allocation import (
    AllocationWithContext,
    create_graph,
    exact_walk,
    greedy_walk,
    leaf_nodes,
)
from graph_star.allocation_runner import (
    run_annealing_pipeline,
    run_greedy_pipeline,
    run_semantic_pipeline,
)


@pytest.fixture
def mismatched_target_graph() -> nx.DiGraph:
    """Target tree with values that don't exactly match any source."""
    return create_graph(
        nodes={
            "t_root": Decimal("100"),
            "t1": Decimal("55"),
            "t2": Decimal("45"),
        },
        edges={"t1": "t_root", "t2": "t_root"},
    )


@pytest.fixture
def mismatched_target_leaves(mismatched_target_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=mismatched_target_graph, skip_zeros=False)


@pytest.fixture
def mismatched_source_graph() -> nx.DiGraph:
    """Source tree with values that don't exactly match any target."""
    return create_graph(
        nodes={
            "s_root": Decimal("100"),
            "s1": Decimal("30"),
            "s2": Decimal("30"),
            "s3": Decimal("40"),
        },
        edges={"s1": "s_root", "s2": "s_root", "s3": "s_root"},
    )


@pytest.fixture
def mismatched_source_leaves(mismatched_source_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=mismatched_source_graph, skip_zeros=False)


class TestRunGreedyPipeline:
    def test_produces_valid_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
        )
        assert isinstance(result, AllocationWithContext)
        assert result.distance >= Decimal("0")

    def test_perfect_match_yields_zero_distance(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
        )
        assert result.distance == Decimal("0")

    def test_distance_at_most_greedy_only(
        self,
        mismatched_target_graph: nx.DiGraph,
        mismatched_target_leaves: list[str],
        mismatched_source_graph: nx.DiGraph,
        mismatched_source_leaves: list[str],
    ) -> None:
        exact_result = exact_walk(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
        )
        greedy_result = greedy_walk(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
            starting_allocations=exact_result.allocations,
        )

        pipeline_result = run_greedy_pipeline(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
        )
        assert pipeline_result.distance <= greedy_result.distance

    def test_unallocated_sets_are_correct_types(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
        )
        assert isinstance(result.unallocated_source_leaves, frozenset)
        assert isinstance(result.unallocated_target_leaves, frozenset)

    def test_exclude_source_leaves_forwarded(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            exclude_source_leaves=["s1"],
        )
        assert isinstance(result, AllocationWithContext)

    def test_max_iterations_forwarded(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            max_iterations=1,
        )
        assert isinstance(result, AllocationWithContext)

    def test_max_group_size_forwarded(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            max_group_size=1,
        )
        assert isinstance(result, AllocationWithContext)


class TestRunAnnealingPipeline:
    def test_produces_valid_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        assert isinstance(result, AllocationWithContext)
        assert result.distance >= Decimal("0")

    def test_perfect_match_yields_zero_distance(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        assert result.distance == Decimal("0")

    def test_distance_at_most_greedy_only(
        self,
        mismatched_target_graph: nx.DiGraph,
        mismatched_target_leaves: list[str],
        mismatched_source_graph: nx.DiGraph,
        mismatched_source_leaves: list[str],
    ) -> None:
        exact_result = exact_walk(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
        )
        greedy_result = greedy_walk(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
            starting_allocations=exact_result.allocations,
        )

        pipeline_result = run_annealing_pipeline(
            target_graph=mismatched_target_graph,
            target_leaves=mismatched_target_leaves,
            source_graph=mismatched_source_graph,
            source_leaves=mismatched_source_leaves,
            seed=42,
            temperature=Decimal("2"),
            cooling_rate=Decimal("0.9"),
            min_temperature=Decimal("0.01"),
            iterations_per_temp=50,
        )
        assert pipeline_result.distance <= greedy_result.distance

    def test_deterministic_with_seed(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result1 = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=99,
        )
        result2 = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=99,
        )
        assert result1.distance == result2.distance
        assert result1.allocations == result2.allocations

    def test_unallocated_sets_are_correct_types(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        assert isinstance(result.unallocated_source_leaves, frozenset)
        assert isinstance(result.unallocated_target_leaves, frozenset)

    def test_sa_parameters_forwarded(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            temperature=Decimal("0.5"),
            cooling_rate=Decimal("0.8"),
            min_temperature=Decimal("0.1"),
            iterations_per_temp=10,
            seed=42,
        )
        assert isinstance(result, AllocationWithContext)

    def test_max_group_size_forwarded(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            max_group_size=1,
            seed=42,
        )
        assert isinstance(result, AllocationWithContext)


class TestRunSemanticPipeline:
    @patch("graph_star.semantic_allocation._load_sentence_transformer")
    def test_produces_valid_allocation(
        self,
        mock_load_st: MagicMock,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        mock_model = MagicMock()

        def _encode(labels: list[str], **_kwargs: object) -> np.ndarray:
            n = len(labels)
            rng = np.random.default_rng(42)
            emb = rng.random((n, 64)).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / norms

        mock_model.encode.side_effect = _encode
        mock_load_st.return_value = mock_model

        result = run_semantic_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            similarity_threshold=0.0,
        )
        assert isinstance(result, AllocationWithContext)
        assert result.distance >= Decimal("0")

    @patch("graph_star.semantic_allocation._load_sentence_transformer")
    def test_model_name_forwarded(
        self,
        mock_load_st: MagicMock,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        mock_model = MagicMock()

        def _encode(labels: list[str], **_kwargs: object) -> np.ndarray:
            n = len(labels)
            return np.eye(n, 8, dtype=np.float32)

        mock_model.encode.side_effect = _encode
        mock_load_st.return_value = mock_model

        run_semantic_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            model_name="custom/model",
            similarity_threshold=0.0,
        )
        mock_load_st.assert_called_with("custom/model")


class TestTopLevelImports:
    """Verify new symbols accessible from package root."""

    def test_run_greedy_pipeline_accessible(self) -> None:
        assert hasattr(graph_star, "run_greedy_pipeline")

    def test_run_annealing_pipeline_accessible(self) -> None:
        assert hasattr(graph_star, "run_annealing_pipeline")

    def test_run_semantic_pipeline_accessible(self) -> None:
        assert hasattr(graph_star, "run_semantic_pipeline")
