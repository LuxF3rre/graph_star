"""Tests for graph_star.semantic_allocation module."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

import graph_star
from graph_star.allocation import AllocationWithContext, create_graph, leaf_nodes
from graph_star.semantic_allocation import (
    Embeddings,
    SimilarityMatrix,
    compute_embeddings,
    compute_similarity_matrix,
    semantic_walk,
)

# ---------------------------------------------------------------------------
# Synthetic embeddings with known cosine similarities
# ---------------------------------------------------------------------------


@pytest.fixture
def source_embeddings() -> Embeddings:
    """Source embeddings: 'revenue' ~ [1,0,0,0], 'costs' ~ [0,1,0,0]."""
    arr = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return Embeddings(arr)


@pytest.fixture
def target_embeddings() -> Embeddings:
    """Target embeddings: 'income' ~ [0.9,0.1,0,0], 'expenses' ~ [0.1,0.9,0,0]."""
    raw = np.array(
        [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
        dtype=np.float32,
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return Embeddings((raw / norms).astype(np.float32))


@pytest.fixture
def identity_embeddings() -> Embeddings:
    """Identity matrix as embeddings — each vector is orthogonal."""
    return Embeddings(np.eye(3, dtype=np.float32))


@pytest.fixture
def semantic_target_graph() -> nx.DiGraph:
    return create_graph(
        nodes={
            "root": Decimal("100"),
            "income": Decimal("60"),
            "expenses": Decimal("40"),
        },
        edges={"income": "root", "expenses": "root"},
    )


@pytest.fixture
def semantic_target_leaves(semantic_target_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=semantic_target_graph, skip_zeros=False)


@pytest.fixture
def semantic_source_graph() -> nx.DiGraph:
    return create_graph(
        nodes={
            "root": Decimal("100"),
            "revenue": Decimal("60"),
            "costs": Decimal("40"),
        },
        edges={"revenue": "root", "costs": "root"},
    )


@pytest.fixture
def semantic_source_leaves(semantic_source_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=semantic_source_graph, skip_zeros=False)


# ---------------------------------------------------------------------------
# TestComputeSimilarityMatrix
# ---------------------------------------------------------------------------


class TestComputeSimilarityMatrix:
    def test_shape(
        self,
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        matrix = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        assert matrix.shape == (2, 2)

    def test_identity_self_similarity(
        self,
        identity_embeddings: Embeddings,
    ) -> None:
        matrix = compute_similarity_matrix(
            source_embeddings=identity_embeddings,
            target_embeddings=identity_embeddings,
        )
        np.testing.assert_array_almost_equal(matrix, np.eye(3))

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        a = Embeddings(np.array([[1, 0, 0]], dtype=np.float32))
        b = Embeddings(np.array([[0, 1, 0]], dtype=np.float32))
        matrix = compute_similarity_matrix(
            source_embeddings=a,
            target_embeddings=b,
        )
        assert matrix[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_values_in_valid_range(
        self,
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        matrix = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        assert np.all(matrix >= -1.0 - 1e-6)
        assert np.all(matrix <= 1.0 + 1e-6)

    def test_dimension_mismatch_raises_value_error(self) -> None:
        a = Embeddings(np.ones((2, 3), dtype=np.float32))
        b = Embeddings(np.ones((2, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="embedding dimensions do not match"):
            compute_similarity_matrix(
                source_embeddings=a,
                target_embeddings=b,
            )

    def test_dtype_is_float32(
        self,
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        matrix = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        assert matrix.dtype == np.float32

    def test_high_similarity_for_similar_vectors(
        self,
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        matrix = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        # source[0]=[1,0,0,0] vs target[0]~[0.99,0.11,...] -> high sim
        assert matrix[0, 0] > 0.9
        # source[1]=[0,1,0,0] vs target[1]~[0.11,0.99,...] -> high sim
        assert matrix[1, 1] > 0.9


# ---------------------------------------------------------------------------
# TestComputeEmbeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    def test_empty_labels_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="labels must not be empty"):
            compute_embeddings(labels=[])

    @patch("graph_star.semantic_allocation._load_sentence_transformer")
    def test_shape_and_dtype(self, mock_load_st: MagicMock) -> None:
        fake_embeddings = np.random.default_rng(0).random((3, 128)).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_load_st.return_value = mock_model

        result = compute_embeddings(labels=["a", "b", "c"])
        assert result.shape == (3, 128)
        assert result.dtype == np.float32

    @patch("graph_star.semantic_allocation._load_sentence_transformer")
    def test_model_called_with_correct_args(
        self,
        mock_load_st: MagicMock,
    ) -> None:
        fake_embeddings = np.ones((2, 64), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_load_st.return_value = mock_model

        compute_embeddings(
            labels=["hello", "world"],
            model_name="test-model",
        )

        mock_load_st.assert_called_once_with("test-model")
        mock_model.encode.assert_called_once_with(
            ["hello", "world"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# ---------------------------------------------------------------------------
# TestSemanticWalk
# ---------------------------------------------------------------------------


class TestSemanticWalk:
    def test_allocation_above_threshold(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        sim = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )

        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
            similarity_threshold=0.5,
        )

        assert isinstance(result, AllocationWithContext)
        assert result.distance >= Decimal("0")
        assert len(result.unallocated_source_leaves) == 0

    def test_allocation_below_threshold_leaves_unallocated(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
    ) -> None:
        # Orthogonal embeddings — max similarity is 0
        src_emb = Embeddings(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                dtype=np.float32,
            )
        )
        tgt_emb = Embeddings(
            np.array(
                [[0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        )
        sim = compute_similarity_matrix(
            source_embeddings=src_emb,
            target_embeddings=tgt_emb,
        )

        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
            similarity_threshold=0.5,
        )

        assert len(result.unallocated_source_leaves) == 2

    def test_shape_mismatch_raises_value_error(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
    ) -> None:
        wrong_shape = SimilarityMatrix(np.ones((5, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="similarity_matrix shape"):
            semantic_walk(
                target_graph=semantic_target_graph,
                target_leaves=semantic_target_leaves,
                source_graph=semantic_source_graph,
                source_leaves=semantic_source_leaves,
                similarity_matrix=wrong_shape,
            )

    def test_distance_non_negative(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        sim = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
        )
        assert result.distance >= Decimal("0")

    def test_threshold_zero_allocates_everything(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
    ) -> None:
        sim = compute_similarity_matrix(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
            similarity_threshold=0.0,
        )
        assert len(result.unallocated_source_leaves) == 0

    def test_threshold_one_allocates_nothing_with_imperfect_sim(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
        target_embeddings: Embeddings,
    ) -> None:
        # Source embeddings slightly different from target
        src_emb = Embeddings(
            np.array(
                [[0.8, 0.2, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
                dtype=np.float32,
            )
        )
        sim = compute_similarity_matrix(
            source_embeddings=src_emb,
            target_embeddings=target_embeddings,
        )
        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
            similarity_threshold=1.0,
        )
        assert len(result.unallocated_source_leaves) == 2

    def test_unallocated_target_leaves_populated(
        self,
        semantic_target_graph: nx.DiGraph,
        semantic_target_leaves: list[str],
        semantic_source_graph: nx.DiGraph,
        semantic_source_leaves: list[str],
    ) -> None:
        # Both sources map to same target (column 0 has highest sim)
        sim = SimilarityMatrix(
            np.array(
                [[0.9, 0.1], [0.8, 0.2]],
                dtype=np.float32,
            )
        )
        result = semantic_walk(
            target_graph=semantic_target_graph,
            target_leaves=semantic_target_leaves,
            source_graph=semantic_source_graph,
            source_leaves=semantic_source_leaves,
            similarity_matrix=sim,
            similarity_threshold=0.5,
        )
        assert len(result.unallocated_target_leaves) >= 1


# ---------------------------------------------------------------------------
# TestTopLevelImports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_embeddings_accessible(self) -> None:
        assert hasattr(graph_star, "Embeddings")

    def test_similarity_matrix_accessible(self) -> None:
        assert hasattr(graph_star, "SimilarityMatrix")

    def test_compute_embeddings_accessible(self) -> None:
        assert hasattr(graph_star, "compute_embeddings")

    def test_compute_similarity_matrix_accessible(self) -> None:
        assert hasattr(graph_star, "compute_similarity_matrix")

    def test_semantic_walk_accessible(self) -> None:
        assert hasattr(graph_star, "semantic_walk")

    def test_run_semantic_pipeline_accessible(self) -> None:
        assert hasattr(graph_star, "run_semantic_pipeline")
