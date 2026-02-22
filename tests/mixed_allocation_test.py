"""Tests for graph_star.mixed_allocation module."""

from decimal import Decimal

import networkx as nx
import numpy as np

from graph_star.allocation import (
    AllocationWithContext,
    create_graph,
    leaf_nodes,
)
from graph_star.mixed_allocation import mixed_exact_walk
from graph_star.semantic_allocation import Embeddings


def _make_embeddings(vectors: list[list[float]]) -> Embeddings:
    """Build a synthetic Embeddings array from raw vectors."""
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return Embeddings((arr / norms).astype(np.float32))


class TestMixedExactWalkPhaseOne:
    """Phase 1: 1:1 matching with semantic preference."""

    def test_prefers_semantically_similar_target_among_value_equals(self) -> None:
        """Two targets with same value; source picks the semantically closer one."""
        target_graph = create_graph(
            nodes={
                "t_root": Decimal("100"),
                "t_close": Decimal("50"),
                "t_far": Decimal("50"),
            },
            edges={"t_close": "t_root", "t_far": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        source_graph = create_graph(
            nodes={
                "s_root": Decimal("50"),
                "s1": Decimal("50"),
            },
            edges={"s1": "s_root"},
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        # s1 embedding is very close to t_close, far from t_far
        source_emb = _make_embeddings([[1.0, 0.0, 0.0]])
        # t_close embedding similar to source, t_far orthogonal
        t_close_idx = target_leaves.index("t_close")
        t_far_idx = target_leaves.index("t_far")
        vecs: list[list[float]] = [[0.0, 0.0, 0.0]] * len(target_leaves)
        vecs[t_close_idx] = [0.9, 0.1, 0.0]
        vecs[t_far_idx] = [0.0, 0.0, 1.0]
        target_emb = _make_embeddings(vecs)

        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        assert result.allocations["t_close"] == ["s1"]
        assert result.allocations["t_far"] == []

    def test_unique_values_behaves_like_exact_walk(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        """When no value ambiguity exists, degenerates to exact_walk."""
        n_src = len(simple_source_leaves)
        n_tgt = len(simple_target_leaves)
        source_emb = _make_embeddings(np.eye(n_src, 8, dtype=np.float32).tolist())
        target_emb = _make_embeddings(np.eye(n_tgt, 8, dtype=np.float32).tolist())

        result = mixed_exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        # s3=40 should match t2=40 (unique value match)
        assert "s3" in result.allocations["t2"]

    def test_no_exact_matches_returns_all_unallocated(self) -> None:
        """No value overlaps → empty allocations, all leaves unallocated."""
        target_graph = create_graph(
            nodes={"t_root": Decimal("10"), "t1": Decimal("10")},
            edges={"t1": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        source_graph = create_graph(
            nodes={"s_root": Decimal("99"), "s1": Decimal("99")},
            edges={"s1": "s_root"},
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        source_emb = _make_embeddings([[1.0, 0.0]])
        target_emb = _make_embeddings([[1.0, 0.0]])

        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        assert result.allocations["t1"] == []
        assert "s1" in result.unallocated_source_leaves
        assert "t1" in result.unallocated_target_leaves


class TestMixedExactWalkPhaseTwo:
    """Phase 2: group matching with semantic preference."""

    def test_group_match_prefers_semantically_similar_group(self) -> None:
        """Among groups with matching sum, picks closest to target embedding."""
        # Target: t1=70, needing a group match
        target_graph = create_graph(
            nodes={"t_root": Decimal("70"), "t1": Decimal("70")},
            edges={"t1": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        # Source: s1=30, s2=40, s3=30, s4=40
        # Two valid groups: (s1,s2)=70 and (s3,s4)=70
        source_graph = create_graph(
            nodes={
                "s_root": Decimal("140"),
                "s1": Decimal("30"),
                "s2": Decimal("40"),
                "s3": Decimal("30"),
                "s4": Decimal("40"),
            },
            edges={
                "s1": "s_root",
                "s2": "s_root",
                "s3": "s_root",
                "s4": "s_root",
            },
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        # Make s1,s2 similar to t1; s3,s4 orthogonal
        s_idx = {leaf: i for i, leaf in enumerate(source_leaves)}
        t_idx = {leaf: i for i, leaf in enumerate(target_leaves)}
        s_vecs: list[list[float]] = [[0.0, 0.0, 0.0]] * len(source_leaves)
        s_vecs[s_idx["s1"]] = [0.8, 0.2, 0.0]
        s_vecs[s_idx["s2"]] = [0.9, 0.1, 0.0]
        s_vecs[s_idx["s3"]] = [0.0, 0.0, 1.0]
        s_vecs[s_idx["s4"]] = [0.0, 0.1, 0.9]
        source_emb = _make_embeddings(s_vecs)

        t_vecs: list[list[float]] = [[0.0, 0.0, 0.0]] * len(target_leaves)
        t_vecs[t_idx["t1"]] = [1.0, 0.0, 0.0]
        target_emb = _make_embeddings(t_vecs)

        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        allocated = set(result.allocations["t1"])
        assert allocated == {"s1", "s2"}

    def test_zero_norm_group_average_handled(self) -> None:
        """Cancelling embeddings don't crash; they get similarity = -1."""
        target_graph = create_graph(
            nodes={"t_root": Decimal("20"), "t1": Decimal("20")},
            edges={"t1": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        source_graph = create_graph(
            nodes={
                "s_root": Decimal("20"),
                "s1": Decimal("10"),
                "s2": Decimal("10"),
            },
            edges={"s1": "s_root", "s2": "s_root"},
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        s_idx = {leaf: i for i, leaf in enumerate(source_leaves)}
        # Opposing vectors that cancel to zero
        s_vecs: list[list[float]] = [[0.0, 0.0]] * len(source_leaves)
        s_vecs[s_idx["s1"]] = [1.0, 0.0]
        s_vecs[s_idx["s2"]] = [-1.0, 0.0]
        source_emb = _make_embeddings(s_vecs)

        target_emb = _make_embeddings([[0.0, 1.0]])

        # Should not crash
        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        assert isinstance(result, AllocationWithContext)


class TestMixedExactWalkGroupSizeControls:
    """max_group_size parameter behavior."""

    def test_max_group_size_none_searches_all(self) -> None:
        """Unbounded group search works."""
        target_graph = create_graph(
            nodes={"t_root": Decimal("60"), "t1": Decimal("60")},
            edges={"t1": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        source_graph = create_graph(
            nodes={
                "s_root": Decimal("60"),
                "s1": Decimal("20"),
                "s2": Decimal("20"),
                "s3": Decimal("20"),
            },
            edges={"s1": "s_root", "s2": "s_root", "s3": "s_root"},
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        n_src = len(source_leaves)
        source_emb = _make_embeddings(np.eye(n_src, 4, dtype=np.float32).tolist())
        target_emb = _make_embeddings([[1.0, 0.0, 0.0, 0.0]])

        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
            max_group_size=None,
        )

        assert set(result.allocations["t1"]) == {"s1", "s2", "s3"}

    def test_max_group_size_one_skips_groups(self) -> None:
        """Only 1:1 matching when max_group_size=1."""
        target_graph = create_graph(
            nodes={"t_root": Decimal("60"), "t1": Decimal("60")},
            edges={"t1": "t_root"},
        )
        target_leaves = leaf_nodes(graph=target_graph, skip_zeros=False)

        source_graph = create_graph(
            nodes={
                "s_root": Decimal("60"),
                "s1": Decimal("20"),
                "s2": Decimal("20"),
                "s3": Decimal("20"),
            },
            edges={"s1": "s_root", "s2": "s_root", "s3": "s_root"},
        )
        source_leaves = leaf_nodes(graph=source_graph, skip_zeros=False)

        n_src = len(source_leaves)
        source_emb = _make_embeddings(np.eye(n_src, 4, dtype=np.float32).tolist())
        target_emb = _make_embeddings([[1.0, 0.0, 0.0, 0.0]])

        result = mixed_exact_walk(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            source_leaves=source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
            max_group_size=1,
        )

        # No 1:1 value match (20 != 60), and groups are disabled
        assert result.allocations["t1"] == []


class TestMixedExactWalkReturnTypes:
    """Verify return structure."""

    def test_returns_valid_allocation_with_context(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        n_src = len(simple_source_leaves)
        n_tgt = len(simple_target_leaves)
        source_emb = _make_embeddings(np.eye(n_src, 8, dtype=np.float32).tolist())
        target_emb = _make_embeddings(np.eye(n_tgt, 8, dtype=np.float32).tolist())

        result = mixed_exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        assert isinstance(result, AllocationWithContext)
        assert result.distance >= Decimal("0")

    def test_unallocated_sets_are_correct_types(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        n_src = len(simple_source_leaves)
        n_tgt = len(simple_target_leaves)
        source_emb = _make_embeddings(np.eye(n_src, 8, dtype=np.float32).tolist())
        target_emb = _make_embeddings(np.eye(n_tgt, 8, dtype=np.float32).tolist())

        result = mixed_exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
        )

        assert isinstance(result.unallocated_source_leaves, frozenset)
        assert isinstance(result.unallocated_target_leaves, frozenset)
