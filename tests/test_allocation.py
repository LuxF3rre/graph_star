"""Tests for graph_star.allocation module."""

from decimal import Decimal

import networkx as nx
import pytest

import graph_star
from graph_star.allocation import (
    AllocationWithContext,
    SourceToTargetAllocations,
    TargetToSourceAllocations,
    create_graph,
    distance,
    evaluate_target_rollups,
    exact_walk,
    find_best_allocation_with_ties,
    get_unallocated_source_leaves,
    get_unallocated_target_leaves,
    greedy_optimization_walk,
    greedy_walk,
    invert_to_source_target,
    invert_to_target_source,
    leaf_nodes,
    simulated_annealing_walk,
)


class TestCreateGraph:
    def test_nodes_have_correct_values(self, simple_target_graph: nx.DiGraph) -> None:
        assert simple_target_graph.nodes["t1"]["value"] == Decimal("60")
        assert simple_target_graph.nodes["t2"]["value"] == Decimal("40")
        assert simple_target_graph.nodes["t_root"]["value"] == Decimal("100")

    def test_edges_point_from_child_to_parent(
        self, simple_target_graph: nx.DiGraph
    ) -> None:
        assert simple_target_graph.has_edge("t1", "t_root")
        assert simple_target_graph.has_edge("t2", "t_root")
        assert not simple_target_graph.has_edge("t_root", "t1")

    def test_single_node_graph(self) -> None:
        graph = create_graph(
            nodes={"only": Decimal("42")},
            edges={},
        )
        assert len(graph.nodes) == 1
        assert graph.nodes["only"]["value"] == Decimal("42")

    def test_node_count(self, simple_target_graph: nx.DiGraph) -> None:
        assert len(simple_target_graph.nodes) == 3

    def test_edge_count(self, simple_target_graph: nx.DiGraph) -> None:
        assert len(simple_target_graph.edges) == 2

    def test_edge_references_unknown_child_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown node"):
            create_graph(
                nodes={"a": Decimal("1")},
                edges={"missing": "a"},
            )

    def test_edge_references_unknown_parent_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown node"):
            create_graph(
                nodes={"a": Decimal("1")},
                edges={"a": "missing"},
            )


class TestLeafNodes:
    def test_returns_leaves_only(self, simple_target_graph: nx.DiGraph) -> None:
        leaves = leaf_nodes(graph=simple_target_graph, skip_zeros=False)
        assert set(leaves) == {"t1", "t2"}

    def test_skip_zeros_excludes_zero_valued_leaves(self) -> None:
        graph = create_graph(
            nodes={
                "root": Decimal("10"),
                "a": Decimal("10"),
                "b": Decimal("0"),
            },
            edges={"a": "root", "b": "root"},
        )
        leaves = leaf_nodes(graph=graph, skip_zeros=True)
        assert leaves == ["a"]

    def test_skip_zeros_false_includes_zero_valued_leaves(self) -> None:
        graph = create_graph(
            nodes={
                "root": Decimal("10"),
                "a": Decimal("10"),
                "b": Decimal("0"),
            },
            edges={"a": "root", "b": "root"},
        )
        leaves = leaf_nodes(graph=graph, skip_zeros=False)
        assert set(leaves) == {"a", "b"}

    def test_single_node_is_leaf(self) -> None:
        graph = create_graph(nodes={"only": Decimal("1")}, edges={})
        leaves = leaf_nodes(graph=graph, skip_zeros=False)
        assert leaves == ["only"]

    def test_root_is_not_leaf(self, simple_target_graph: nx.DiGraph) -> None:
        leaves = leaf_nodes(graph=simple_target_graph, skip_zeros=False)
        assert "t_root" not in leaves


class TestDistance:
    def test_zero_distance_when_perfect_match(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=perfect_allocation,
        )
        assert distance(target_graph=evaluated) == Decimal("0")

    def test_nonzero_distance_when_mismatch(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        allocations = TargetToSourceAllocations(
            {
                "t1": ["s1", "s2", "s3"],
                "t2": [],
            }
        )
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=allocations,
        )
        # t1: |60-100|=40, t2: |40-0|=40, t_root: |100-100|=0
        assert distance(target_graph=evaluated) == Decimal("80")

    def test_empty_allocations_distance(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        allocations = TargetToSourceAllocations({"t1": [], "t2": []})
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=allocations,
        )
        # t1: |60-0|=60, t2: |40-0|=40, t_root: |100-0|=100
        assert distance(target_graph=evaluated) == Decimal("200")


class TestInvertToSourceTarget:
    def test_basic_inversion(self) -> None:
        allocations = TargetToSourceAllocations(
            {
                "t1": ["s1", "s2"],
                "t2": ["s3"],
            }
        )
        inverted = invert_to_source_target(allocations=allocations)
        assert inverted == {"s1": "t1", "s2": "t1", "s3": "t2"}

    def test_empty_allocations(self) -> None:
        allocations = TargetToSourceAllocations({})
        inverted = invert_to_source_target(allocations=allocations)
        assert inverted == {}

    def test_empty_source_lists(self) -> None:
        allocations = TargetToSourceAllocations({"t1": [], "t2": []})
        inverted = invert_to_source_target(allocations=allocations)
        assert inverted == {}


class TestInvertToTargetSource:
    def test_basic_inversion(self) -> None:
        allocations = SourceToTargetAllocations(
            {
                "s1": "t1",
                "s2": "t1",
                "s3": "t2",
            }
        )
        reverted = invert_to_target_source(
            allocations=allocations,
            target_leaves=["t1", "t2"],
        )
        assert set(reverted["t1"]) == {"s1", "s2"}
        assert reverted["t2"] == ["s3"]

    def test_unmatched_targets_get_empty_lists(self) -> None:
        allocations = SourceToTargetAllocations({"s1": "t1"})
        reverted = invert_to_target_source(
            allocations=allocations,
            target_leaves=["t1", "t2", "t3"],
        )
        assert reverted["t2"] == []
        assert reverted["t3"] == []

    def test_empty_allocations(self) -> None:
        reverted = invert_to_target_source(
            allocations=SourceToTargetAllocations({}),
            target_leaves=["t1", "t2"],
        )
        assert reverted["t1"] == []
        assert reverted["t2"] == []


class TestGetUnallocatedSourceLeaves:
    def test_identifies_unallocated_sources(self) -> None:
        allocations = TargetToSourceAllocations({"t1": ["s1"]})
        result = get_unallocated_source_leaves(
            source_leaves=["s1", "s2", "s3"],
            allocations=allocations,
        )
        assert result == frozenset({"s2", "s3"})

    def test_all_allocated_returns_empty(
        self, perfect_allocation: TargetToSourceAllocations
    ) -> None:
        result = get_unallocated_source_leaves(
            source_leaves=["s1", "s2", "s3"],
            allocations=perfect_allocation,
        )
        assert result == frozenset()

    def test_none_allocated(self) -> None:
        result = get_unallocated_source_leaves(
            source_leaves=["s1", "s2"],
            allocations=TargetToSourceAllocations({}),
        )
        assert result == frozenset({"s1", "s2"})


class TestGetUnallocatedTargetLeaves:
    def test_identifies_unallocated_targets(self) -> None:
        allocations = TargetToSourceAllocations(
            {
                "t1": ["s1"],
                "t2": [],
            }
        )
        result = get_unallocated_target_leaves(
            target_leaves=["t1", "t2"],
            allocations=allocations,
        )
        assert result == frozenset({"t2"})

    def test_missing_target_in_allocations(self) -> None:
        allocations = TargetToSourceAllocations({"t1": ["s1"]})
        result = get_unallocated_target_leaves(
            target_leaves=["t1", "t2"],
            allocations=allocations,
        )
        assert result == frozenset({"t2"})

    def test_all_allocated(self, perfect_allocation: TargetToSourceAllocations) -> None:
        result = get_unallocated_target_leaves(
            target_leaves=["t1", "t2"],
            allocations=perfect_allocation,
        )
        assert result == frozenset()


class TestEvaluateTargetRollups:
    def test_leaf_values_come_from_source_allocations(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=perfect_allocation,
        )
        assert evaluated.nodes["t1"]["allocated"] == Decimal("60")
        assert evaluated.nodes["t2"]["allocated"] == Decimal("40")

    def test_root_rolls_up_from_children(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=perfect_allocation,
        )
        assert evaluated.nodes["t_root"]["allocated"] == Decimal("100")

    def test_does_not_mutate_original_graph(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=perfect_allocation,
        )
        assert "allocated" not in simple_target_graph.nodes["t1"]

    def test_unallocated_leaves_get_zero(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        allocations = TargetToSourceAllocations({"t1": ["s1"], "t2": []})
        evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=allocations,
        )
        assert evaluated.nodes["t2"]["allocated"] == Decimal("0")


class TestFindBestAllocationWithTies:
    def test_single_best(self) -> None:
        allocations = [
            AllocationWithContext(
                allocations=TargetToSourceAllocations({}), distance=Decimal("10")
            ),
            AllocationWithContext(
                allocations=TargetToSourceAllocations({}), distance=Decimal("5")
            ),
            AllocationWithContext(
                allocations=TargetToSourceAllocations({}), distance=Decimal("20")
            ),
        ]
        result = find_best_allocation_with_ties(allocations_with_context=allocations)
        assert len(result) == 1
        assert result[0].distance == Decimal("5")

    def test_ties_returns_all(self) -> None:
        allocations = [
            AllocationWithContext(
                allocations=TargetToSourceAllocations({"a": ["1"]}),
                distance=Decimal("5"),
            ),
            AllocationWithContext(
                allocations=TargetToSourceAllocations({"b": ["2"]}),
                distance=Decimal("5"),
            ),
            AllocationWithContext(
                allocations=TargetToSourceAllocations({"c": ["3"]}),
                distance=Decimal("10"),
            ),
        ]
        result = find_best_allocation_with_ties(allocations_with_context=allocations)
        assert len(result) == 2
        assert all(r.distance == Decimal("5") for r in result)


class TestExactWalk:
    def test_single_exact_matches(self) -> None:
        target = create_graph(
            nodes={
                "r": Decimal("70"),
                "t1": Decimal("30"),
                "t2": Decimal("40"),
            },
            edges={"t1": "r", "t2": "r"},
        )
        source = create_graph(
            nodes={
                "r": Decimal("70"),
                "s1": Decimal("30"),
                "s2": Decimal("40"),
            },
            edges={"s1": "r", "s2": "r"},
        )
        result = exact_walk(
            target_graph=target,
            target_leaves=["t1", "t2"],
            source_graph=source,
            source_leaves=["s1", "s2"],
        )
        assert result.distance == Decimal("0")
        assert result.allocations["t1"] == ["s1"]
        assert result.allocations["t2"] == ["s2"]

    def test_group_exact_match(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
        )
        # s3(40)=t2(40), s1(30)+s2(30)=t1(60)
        assert result.distance == Decimal("0")

    def test_max_group_size_none_searches_all(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            max_group_size=None,
        )
        assert result.distance == Decimal("0")

    def test_no_match_returns_nonzero_distance(self) -> None:
        target = create_graph(
            nodes={"r": Decimal("100"), "t1": Decimal("100")},
            edges={"t1": "r"},
        )
        source = create_graph(
            nodes={"r": Decimal("50"), "s1": Decimal("50")},
            edges={"s1": "r"},
        )
        result = exact_walk(
            target_graph=target,
            target_leaves=["t1"],
            source_graph=source,
            source_leaves=["s1"],
        )
        assert result.distance > Decimal("0")
        assert result.unallocated_source_leaves == frozenset({"s1"})

    def test_unallocated_target_leaves_reported(self) -> None:
        target = create_graph(
            nodes={
                "r": Decimal("100"),
                "t1": Decimal("50"),
                "t2": Decimal("50"),
            },
            edges={"t1": "r", "t2": "r"},
        )
        source = create_graph(
            nodes={"r": Decimal("50"), "s1": Decimal("50")},
            edges={"s1": "r"},
        )
        result = exact_walk(
            target_graph=target,
            target_leaves=["t1", "t2"],
            source_graph=source,
            source_leaves=["s1"],
        )
        # s1 matches t1, t2 is unallocated
        assert "t2" in result.unallocated_target_leaves

    def test_max_group_size_one_skips_groups(self) -> None:
        target = create_graph(
            nodes={
                "r": Decimal("100"),
                "t1": Decimal("60"),
                "t2": Decimal("40"),
            },
            edges={"t1": "r", "t2": "r"},
        )
        source = create_graph(
            nodes={
                "r": Decimal("100"),
                "s1": Decimal("30"),
                "s2": Decimal("30"),
                "s3": Decimal("40"),
            },
            edges={"s1": "r", "s2": "r", "s3": "r"},
        )
        result = exact_walk(
            target_graph=target,
            target_leaves=["t1", "t2"],
            source_graph=source,
            source_leaves=["s1", "s2", "s3"],
            max_group_size=1,
        )
        # s3=40=t2 matches, but s1+s2=60=t1 can't be found with size=1
        assert "t1" in result.unallocated_target_leaves
        assert result.unallocated_source_leaves == frozenset({"s1", "s2"})

    def test_group_no_match_iterates_without_matching(self) -> None:
        """Groups are generated but no group sum matches any target."""
        target = create_graph(
            nodes={"r": Decimal("100"), "t1": Decimal("35")},
            edges={"t1": "r"},
        )
        source = create_graph(
            nodes={
                "r": Decimal("30"),
                "s1": Decimal("10"),
                "s2": Decimal("20"),
            },
            edges={"s1": "r", "s2": "r"},
        )
        result = exact_walk(
            target_graph=target,
            target_leaves=["t1"],
            source_graph=source,
            source_leaves=["s1", "s2"],
        )
        # s1+s2=30 != 35, no group match found
        assert result.distance > Decimal("0")
        assert result.unallocated_source_leaves == frozenset({"s1", "s2"})
        assert result.unallocated_target_leaves == frozenset({"t1"})


class TestGreedyWalk:
    def test_produces_valid_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = greedy_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
        )
        assert result.distance >= Decimal("0")
        assert result.unallocated_source_leaves == frozenset()

    def test_with_starting_allocations(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        starting = TargetToSourceAllocations(
            {
                "t1": ["s1"],
                "t2": [],
            }
        )
        result = greedy_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=["s1", "s2", "s3"],
            starting_allocations=starting,
        )
        assert result.distance >= Decimal("0")

    def test_all_allocated_without_starting_raises_type_error(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        with pytest.raises(TypeError, match="starting_allocations must not be"):
            greedy_walk(
                target_graph=simple_target_graph,
                target_leaves=simple_target_leaves,
                source_graph=simple_source_graph,
                source_leaves=[],
            )

    def test_all_allocated_with_starting_returns_result(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        result = greedy_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=["s1", "s2", "s3"],
            starting_allocations=perfect_allocation,
        )
        assert result.allocations == perfect_allocation
        assert result.distance == Decimal("0")


class TestGreedOptimizationWalk:
    def test_does_not_worsen_optimal_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        result = greedy_optimization_walk(
            previous_best_proposed_allocations=perfect_allocation,
            previous_best_distance=Decimal("0"),
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
        )
        assert result.distance == Decimal("0")

    def test_improves_suboptimal_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        suboptimal = TargetToSourceAllocations(
            {
                "t1": ["s1", "s2", "s3"],
                "t2": [],
            }
        )
        suboptimal_evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=suboptimal,
        )
        suboptimal_distance = distance(target_graph=suboptimal_evaluated)

        result = greedy_optimization_walk(
            previous_best_proposed_allocations=suboptimal,
            previous_best_distance=suboptimal_distance,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
        )
        assert result.distance <= suboptimal_distance

    def test_handles_no_optimizable_leaves(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        empty_allocation = TargetToSourceAllocations(
            {
                "t1": [],
                "t2": [],
            }
        )
        result = greedy_optimization_walk(
            previous_best_proposed_allocations=empty_allocation,
            previous_best_distance=Decimal("200"),
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
        )
        assert result.distance == Decimal("200")

    def test_max_iterations_respected(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
    ) -> None:
        suboptimal = TargetToSourceAllocations(
            {
                "t1": ["s1", "s2", "s3"],
                "t2": [],
            }
        )
        suboptimal_evaluated = evaluate_target_rollups(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            allocations=suboptimal,
        )
        suboptimal_distance = distance(target_graph=suboptimal_evaluated)

        # Should not crash with max_iterations=1
        result = greedy_optimization_walk(
            previous_best_proposed_allocations=suboptimal,
            previous_best_distance=suboptimal_distance,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            max_iterations=1,
        )
        assert result.distance <= suboptimal_distance

    def test_exclude_source_leaves(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        result = greedy_optimization_walk(
            previous_best_proposed_allocations=perfect_allocation,
            previous_best_distance=Decimal("0"),
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            exclude_source_leaves=["s1", "s2", "s3"],
        )
        assert result.distance == Decimal("0")

    def test_add_strategy_evaluates_unattached_leaves(self) -> None:
        """Delete improves first, then Add strategy is evaluated next iteration."""
        target = create_graph(
            nodes={
                "r": Decimal("100"),
                "t1": Decimal("50"),
                "t2": Decimal("50"),
            },
            edges={"t1": "r", "t2": "r"},
        )
        source = create_graph(
            nodes={
                "r": Decimal("110"),
                "s1": Decimal("50"),
                "s2": Decimal("50"),
                "s3": Decimal("10"),
            },
            edges={"s1": "r", "s2": "r", "s3": "r"},
        )
        # s3 is extra — deleting it improves distance to 0
        initial = TargetToSourceAllocations(
            {
                "t1": ["s1", "s3"],
                "t2": ["s2"],
            }
        )
        initial_evaluated = evaluate_target_rollups(
            target_graph=target,
            target_leaves=["t1", "t2"],
            source_graph=source,
            allocations=initial,
        )
        initial_distance = distance(target_graph=initial_evaluated)

        result = greedy_optimization_walk(
            previous_best_proposed_allocations=initial,
            previous_best_distance=initial_distance,
            target_graph=target,
            target_leaves=["t1", "t2"],
            source_graph=source,
        )
        assert result.distance < initial_distance


class TestSimulatedAnnealingWalk:
    def test_deterministic_with_seed(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        result1 = simulated_annealing_walk(
            starting_allocation=perfect_allocation,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        result2 = simulated_annealing_walk(
            starting_allocation=perfect_allocation,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        assert result1.distance == result2.distance
        assert result1.allocations == result2.allocations

    def test_produces_valid_result(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        starting = TargetToSourceAllocations(
            {
                "t1": ["s1"],
                "t2": ["s2", "s3"],
            }
        )
        result = simulated_annealing_walk(
            starting_allocation=starting,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=123,
            temperature=0.5,
            cooling_rate=0.9,
            min_temperature=0.1,
            iterations_per_temp=10,
        )
        assert result.distance >= Decimal("0")

    def test_preserves_optimal_allocation(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
        perfect_allocation: TargetToSourceAllocations,
    ) -> None:
        result = simulated_annealing_walk(
            starting_allocation=perfect_allocation,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
        )
        assert result.distance == Decimal("0")

    def test_reports_unallocated_leaves(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        starting = TargetToSourceAllocations(
            {
                "t1": ["s1", "s2", "s3"],
                "t2": [],
            }
        )
        result = simulated_annealing_walk(
            starting_allocation=starting,
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            seed=42,
            temperature=0.1,
            min_temperature=0.05,
            iterations_per_temp=5,
        )
        assert isinstance(result.unallocated_source_leaves, frozenset)
        assert isinstance(result.unallocated_target_leaves, frozenset)

    def test_empty_source_leaves_breaks_early(self) -> None:
        """No sources at all — valid_ops is empty, inner loop breaks."""
        target = create_graph(
            nodes={"r": Decimal("50"), "t1": Decimal("50")},
            edges={"t1": "r"},
        )
        source = create_graph(
            nodes={"r": Decimal("0")},
            edges={},
        )
        result = simulated_annealing_walk(
            starting_allocation=TargetToSourceAllocations({"t1": []}),
            target_graph=target,
            target_leaves=["t1"],
            source_graph=source,
            source_leaves=[],
            seed=42,
        )
        # Distance = |50-0| + |50-0| = 100
        assert result.distance == Decimal("100")

    def test_single_target_single_source(self) -> None:
        """One target, one source — no move or swap possible."""
        target = create_graph(
            nodes={"r": Decimal("50"), "t1": Decimal("50")},
            edges={"t1": "r"},
        )
        source = create_graph(
            nodes={"r": Decimal("30"), "s1": Decimal("30")},
            edges={"s1": "r"},
        )
        result = simulated_annealing_walk(
            starting_allocation=TargetToSourceAllocations({"t1": ["s1"]}),
            target_graph=target,
            target_leaves=["t1"],
            source_graph=source,
            source_leaves=["s1"],
            seed=42,
            temperature=0.1,
            min_temperature=0.05,
            iterations_per_temp=5,
        )
        assert result.distance >= Decimal("0")


class TestAllocationWithContext:
    def test_is_frozen(self) -> None:
        ctx = AllocationWithContext(
            allocations=TargetToSourceAllocations({"t1": ["s1"]}),
            distance=Decimal("0"),
        )
        with pytest.raises(AttributeError):
            ctx.distance = Decimal("1")  # type: ignore[misc]

    def test_defaults(self) -> None:
        ctx = AllocationWithContext(
            allocations=TargetToSourceAllocations({}),
            distance=Decimal("0"),
        )
        assert ctx.unallocated_source_leaves == frozenset()
        assert ctx.unallocated_target_leaves == frozenset()

    def test_with_unallocated_leaves(self) -> None:
        ctx = AllocationWithContext(
            allocations=TargetToSourceAllocations({"t1": ["s1"]}),
            distance=Decimal("10"),
            unallocated_source_leaves=frozenset({"s2"}),
            unallocated_target_leaves=frozenset({"t2"}),
        )
        assert ctx.unallocated_source_leaves == frozenset({"s2"})
        assert ctx.unallocated_target_leaves == frozenset({"t2"})


class TestTopLevelImports:
    """Verify that public API is accessible from the package root."""

    def test_all_public_symbols_accessible(self) -> None:
        expected = [
            "AllocationWithContext",
            "SourceToTargetAllocations",
            "TargetToSourceAllocations",
            "create_graph",
            "distance",
            "evaluate_target_rollups",
            "exact_walk",
            "find_best_allocation_with_ties",
            "get_unallocated_source_leaves",
            "get_unallocated_target_leaves",
            "greedy_optimization_walk",
            "greedy_walk",
            "invert_to_source_target",
            "invert_to_target_source",
            "leaf_nodes",
            "simulated_annealing_walk",
        ]
        for symbol in expected:
            assert hasattr(graph_star, symbol), f"missing: {symbol}"
