"""Graph-based allocation optimizer for financial reconciliation."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations, count
from random import Random
from typing import Final, NewType

import networkx as nx

__all__ = [
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

TargetToSourceAllocations = NewType("TargetToSourceAllocations", dict[str, list[str]])
SourceToTargetAllocations = NewType("SourceToTargetAllocations", dict[str, str])


VALUE: Final = "value"
ALLOCATED: Final = "allocated"


@dataclass(frozen=True, slots=True)
class AllocationWithContext:
    """Result of an allocation walk.

    Attributes:
        allocations: Mapping from target leaf to list of allocated source
            leaves.
        distance: Total hierarchical distance between target and allocated
            values.
        unallocated_source_leaves: Source leaves not present in any
            allocation.
        unallocated_target_leaves: Target leaves with no allocated source
            nodes.
    """

    allocations: TargetToSourceAllocations
    distance: Decimal
    unallocated_source_leaves: frozenset[str] = frozenset()
    unallocated_target_leaves: frozenset[str] = frozenset()


def create_graph(
    *,
    nodes: dict[str, Decimal],
    edges: dict[str, str],
) -> nx.DiGraph:
    """Build a directed graph from node values and child-to-parent edges.

    The graph represents a tree where each child has exactly one parent.
    Edges point from child to parent (child -> parent), so leaf nodes
    have in-degree 0.

    Args:
        nodes: Mapping of node name to its value.
        edges: Mapping of child node to parent node. Each key may appear
            only once, enforcing a tree structure.

    Returns:
        A directed graph with `value` attributes on each node.

    Raises:
        ValueError: If an edge references a node not present in `nodes`.
    """
    graph = nx.DiGraph()

    for node, value in nodes.items():
        graph.add_node(node, value=value)

    for child_node, parent_node in edges.items():
        missing = [n for n in (child_node, parent_node) if n not in nodes]
        if missing:
            msg = f"edge references unknown node(s): {', '.join(missing)}"
            raise ValueError(msg)
        graph.add_edge(child_node, parent_node)

    return graph


def leaf_nodes(
    *,
    graph: nx.DiGraph,
    skip_zeros: bool,
) -> list[str]:
    """Return leaf nodes of a graph, optionally excluding zero-valued nodes.

    Leaf nodes are defined as nodes with in-degree 0. All nodes must have
    been created via `create_graph` and carry a `value` attribute.

    Args:
        graph: A directed graph built by `create_graph`.
        skip_zeros: When `True`, exclude nodes whose value is zero.

    Returns:
        List of leaf node names.
    """
    return [
        node
        for node, degree in dict(graph.in_degree()).items()
        if degree == 0 and (graph.nodes[node][VALUE] != 0 or not skip_zeros)
    ]


def distance(
    *,
    target_graph: nx.DiGraph,
) -> Decimal:
    """Compute total distance between target and allocated values.

    Distance is summed across **all** nodes (leaves and intermediates).
    This intentionally double-counts leaf mismatches through the hierarchy,
    weighting aggregate-level accuracy more heavily.

    Args:
        target_graph: A target graph with both `value` and `allocated`
            attributes on every node (as returned by
            `evaluate_target_rollups`).

    Returns:
        Non-negative total distance.
    """
    return sum(
        (
            abs(attributes[VALUE] - attributes[ALLOCATED])
            for _, attributes in target_graph.nodes(data=True)
        ),
        start=Decimal("0"),
    )


def invert_to_source_target(
    *,
    allocations: TargetToSourceAllocations,
) -> SourceToTargetAllocations:
    """Invert a target-to-source allocation mapping to source-to-target.

    Args:
        allocations: Mapping from target leaf to list of source leaves.

    Returns:
        Mapping from source leaf to its assigned target leaf.
    """
    inverted: SourceToTargetAllocations = SourceToTargetAllocations({})
    for target_leaf, list_source_leaves in allocations.items():
        for source_leaf in list_source_leaves:
            inverted[source_leaf] = target_leaf

    return inverted


def invert_to_target_source(
    *,
    allocations: SourceToTargetAllocations,
    target_leaves: list[str],
) -> TargetToSourceAllocations:
    """Invert a source-to-target allocation mapping to target-to-source.

    Args:
        allocations: Mapping from source leaf to target leaf.
        target_leaves: All target leaf names; unmatched targets get empty
            lists.

    Returns:
        Mapping from target leaf to list of source leaves.
    """
    reverted: TargetToSourceAllocations = TargetToSourceAllocations(defaultdict(list))
    for source_leaf, target_leaf in allocations.items():
        reverted[target_leaf].append(source_leaf)

    for target_leaf in target_leaves:
        if target_leaf not in reverted:
            reverted[target_leaf] = []
    return reverted


def get_unallocated_source_leaves(
    *,
    source_leaves: list[str],
    allocations: TargetToSourceAllocations,
) -> frozenset[str]:
    """Identify source leaves not present in any allocation.

    Args:
        source_leaves: All source leaf names.
        allocations: Current target-to-source allocation mapping.

    Returns:
        Frozenset of unallocated source leaf names.
    """
    allocated = invert_to_source_target(allocations=allocations)
    return frozenset(leaf for leaf in source_leaves if leaf not in allocated)


def get_unallocated_target_leaves(
    *,
    target_leaves: list[str],
    allocations: TargetToSourceAllocations,
) -> frozenset[str]:
    """Identify target leaves with no allocated source nodes.

    Args:
        target_leaves: All target leaf names.
        allocations: Current target-to-source allocation mapping.

    Returns:
        Frozenset of unallocated target leaf names.
    """
    return frozenset(
        leaf
        for leaf in target_leaves
        if leaf not in allocations or not allocations[leaf]
    )


def evaluate_target_rollups(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    allocations: TargetToSourceAllocations,
) -> nx.DiGraph:
    """Roll up allocated values through the target graph hierarchy.

    Creates a deep copy of the target graph and sets an `allocated`
    attribute on each node. Leaf nodes receive the sum of their allocated
    source values; intermediate nodes receive the sum of their children's
    allocated values.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        allocations: Current target-to-source allocation mapping.

    Returns:
        A copy of the target graph with `allocated` attributes populated.
    """
    _target_graph = deepcopy(target_graph)
    _target_leaves = set(target_leaves)
    for node in _target_graph.nodes():
        _target_graph.nodes[node][ALLOCATED] = Decimal("0")
    dependency_queue = nx.topological_sort(_target_graph)

    for target_node in dependency_queue:
        if target_node in _target_leaves:  # evaluate allocations from source
            _target_graph.nodes[target_node][ALLOCATED] = sum(
                (
                    source_graph.nodes[source_node][VALUE]
                    for source_node in allocations.get(target_node, [])
                ),
                start=Decimal("0"),
            )
        else:  # evaluate rollup
            _target_graph.nodes[target_node][ALLOCATED] = sum(
                (
                    _target_graph.nodes[predecessor][ALLOCATED]
                    for predecessor in _target_graph.predecessors(target_node)
                ),
                start=Decimal("0"),
            )

    return _target_graph


def find_best_allocation_with_ties(
    *,
    allocations_with_context: list[AllocationWithContext],
) -> list[AllocationWithContext]:
    """Select allocations with the minimum distance, including ties.

    Args:
        allocations_with_context: Candidate allocations to evaluate.

    Returns:
        All allocations sharing the minimum distance.

    Raises:
        ValueError: If the input list is empty.
    """
    minimal_distance = min(
        allocations_with_context, key=lambda allocation: allocation.distance
    ).distance
    return [tie for tie in allocations_with_context if tie.distance == minimal_distance]


def exact_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    max_group_size: int | None = 4,
) -> AllocationWithContext:
    """Find allocations where source and target leaf values match exactly.

    First matches individual sources to targets with equal values, then
    searches for groups of sources whose combined value matches a target.

    Note:
        The 1:1 matching phase is greedy: each source is assigned to the
        first target with an equal value. If multiple sources share the
        same value and multiple targets also share that value, the match
        order depends on iteration order.

    Warning:
        The group search generates C(n, k) combinations for each group size
        k up to `max_group_size`. This grows rapidly — for example, 20
        unmatched sources with `max_group_size=10` produces over 180,000
        combinations. Pass `None` with care.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        max_group_size: Maximum number of sources to combine when searching
            for group matches. `None` removes the limit, searching all
            possible group sizes.

    Returns:
        Allocation result with exact matches found.
    """
    _target_leaves = list(target_leaves)
    allocations: TargetToSourceAllocations = TargetToSourceAllocations({})
    for source_leaf in source_leaves:
        for target_leaf in _target_leaves:
            if (
                source_graph.nodes[source_leaf][VALUE]
                == target_graph.nodes[target_leaf][VALUE]
            ):
                allocations[target_leaf] = [source_leaf]
                _target_leaves.remove(target_leaf)
                break

    allocated_sources = invert_to_source_target(allocations=allocations)
    source_leaves_for_potential_group_matches = [
        leaf for leaf in source_leaves if leaf not in allocated_sources
    ]
    potential_group_matches: list[tuple[str, ...]] = []
    n = len(source_leaves_for_potential_group_matches)
    upper_bound = n + 1 if max_group_size is None else min(n + 1, max_group_size + 1)
    for length in range(2, upper_bound):
        potential_group_matches.extend(
            combinations(source_leaves_for_potential_group_matches, length)
        )

    for potential_group_match in potential_group_matches:
        total_source_value = sum(
            source_graph.nodes[leaf][VALUE] for leaf in potential_group_match
        )
        for target_leaf in _target_leaves:
            if total_source_value == target_graph.nodes[target_leaf][VALUE] and not set(
                potential_group_match
            ).intersection(set(allocated_sources)):
                allocations[target_leaf] = list(potential_group_match)
                allocated_sources = invert_to_source_target(allocations=allocations)
                _target_leaves.remove(target_leaf)
                break

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


def greedy_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    starting_allocations: TargetToSourceAllocations | None = None,
) -> AllocationWithContext:
    """Greedily allocate source leaves to target leaves minimizing distance.

    Each source leaf is assigned to the target leaf that produces the
    smallest distance increase, evaluated one at a time.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        starting_allocations: Optional pre-existing allocations to build
            upon.

    Returns:
        Allocation result after greedy assignment of all source leaves.

    Raises:
        TypeError: If all source leaves are already allocated but
            `starting_allocations` is `None`.
    """
    best_proposed_allocations: TargetToSourceAllocations = TargetToSourceAllocations(
        defaultdict(list, starting_allocations or {})
    )

    allocated = invert_to_source_target(allocations=best_proposed_allocations)
    unallocated_source_leaves = [
        leaf for leaf in source_leaves if leaf not in allocated
    ]

    if not unallocated_source_leaves:
        if starting_allocations is None:
            msg = (
                "starting_allocations must not be"
                " None when all source leaves are"
                " allocated"
            )
            raise TypeError(msg)

        return AllocationWithContext(
            allocations=starting_allocations,
            distance=distance(
                target_graph=evaluate_target_rollups(
                    target_graph=target_graph,
                    target_leaves=target_leaves,
                    source_graph=source_graph,
                    allocations=starting_allocations,
                ),
            ),
            unallocated_source_leaves=frozenset(unallocated_source_leaves),
            unallocated_target_leaves=get_unallocated_target_leaves(
                target_leaves=target_leaves,
                allocations=starting_allocations,
            ),
        )

    best_distance: Decimal
    for source_leaf in unallocated_source_leaves:
        potential_allocations: list[AllocationWithContext] = []
        for target_leaf in target_leaves:
            proposed_allocation = deepcopy(best_proposed_allocations)
            proposed_allocation[target_leaf].append(source_leaf)
            proposed_allocation_distance = distance(
                target_graph=evaluate_target_rollups(
                    target_graph=target_graph,
                    target_leaves=target_leaves,
                    source_graph=source_graph,
                    allocations=proposed_allocation,
                ),
            )
            potential_allocations.append(
                AllocationWithContext(
                    allocations=proposed_allocation,
                    distance=proposed_allocation_distance,
                )
            )

        new_best_allocation = find_best_allocation_with_ties(
            allocations_with_context=potential_allocations
        )[0]
        best_proposed_allocations = TargetToSourceAllocations(
            deepcopy(new_best_allocation.allocations)
        )
        best_distance = new_best_allocation.distance

    return AllocationWithContext(
        allocations=best_proposed_allocations,
        distance=best_distance,
        unallocated_target_leaves=get_unallocated_target_leaves(
            target_leaves=target_leaves,
            allocations=best_proposed_allocations,
        ),
        unallocated_source_leaves=get_unallocated_source_leaves(
            source_leaves=source_leaves,
            allocations=best_proposed_allocations,
        ),
    )


def greedy_optimization_walk(
    *,
    previous_best_proposed_allocations: TargetToSourceAllocations,
    previous_best_distance: Decimal,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    exclude_source_leaves: list[str] | None = None,
    max_iterations: int | None = 1000,
) -> AllocationWithContext:
    """Optimize an allocation through move, swap, delete, and add strategies.

    Iteratively tries all single-step modifications to the current allocation
    and accepts the best improvement, stopping when no improvement is found
    or `max_iterations` is reached.

    Args:
        previous_best_proposed_allocations: Starting allocation to optimize.
        previous_best_distance: Distance of the starting allocation.
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        exclude_source_leaves: Source leaves to exclude from optimization.
        max_iterations: Maximum optimization iterations before stopping.
            ``None`` removes the limit, iterating until convergence.

    Returns:
        Optimized allocation result.
    """
    if exclude_source_leaves is None:
        exclude_source_leaves = []
    all_source_leaves_with_explicit_exclusions = [
        leaf
        for leaf in invert_to_source_target(
            allocations=previous_best_proposed_allocations
        )
        if leaf not in exclude_source_leaves
    ]
    if not all_source_leaves_with_explicit_exclusions:  # no leaf to optimize
        return AllocationWithContext(
            allocations=previous_best_proposed_allocations,
            distance=previous_best_distance,
            unallocated_target_leaves=get_unallocated_target_leaves(
                target_leaves=target_leaves,
                allocations=previous_best_proposed_allocations,
            ),
            unallocated_source_leaves=get_unallocated_source_leaves(
                source_leaves=all_source_leaves_with_explicit_exclusions
                + exclude_source_leaves,
                allocations=previous_best_proposed_allocations,
            ),
        )

    iterator = range(max_iterations) if max_iterations is not None else count()
    for _ in iterator:
        cached_inverted = invert_to_source_target(
            allocations=previous_best_proposed_allocations
        )
        attached_source_leaves = [
            leaf for leaf in cached_inverted if leaf not in exclude_source_leaves
        ]
        proposed_allocations: list[AllocationWithContext] = []
        for source_leaf in attached_source_leaves:
            # Move strategy
            for target_leaf in target_leaves:
                inverted_allocations = SourceToTargetAllocations(dict(cached_inverted))
                if inverted_allocations[source_leaf] == target_leaf:
                    continue
                inverted_allocations[source_leaf] = target_leaf
                proposed_allocation = invert_to_target_source(
                    allocations=inverted_allocations,
                    target_leaves=target_leaves,
                )
                proposed_distance = distance(
                    target_graph=evaluate_target_rollups(
                        target_graph=target_graph,
                        target_leaves=target_leaves,
                        source_graph=source_graph,
                        allocations=proposed_allocation,
                    ),
                )
                proposed_allocations.append(
                    AllocationWithContext(
                        allocations=proposed_allocation,
                        distance=proposed_distance,
                    )
                )

            # Single swap strategy
            for inner_source_leaf in attached_source_leaves:
                if source_leaf == inner_source_leaf:
                    continue
                inverted_allocations = SourceToTargetAllocations(dict(cached_inverted))
                swap_1 = inverted_allocations[source_leaf]
                swap_2 = inverted_allocations[inner_source_leaf]
                if swap_1 == swap_2:
                    continue
                inverted_allocations[source_leaf] = swap_2
                inverted_allocations[inner_source_leaf] = swap_1
                proposed_allocation = invert_to_target_source(
                    allocations=inverted_allocations,
                    target_leaves=target_leaves,
                )
                proposed_distance = distance(
                    target_graph=evaluate_target_rollups(
                        target_graph=target_graph,
                        target_leaves=target_leaves,
                        source_graph=source_graph,
                        allocations=proposed_allocation,
                    ),
                )
                proposed_allocations.append(
                    AllocationWithContext(
                        allocations=proposed_allocation,
                        distance=proposed_distance,
                    )
                )

            # Delete strategy
            delete_allocation_inverted = SourceToTargetAllocations(
                dict(cached_inverted)
            )
            del delete_allocation_inverted[source_leaf]
            delete_allocation = invert_to_target_source(
                allocations=delete_allocation_inverted,
                target_leaves=target_leaves,
            )

            proposed_allocations.append(
                AllocationWithContext(
                    allocations=delete_allocation,
                    distance=distance(
                        target_graph=evaluate_target_rollups(
                            target_graph=target_graph,
                            target_leaves=target_leaves,
                            source_graph=source_graph,
                            allocations=delete_allocation,
                        ),
                    ),
                )
            )

        # Add strategy
        unattached_source_leaves = [
            leaf
            for leaf in all_source_leaves_with_explicit_exclusions
            if leaf not in cached_inverted
        ]
        for unattached_source_leaf in unattached_source_leaves:
            for target_leaf in target_leaves:
                add_allocation = deepcopy(previous_best_proposed_allocations)
                add_allocation[target_leaf].append(unattached_source_leaf)
                proposed_allocations.append(
                    AllocationWithContext(
                        allocations=add_allocation,
                        distance=distance(
                            target_graph=evaluate_target_rollups(
                                target_graph=target_graph,
                                target_leaves=target_leaves,
                                source_graph=source_graph,
                                allocations=add_allocation,
                            ),
                        ),
                    )
                )

        new_best_allocation = find_best_allocation_with_ties(
            allocations_with_context=proposed_allocations
        )[0]

        if previous_best_distance <= new_best_allocation.distance:
            return AllocationWithContext(
                allocations=previous_best_proposed_allocations,
                distance=previous_best_distance,
                unallocated_target_leaves=get_unallocated_target_leaves(
                    target_leaves=target_leaves,
                    allocations=previous_best_proposed_allocations,
                ),
                unallocated_source_leaves=get_unallocated_source_leaves(
                    source_leaves=all_source_leaves_with_explicit_exclusions
                    + exclude_source_leaves,
                    allocations=previous_best_proposed_allocations,
                ),
            )

        previous_best_proposed_allocations = TargetToSourceAllocations(
            new_best_allocation.allocations
        )
        previous_best_distance = new_best_allocation.distance

    # max_iterations reached without convergence
    return AllocationWithContext(
        allocations=previous_best_proposed_allocations,
        distance=previous_best_distance,
        unallocated_target_leaves=get_unallocated_target_leaves(
            target_leaves=target_leaves,
            allocations=previous_best_proposed_allocations,
        ),
        unallocated_source_leaves=get_unallocated_source_leaves(
            source_leaves=all_source_leaves_with_explicit_exclusions
            + exclude_source_leaves,
            allocations=previous_best_proposed_allocations,
        ),
    )


def simulated_annealing_walk(
    *,
    starting_allocation: TargetToSourceAllocations,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    temperature: Decimal = Decimal("1"),
    cooling_rate: Decimal = Decimal("0.95"),
    min_temperature: Decimal = Decimal("0.01"),
    iterations_per_temp: int = 100,
    seed: int | None = None,
) -> AllocationWithContext:
    """Refine an allocation using simulated annealing metaheuristic.

    Randomly applies move, swap, delete, and add operations, accepting
    worse solutions with a probability that decreases as the temperature
    cools, allowing escape from local minima.

    Args:
        starting_allocation: Initial allocation to refine.
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        temperature: Initial temperature controlling acceptance probability.
        cooling_rate: Multiplicative factor applied to temperature each
            round.
        min_temperature: Temperature threshold at which annealing stops.
        iterations_per_temp: Number of random operations per temperature
            step.
        seed: Random seed for reproducibility.

    Returns:
        Best allocation found during the annealing process.
    """
    rng = Random(seed)  # noqa: S311
    current_allocation = deepcopy(starting_allocation)
    current_distance = distance(
        target_graph=evaluate_target_rollups(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            allocations=current_allocation,
        ),
    )
    best_allocation = deepcopy(current_allocation)
    best_distance = current_distance

    current_temp = temperature

    while current_temp > min_temperature:
        for _ in range(iterations_per_temp):
            inverted = invert_to_source_target(allocations=current_allocation)
            attached_source_leaves = list(inverted.keys())
            unattached_source_leaves = [
                leaf for leaf in source_leaves if leaf not in attached_source_leaves
            ]
            valid_ops = ["add"] if unattached_source_leaves else []
            if attached_source_leaves:
                valid_ops.append("delete")
                if len(target_leaves) >= 2:  # noqa: PLR2004
                    valid_ops.append("move")
            if len(attached_source_leaves) >= 2:  # noqa: PLR2004
                valid_ops.append("swap")
            if not valid_ops:
                break

            proposed_allocation = None
            operation = rng.choice(valid_ops)

            if operation == "move":
                attached_source_leaf = rng.choice(attached_source_leaves)
                new_target_leaf = rng.choice(
                    [x for x in target_leaves if x != inverted[attached_source_leaf]]
                )
                proposed_allocation = deepcopy(current_allocation)
                old_target_leaf = inverted[attached_source_leaf]
                proposed_allocation[old_target_leaf].remove(attached_source_leaf)
                proposed_allocation[new_target_leaf].append(attached_source_leaf)

            elif operation == "swap":
                source_leaf1 = rng.choice(attached_source_leaves)
                source_leaf2 = rng.choice(
                    [x for x in attached_source_leaves if x != source_leaf1]
                )
                target_leaf1 = inverted[source_leaf1]
                target_leaf2 = inverted[source_leaf2]
                if target_leaf1 == target_leaf2:
                    continue
                proposed_allocation = deepcopy(current_allocation)
                proposed_allocation[target_leaf1].remove(source_leaf1)
                proposed_allocation[target_leaf2].remove(source_leaf2)
                proposed_allocation[target_leaf1].append(source_leaf2)
                proposed_allocation[target_leaf2].append(source_leaf1)

            elif operation == "delete":
                source_leaf = rng.choice(attached_source_leaves)
                target_leaf = inverted[source_leaf]
                proposed_allocation = deepcopy(current_allocation)
                proposed_allocation[target_leaf].remove(source_leaf)

            elif operation == "add":
                source_leaf = rng.choice(unattached_source_leaves)
                target_leaf = rng.choice(target_leaves)
                proposed_allocation = deepcopy(current_allocation)
                proposed_allocation[target_leaf].append(source_leaf)

            if proposed_allocation is None:  # pragma: no cover
                msg = "proposed_allocation must not be None after operation selection"
                raise TypeError(msg)

            proposed_distance = distance(
                target_graph=evaluate_target_rollups(
                    target_graph=target_graph,
                    target_leaves=target_leaves,
                    source_graph=source_graph,
                    allocations=proposed_allocation,
                ),
            )

            delta = proposed_distance - current_distance
            acceptance_probability = (
                (-delta / current_temp).exp() if delta > 0 else Decimal("1")
            )

            if rng.random() < acceptance_probability:
                current_allocation = proposed_allocation
                current_distance = proposed_distance

                if current_distance < best_distance:
                    best_allocation = deepcopy(current_allocation)
                    best_distance = current_distance

        current_temp *= cooling_rate

    return AllocationWithContext(
        allocations=best_allocation,
        distance=best_distance,
        unallocated_target_leaves=get_unallocated_target_leaves(
            target_leaves=target_leaves, allocations=best_allocation
        ),
        unallocated_source_leaves=frozenset(
            leaf
            for leaf in source_leaves
            if leaf not in invert_to_source_target(allocations=best_allocation)
        ),
    )
