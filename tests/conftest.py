from decimal import Decimal

import networkx as nx
import pytest

from graph_star.allocation import (
    TargetToSourceAllocations,
    create_graph,
    leaf_nodes,
)


@pytest.fixture
def simple_target_graph() -> nx.DiGraph:
    """Target tree: t_root(100) <- t1(60), t2(40)."""
    return create_graph(
        nodes={
            "t_root": Decimal("100"),
            "t1": Decimal("60"),
            "t2": Decimal("40"),
        },
        edges={"t1": "t_root", "t2": "t_root"},
    )


@pytest.fixture
def simple_target_leaves(simple_target_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=simple_target_graph, skip_zeros=False)


@pytest.fixture
def simple_source_graph() -> nx.DiGraph:
    """Source tree: s_root(100) <- s1(30), s2(30), s3(40)."""
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
def simple_source_leaves(simple_source_graph: nx.DiGraph) -> list[str]:
    return leaf_nodes(graph=simple_source_graph, skip_zeros=False)


@pytest.fixture
def perfect_allocation() -> TargetToSourceAllocations:
    """Allocation where t1<-[s1,s2]=60, t2<-[s3]=40, total distance=0."""
    return TargetToSourceAllocations(
        {
            "t1": ["s1", "s2"],
            "t2": ["s3"],
        }
    )
