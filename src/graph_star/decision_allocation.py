"""Decision-model allocation using typed-decision models such as Clef.

A decision model (e.g. `Cloudflare/clef-flash`) reads a state and a schema of
typed questions and returns a probability for every allowed option of every
question. This module uses it in two ways:

- `decision_walk` asks the model which target leaf each source leaf maps to.
- `decision_exact_walk` keeps exact numerical matching and only consults the
  model to break ties between conflicting exact matches.
"""

import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal
from functools import cache
from itertools import combinations
from typing import Any, Final, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import networkx as nx

from graph_star.allocation import (
    VALUE,
    AllocationWithContext,
    TargetToSourceAllocations,
    _rollup_distance,
    get_unallocated_source_leaves,
    get_unallocated_target_leaves,
)

__all__ = [
    "DEFAULT_DECISION_MODEL",
    "DecisionModel",
    "decision_exact_walk",
    "decision_walk",
    "http_decision_model",
    "load_decision_model",
]

DEFAULT_DECISION_MODEL: Final = "Cloudflare/clef-flash"

_MAPPING_QUESTION_ID: Final = "target"
_TIE_BREAK_QUESTION_ID: Final = "same_concept"


class DecisionModel(Protocol):
    """Callable that answers a Jev/SystemOne decision request.

    Implementations receive the request `state` and `questions` and return
    the `answers` mapping of a SystemOne response, keyed by question ID.
    A `choice` answer carries `choice`, `confidence`, and `probabilities`;
    a `noul` answer carries `noul`, the probability of true.
    """

    def __call__(
        self,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Answer `questions` about `state`."""
        ...


def _systemone_decision_model(
    *,
    model_name: str,
    send: Callable[[dict[str, Any]], dict[str, Any]],
) -> DecisionModel:
    """Wrap a SystemOne request-body handler as a `DecisionModel`."""

    def decide(
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        response = send({"model": model_name, "state": state, "questions": questions})
        return response["answers"]

    return decide


@cache
def _load_clef(model_name: str, device: str) -> tuple[Any, Any, Any]:
    """Download, load and cache a Clef release by name.

    Returns:
        The release's `joint_schema_model` module, the model and its
        processor.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    path = snapshot_download(model_name)
    # Clef ships its model code alongside the weights; import it by path so
    # different releases do not collide on a shared module name.
    module_name = f"_graph_star_clef_{abs(hash((model_name, path)))}"
    spec = importlib.util.spec_from_file_location(
        module_name, f"{path}/joint_schema_model.py"
    )
    if spec is None or spec.loader is None:
        msg = f"{model_name} does not ship joint_schema_model.py"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    model, processor = module.load_release_model(path, device=device)
    return module, model, processor


def load_decision_model(
    *,
    model_name: str = DEFAULT_DECISION_MODEL,
    device: str = "cuda",
) -> DecisionModel:
    """Load a Clef decision model from the Hugging Face Hub.

    The release is downloaded once and its bundled `joint_schema_model.py`
    is executed to build the model, so only load releases you trust.
    Requires a `transformers` version that supports the model's backbone.

    Args:
        model_name: Hugging Face model identifier or local path.
        device: Torch device to load the model on.

    Returns:
        A decision model answering requests in-process.
    """
    module, model, processor = _load_clef(model_name, device)
    return _systemone_decision_model(
        model_name=model_name,
        send=lambda request: module.systemone(model, processor, request),
    )


def http_decision_model(
    *,
    base_url: str,
    model_name: str = DEFAULT_DECISION_MODEL,
    timeout: float = 60.0,
) -> DecisionModel:
    """Build a decision model backed by a Jev/SystemOne HTTP endpoint.

    Works with any server answering `POST /v1/systemone`, e.g. Clef served
    by SGLang.

    Args:
        base_url: Server root, e.g. `http://localhost:30000`.
        model_name: Model identifier sent in each request.
        timeout: Request timeout in seconds.

    Returns:
        A decision model answering requests over HTTP.

    Raises:
        ValueError: If `base_url` is not an http(s) URL.
    """
    if urlparse(base_url).scheme not in {"http", "https"}:
        msg = f"base_url must be an http(s) URL, got {base_url!r}"
        raise ValueError(msg)
    endpoint = f"{base_url.rstrip('/')}/v1/systemone"

    def send(request: dict[str, Any]) -> dict[str, Any]:
        http_request = Request(  # noqa: S310 - scheme validated above
            endpoint,
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(http_request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())

    return _systemone_decision_model(model_name=model_name, send=send)


def _hierarchy(*, graph: nx.DiGraph, leaf: str) -> list[str]:
    """Return the ancestors of `leaf` from its parent up to the root."""
    path: list[str] = []
    node = next(iter(graph.successors(leaf)), None)
    while node is not None:
        path.append(node)
        node = next(iter(graph.successors(node)), None)
    return path


def _describe(*, graph: nx.DiGraph, leaf: str) -> dict[str, Any]:
    """Describe an account by its name and position in the hierarchy."""
    return {"name": leaf, "hierarchy": _hierarchy(graph=graph, leaf=leaf)}


def _finalize(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    allocations: TargetToSourceAllocations,
) -> AllocationWithContext:
    """Fill empty targets and wrap allocations with distance and leftovers."""
    for target_leaf in target_leaves:
        if target_leaf not in allocations:
            allocations[target_leaf] = []

    return AllocationWithContext(
        allocations=allocations,
        distance=_rollup_distance(
            target_graph=target_graph,
            target_leaves=target_leaves,
            source_graph=source_graph,
            allocations=allocations,
        ),
        unallocated_target_leaves=get_unallocated_target_leaves(
            target_leaves=target_leaves, allocations=allocations
        ),
        unallocated_source_leaves=get_unallocated_source_leaves(
            source_leaves=source_leaves, allocations=allocations
        ),
    )


def decision_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    decision_model: DecisionModel,
    confidence_threshold: float = 0.5,
) -> AllocationWithContext:
    """Allocate source leaves to target leaves using a decision model.

    For each source leaf, asks the model a `choice` question whose options
    are the target leaves, describing every account by its name and its
    ancestors in the hierarchy. Sources whose chosen target has a
    confidence below the threshold remain unallocated.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        decision_model: Model answering the mapping questions.
        confidence_threshold: Minimum probability of the chosen target for
            the source to be allocated.

    Returns:
        Allocation result with model-chosen matches.
    """
    allocations: TargetToSourceAllocations = TargetToSourceAllocations(
        defaultdict(list)
    )

    if target_leaves:  # a choice question needs at least one option
        criteria = {
            target_leaf: json.dumps(_describe(graph=target_graph, leaf=target_leaf))
            for target_leaf in target_leaves
        }
        question = {
            "type": "choice",
            "instructions": (
                "The state describes a source account from one financial"
                " report. Which target account from another report does it"
                " map to?"
            ),
            "criteria": criteria,
        }
        for source_leaf in source_leaves:
            answer = decision_model(
                state={
                    "source_account": _describe(graph=source_graph, leaf=source_leaf)
                },
                questions={_MAPPING_QUESTION_ID: question},
            )[_MAPPING_QUESTION_ID]
            if answer["confidence"] >= confidence_threshold:
                allocations[answer["choice"]].append(source_leaf)

    return _finalize(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        allocations=allocations,
    )


def decision_exact_walk(
    *,
    target_graph: nx.DiGraph,
    target_leaves: list[str],
    source_graph: nx.DiGraph,
    source_leaves: list[str],
    decision_model: DecisionModel,
    max_group_size: int | None = 4,
) -> AllocationWithContext:
    """Find exact-value allocations using a decision model to break ties.

    Builds the same unified candidate pool as `mixed_exact_walk`: every
    group of up to `max_group_size` sources whose summed value equals a
    target leaf value. Candidates that share no source or target with any
    other candidate are unambiguous and assigned directly. Only conflicting
    candidates are sent to the model, which scores each with the
    probability that the sources and the target represent the same
    concept; they are then greedily assigned in descending probability.

    Warning:
        The candidate search generates C(n, k) combinations of **all**
        source leaves for each group size k up to `max_group_size`, and
        each conflicting candidate costs one model call. Pass `None` with
        care.

    Args:
        target_graph: The target graph built by `create_graph`.
        target_leaves: Names of the target leaf nodes.
        source_graph: The source graph built by `create_graph`.
        source_leaves: Names of the source leaf nodes.
        decision_model: Model scoring conflicting candidates.
        max_group_size: Maximum number of sources to combine when searching
            for group matches. ``1`` disables groups. ``None`` removes the
            limit.

    Returns:
        Allocation result with exact matches found, preferring the model's
        judgement when values alone are ambiguous.
    """
    targets_by_value: dict[Decimal, list[str]] = defaultdict(list)
    for target_leaf in target_leaves:
        targets_by_value[target_graph.nodes[target_leaf][VALUE]].append(target_leaf)

    n = len(source_leaves)
    upper_bound = n + 1 if max_group_size is None else min(n + 1, max_group_size + 1)

    candidates: list[tuple[tuple[str, ...], str]] = []
    for length in range(1, upper_bound):
        for group in combinations(source_leaves, length):
            total_value = sum(source_graph.nodes[leaf][VALUE] for leaf in group)
            candidates.extend(
                (group, target_leaf)
                for target_leaf in targets_by_value.get(total_value, [])
            )

    # Source and target reports may reuse account names, so count separately.
    candidates_per_source: dict[str, int] = defaultdict(int)
    candidates_per_target: dict[str, int] = defaultdict(int)
    for group, target_leaf in candidates:
        for leaf in group:
            candidates_per_source[leaf] += 1
        candidates_per_target[target_leaf] += 1

    question = {
        "type": "noul",
        "instructions": (
            "The state pairs source accounts from one financial report with a"
            " target account from another report. Together, do the source"
            " accounts represent the same concept as the target account?"
        ),
    }
    scored: list[tuple[float, tuple[str, ...], str]] = []
    for group, target_leaf in candidates:
        if candidates_per_target[target_leaf] == 1 and all(
            candidates_per_source[leaf] == 1 for leaf in group
        ):
            score = 1.0  # unambiguous: nothing competes for these leaves
        else:
            answer = decision_model(
                state={
                    "source_accounts": [
                        _describe(graph=source_graph, leaf=leaf) for leaf in group
                    ],
                    "target_account": _describe(graph=target_graph, leaf=target_leaf),
                },
                questions={_TIE_BREAK_QUESTION_ID: question},
            )[_TIE_BREAK_QUESTION_ID]
            score = float(answer["noul"])
        scored.append((score, group, target_leaf))

    # Stable sort keeps generation order (smaller groups first) among equals.
    scored.sort(key=lambda c: c[0], reverse=True)

    allocations: TargetToSourceAllocations = TargetToSourceAllocations({})
    used_sources: set[str] = set()
    for _score, group, target_leaf in scored:
        if target_leaf in allocations or any(s in used_sources for s in group):
            continue
        allocations[target_leaf] = list(group)
        used_sources.update(group)

    return _finalize(
        target_graph=target_graph,
        target_leaves=target_leaves,
        source_graph=source_graph,
        source_leaves=source_leaves,
        allocations=allocations,
    )
