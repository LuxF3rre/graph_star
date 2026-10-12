"""Tests for graph_star.decision_allocation module and decision pipelines."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

import graph_star
from graph_star.allocation import (
    AllocationWithContext,
    create_graph,
    leaf_nodes,
)
from graph_star.allocation_runner import (
    run_decision_annealing_pipeline,
    run_decision_greedy_pipeline,
    run_decision_pipeline,
)
from graph_star.decision_allocation import (
    DEFAULT_DECISION_MODEL,
    decision_exact_walk,
    decision_walk,
    http_decision_model,
    load_decision_model,
)


class FakeDecisionModel:
    """Decision model answering from fixed preferences, recording requests.

    `choices` maps a source leaf to `(target leaf, confidence)` for choice
    questions. `scores` maps `(source leaves, target leaf)` to the noul
    probability for tie-break questions; unknown pairs score `default`.
    """

    def __init__(
        self,
        *,
        choices: dict[str, tuple[str, float]] | None = None,
        scores: dict[tuple[tuple[str, ...], str], float] | None = None,
        default: float = 0.5,
    ) -> None:
        self.choices = choices or {}
        self.scores = scores or {}
        self.default = default
        self.requests: list[tuple[Any, dict[str, dict[str, Any]]]] = []

    def __call__(
        self,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        self.requests.append((state, questions))
        answers: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            if question["type"] == "choice":
                source = state["source_account"]["name"]
                options = list(question["criteria"])
                choice, confidence = self.choices.get(source, (options[0], 1.0))
                answers[question_id] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                }
            else:
                group = tuple(a["name"] for a in state["source_accounts"])
                target = state["target_account"]["name"]
                answers[question_id] = {
                    "type": "noul",
                    "noul": self.scores.get((group, target), self.default),
                }
        return answers


@pytest.fixture
def tied_target_graph() -> nx.DiGraph:
    """Two targets with the same value under different parents."""
    return create_graph(
        nodes={
            "t_root": Decimal("100"),
            "revenue": Decimal("50"),
            "sales": Decimal("50"),
            "costs": Decimal("50"),
            "materials": Decimal("50"),
        },
        edges={
            "revenue": "t_root",
            "sales": "revenue",
            "costs": "t_root",
            "materials": "costs",
        },
    )


@pytest.fixture
def tied_source_graph() -> nx.DiGraph:
    """Two sources with the same value."""
    return create_graph(
        nodes={
            "s_root": Decimal("100"),
            "turnover": Decimal("50"),
            "raw materials": Decimal("50"),
        },
        edges={"turnover": "s_root", "raw materials": "s_root"},
    )


class TestDecisionWalk:
    def test_allocates_model_choice(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        model = FakeDecisionModel(
            choices={"s1": ("t1", 0.9), "s2": ("t1", 0.8), "s3": ("t2", 0.7)}
        )
        result = decision_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=model,
        )
        assert result.allocations == {"t1": ["s1", "s2"], "t2": ["s3"]}
        assert result.distance == Decimal("0")
        assert result.unallocated_source_leaves == frozenset()

    def test_low_confidence_left_unallocated(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        model = FakeDecisionModel(
            choices={"s1": ("t1", 0.9), "s2": ("t1", 0.2), "s3": ("t2", 0.6)}
        )
        result = decision_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=model,
            confidence_threshold=0.6,
        )
        assert result.allocations == {"t1": ["s1"], "t2": ["s3"]}
        assert result.unallocated_source_leaves == frozenset({"s2"})

    def test_request_describes_hierarchy(
        self,
        tied_target_graph: nx.DiGraph,
        tied_source_graph: nx.DiGraph,
    ) -> None:
        target_leaves = leaf_nodes(graph=tied_target_graph, skip_zeros=False)
        model = FakeDecisionModel()
        decision_walk(
            target_graph=tied_target_graph,
            target_leaves=target_leaves,
            source_graph=tied_source_graph,
            source_leaves=["turnover"],
            decision_model=model,
        )
        [(state, questions)] = model.requests
        assert state == {
            "source_account": {"name": "turnover", "hierarchy": ["s_root"]}
        }
        [question] = questions.values()
        assert question["type"] == "choice"
        assert set(question["criteria"]) == set(target_leaves)
        assert json.loads(question["criteria"]["sales"]) == {
            "name": "sales",
            "hierarchy": ["revenue", "t_root"],
        }

    def test_no_targets_skips_model(
        self,
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        target_graph = create_graph(nodes={"t_root": Decimal("0")}, edges={})
        model = FakeDecisionModel()
        result = decision_walk(
            target_graph=target_graph,
            target_leaves=[],
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=model,
        )
        assert model.requests == []
        assert result.unallocated_source_leaves == frozenset(simple_source_leaves)


class TestDecisionExactWalk:
    def test_model_breaks_value_tie(
        self,
        tied_target_graph: nx.DiGraph,
        tied_source_graph: nx.DiGraph,
    ) -> None:
        model = FakeDecisionModel(
            scores={
                (("turnover",), "sales"): 0.95,
                (("raw materials",), "materials"): 0.9,
            },
            default=0.1,
        )
        result = decision_exact_walk(
            target_graph=tied_target_graph,
            target_leaves=["sales", "materials"],
            source_graph=tied_source_graph,
            source_leaves=["turnover", "raw materials"],
            decision_model=model,
            max_group_size=1,
        )
        assert result.allocations == {
            "sales": ["turnover"],
            "materials": ["raw materials"],
        }
        assert len(model.requests) == 4

    def test_tie_break_reverses_iteration_order(
        self,
        tied_target_graph: nx.DiGraph,
        tied_source_graph: nx.DiGraph,
    ) -> None:
        """The model wins over iteration order, which would pair them 1:1."""
        model = FakeDecisionModel(
            scores={
                (("turnover",), "materials"): 0.9,
                (("raw materials",), "sales"): 0.8,
            },
            default=0.1,
        )
        result = decision_exact_walk(
            target_graph=tied_target_graph,
            target_leaves=["sales", "materials"],
            source_graph=tied_source_graph,
            source_leaves=["turnover", "raw materials"],
            decision_model=model,
            max_group_size=1,
        )
        assert result.allocations == {
            "materials": ["turnover"],
            "sales": ["raw materials"],
        }

    def test_unambiguous_matches_skip_model(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        model = FakeDecisionModel()
        result = decision_exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=model,
        )
        assert model.requests == []
        assert result.allocations == {"t1": ["s1", "s2"], "t2": ["s3"]}
        assert result.distance == Decimal("0")

    def test_group_competes_with_single(self) -> None:
        """A group and a single source matching one target go to the model."""
        target_graph = create_graph(
            nodes={"t_root": Decimal("50"), "t": Decimal("50")},
            edges={"t": "t_root"},
        )
        source_graph = create_graph(
            nodes={
                "s_root": Decimal("100"),
                "single": Decimal("50"),
                "half_a": Decimal("25"),
                "half_b": Decimal("25"),
            },
            edges={"single": "s_root", "half_a": "s_root", "half_b": "s_root"},
        )
        model = FakeDecisionModel(
            scores={(("half_a", "half_b"), "t"): 0.9, (("single",), "t"): 0.3}
        )
        result = decision_exact_walk(
            target_graph=target_graph,
            target_leaves=["t"],
            source_graph=source_graph,
            source_leaves=["single", "half_a", "half_b"],
            decision_model=model,
        )
        assert result.allocations == {"t": ["half_a", "half_b"]}
        assert result.unallocated_source_leaves == frozenset({"single"})
        [(state, _)] = [r for r in model.requests if len(r[0]["source_accounts"]) > 1]
        assert state == {
            "source_accounts": [
                {"name": "half_a", "hierarchy": ["s_root"]},
                {"name": "half_b", "hierarchy": ["s_root"]},
            ],
            "target_account": {"name": "t", "hierarchy": ["t_root"]},
        }

    def test_shared_names_across_reports_are_not_conflicts(self) -> None:
        """A source and a target sharing a name do not count as a conflict."""
        target_graph = create_graph(
            nodes={"root": Decimal("10"), "cash": Decimal("10")},
            edges={"cash": "root"},
        )
        source_graph = create_graph(
            nodes={"root": Decimal("10"), "cash": Decimal("10")},
            edges={"cash": "root"},
        )
        model = FakeDecisionModel()
        result = decision_exact_walk(
            target_graph=target_graph,
            target_leaves=["cash"],
            source_graph=source_graph,
            source_leaves=["cash"],
            decision_model=model,
        )
        assert model.requests == []
        assert result.allocations == {"cash": ["cash"]}

    def test_unlimited_group_size(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = decision_exact_walk(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=FakeDecisionModel(),
            max_group_size=None,
        )
        assert result.distance == Decimal("0")


class TestLoadDecisionModel:
    @pytest.fixture
    def release_dir(self, tmp_path: Path) -> Path:
        """A fake Clef release whose systemone echoes the request."""
        (tmp_path / "joint_schema_model.py").write_text(
            "def load_release_model(path, device):\n"
            "    return ('model', path, device), 'processor'\n"
            "\n"
            "def systemone(model, processor, request):\n"
            "    return {\n"
            "        'model': request['model'],\n"
            "        'answers': {'q': {'loaded': model, 'request': request}},\n"
            "    }\n"
        )
        return tmp_path

    def test_loads_release_and_answers(self, release_dir: Path) -> None:
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(release_dir)
        ) as mock_download:
            model = load_decision_model(model_name="org/clef", device="cpu")
            again = load_decision_model(model_name="org/clef", device="cpu")
        mock_download.assert_called_once_with("org/clef")

        questions = {"q": {"type": "noul"}}
        answers = model(state="s", questions=questions)
        assert answers["q"]["loaded"] == ("model", str(release_dir), "cpu")
        assert answers["q"]["request"] == {
            "model": "org/clef",
            "state": "s",
            "questions": questions,
        }
        assert again(state="s", questions=questions) == answers

    def test_missing_model_code_raises(self, tmp_path: Path) -> None:
        with (
            patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)),
            patch("importlib.util.spec_from_file_location", return_value=None),
            pytest.raises(ImportError, match="joint_schema_model"),
        ):
            load_decision_model(model_name="org/empty")


class TestHttpDecisionModel:
    def test_posts_systemone_request(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"answers": {"q": {"type": "noul", "noul": 0.7}}}
        ).encode()
        response.__enter__.return_value = response

        with patch(
            "graph_star.decision_allocation.urlopen", return_value=response
        ) as mock_urlopen:
            model = http_decision_model(base_url="http://localhost:30000/", timeout=5.0)
            answers = model(state="s", questions={"q": {"type": "noul"}})

        assert answers == {"q": {"type": "noul", "noul": 0.7}}
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:30000/v1/systemone"
        assert request.get_method() == "POST"
        assert json.loads(request.data) == {
            "model": DEFAULT_DECISION_MODEL,
            "state": "s",
            "questions": {"q": {"type": "noul"}},
        }
        assert mock_urlopen.call_args.kwargs == {"timeout": 5.0}

    def test_rejects_non_http_url(self) -> None:
        with pytest.raises(ValueError, match="http"):
            http_decision_model(base_url="file:///etc/passwd")


class TestRunDecisionPipeline:
    def test_uses_given_model(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        model = FakeDecisionModel(
            choices={"s1": ("t1", 0.9), "s2": ("t1", 0.4), "s3": ("t2", 0.9)}
        )
        result = run_decision_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=model,
            confidence_threshold=0.5,
        )
        assert result.allocations == {"t1": ["s1"], "t2": ["s3"]}

    @patch("graph_star.allocation_runner.load_decision_model")
    def test_loads_model_by_name(
        self,
        mock_load: MagicMock,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        mock_load.return_value = FakeDecisionModel()
        run_decision_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            model_name="custom/model",
        )
        mock_load.assert_called_once_with(model_name="custom/model")


class TestRunDecisionGreedyPipeline:
    def test_perfect_match_yields_zero_distance(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_decision_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=FakeDecisionModel(),
        )
        assert isinstance(result, AllocationWithContext)
        assert result.distance == Decimal("0")

    def test_keeps_tie_break_choice(
        self,
        tied_target_graph: nx.DiGraph,
        tied_source_graph: nx.DiGraph,
    ) -> None:
        model = FakeDecisionModel(
            scores={
                (("turnover",), "sales"): 0.95,
                (("raw materials",), "materials"): 0.9,
            },
            default=0.1,
        )
        result = run_decision_greedy_pipeline(
            target_graph=tied_target_graph,
            target_leaves=["sales", "materials"],
            source_graph=tied_source_graph,
            source_leaves=["turnover", "raw materials"],
            decision_model=model,
        )
        assert result.allocations["sales"] == ["turnover"]
        assert result.allocations["materials"] == ["raw materials"]

    @patch("graph_star.allocation_runner.load_decision_model")
    def test_loads_model_by_name(
        self,
        mock_load: MagicMock,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        mock_load.return_value = FakeDecisionModel()
        run_decision_greedy_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            model_name="custom/model",
        )
        mock_load.assert_called_once_with(model_name="custom/model")


class TestRunDecisionAnnealingPipeline:
    def test_perfect_match_yields_zero_distance(
        self,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        result = run_decision_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            decision_model=FakeDecisionModel(),
            iterations_per_temp=10,
            seed=42,
        )
        assert isinstance(result, AllocationWithContext)
        assert result.distance == Decimal("0")

    @patch("graph_star.allocation_runner.load_decision_model")
    def test_loads_model_by_name(
        self,
        mock_load: MagicMock,
        simple_target_graph: nx.DiGraph,
        simple_target_leaves: list[str],
        simple_source_graph: nx.DiGraph,
        simple_source_leaves: list[str],
    ) -> None:
        mock_load.return_value = FakeDecisionModel()
        run_decision_annealing_pipeline(
            target_graph=simple_target_graph,
            target_leaves=simple_target_leaves,
            source_graph=simple_source_graph,
            source_leaves=simple_source_leaves,
            model_name="custom/model",
            iterations_per_temp=10,
            seed=42,
        )
        mock_load.assert_called_once_with(model_name="custom/model")


class TestTopLevelImports:
    @pytest.mark.parametrize(
        "name",
        [
            "DecisionModel",
            "decision_exact_walk",
            "decision_walk",
            "http_decision_model",
            "load_decision_model",
            "run_decision_annealing_pipeline",
            "run_decision_greedy_pipeline",
            "run_decision_pipeline",
        ],
    )
    def test_accessible(self, name: str) -> None:
        assert hasattr(graph_star, name)
