"""Verify evaluation input expansion and scoring without model downloads."""

import pytest
from datasets import Dataset

from social_jev import Query, Result, SocialJev
from social_jev.evaluation import compute_calibration
from social_jev.metrics import accuracy, f1


class EvaluationModel(SocialJev):
    """Return option probabilities supplied through the document state."""

    def __init__(self):
        self.calls = []

    def query(self, requests):
        self.calls.append(requests)
        return [
            [
                Result(
                    choice=query.options[int(request.state)],
                    # Reverse insertion order to test query-option alignment.
                    probabilities={
                        option: float(i == int(request.state))
                        for i, option in reversed(list(enumerate(query.options)))
                    },
                )
                for query in request.queries
            ]
            for request in requests
        ]


def test_shared_query_reports_named_metrics_and_order():
    """Score an iterable once while retaining prediction and label order."""
    model = EvaluationModel()
    query = Query(text="Relevant?", options=["Yes", "No"])
    data = iter([{"state": "0", "label": 0}, {"state": "0", "label": 1}])
    result = model.evaluate(queries=query, data=data, metrics=[accuracy, f1])
    assert result.scores == {"accuracy": 0.5, "f1": pytest.approx(2 / 3)}
    assert result.labels == [0, 1]
    assert [prediction.choice for prediction in result.predictions] == ["Yes", "Yes"]
    assert len(model.calls) == 1
    assert [request.queries for request in model.calls[0]] == [[query], [query]]


def test_all_shared_queries_apply_to_all_rows_even_when_lengths_match():
    """Two queries and two documents produce four decisions, not two pairs."""
    model = EvaluationModel()
    queries = [
        Query(text="Budget?", options=["Yes", "No"]),
        Query(text="Topic?", options=["Budget", "Audits", "Other"]),
    ]
    result = model.evaluate(
        queries=queries,
        data=[{"state": "0", "label": [0, 1]}, {"state": "1", "label": [1, 2]}],
    )
    assert result.scores == {"accuracy": 0.5}
    assert result.labels == [0, 1, 1, 2]
    assert [prediction.choice for prediction in result.predictions] == [
        "Yes",
        "Budget",
        "No",
        "Audits",
    ]
    assert len(model.calls) == 1
    assert [request.queries for request in model.calls[0]] == [queries, queries]


def test_custom_huggingface_rows_select_metrics():
    """Score each named metric across only its selected rows, not per document."""
    model = EvaluationModel()
    data = Dataset.from_list(
        [
            {
                "state": "0",
                "query": "Relevant?",
                "options": ["Yes", "No"],
                "label": 0,
                "metrics": ["correct", "positive_f1"],
            },
            {
                "state": "0",
                "query": "Uncivil?",
                "options": ["Yes", "No"],
                "label": 1,
                "metrics": ["positive_f1"],
            },
            {
                "state": "2",
                "query": "Topic?",
                "options": ["A", "B", "C"],
                "label": 1,
                "metrics": [],
            },
        ]
    )
    result = model.evaluate(
        data=data, metrics={"correct": accuracy, "positive_f1": f1, "unused": accuracy}
    )
    assert result.scores == {"correct": 1.0, "positive_f1": pytest.approx(2 / 3)}
    assert result.labels == [0, 1, 1]
    assert [prediction.choice for prediction in result.predictions] == ["Yes", "Yes", "C"]
    assert len(model.calls) == 1
    assert [request.queries[0].text for request in model.calls[0]] == [
        "Relevant?",
        "Uncivil?",
        "Topic?",
    ]


def test_custom_rows_without_metric_selections_use_every_metric():
    model = EvaluationModel()
    rows = [
        {"state": "0", "query": "Relevant?", "options": ["Yes", "No"], "label": 0},
        {"state": "1", "query": "Uncivil?", "options": ["Yes", "No"], "label": 0},
    ]
    result = model.evaluate(data=rows, metrics={"correct": accuracy, "positive_f1": f1})
    assert result.scores == {"correct": 0.5, "positive_f1": pytest.approx(2 / 3)}


def test_shared_queries_expand_metric_selections():
    model = EvaluationModel()
    queries = [Query(text=text, options=["Yes", "No"]) for text in ["Budget?", "Audits?"]]
    result = model.evaluate(
        queries=queries,
        data=[
            {"state": "0", "label": [0, 1], "metrics": ["accuracy"]},
            {"state": "1", "label": [1, 1], "metrics": []},
        ],
    )
    assert result.scores == {"accuracy": 0.5}
    assert len(result.predictions) == 4


def distribution_score(probabilities, labels):
    """Keep soft labels intact and use option-ordered probabilities."""
    return sum(
        sum(p * target for p, target in zip(prediction, label, strict=True))
        for prediction, label in zip(probabilities, labels, strict=True)
    ) / len(labels)


@pytest.mark.parametrize("mode", ["single", "multiple", "custom"])
def test_distributional_labels(mode):
    model = EvaluationModel()
    query = Query(text="Relevant?", options=["Yes", "No"])
    row = {"state": "0", "label": [0.8, 0.2]}
    kwargs = {"queries": query}
    if mode == "multiple":
        kwargs["queries"] = [query, Query(text="Uncivil?", options=["Yes", "No"])]
        row["label"] = [[0.8, 0.2], [0.4, 0.6]]
    elif mode == "custom":
        kwargs = {}
        row.update(query=query.text, options=query.options)
    result = model.evaluate(data=[row], metrics=distribution_score, queries=kwargs.get("queries"))
    assert result.scores["distribution_score"] == pytest.approx(0.6 if mode == "multiple" else 0.8)


@pytest.mark.parametrize(
    ("rows", "query", "message"),
    [
        (
            [{"state": "0", "label": 0, "metrics": []}, {"state": "1", "label": 1}],
            Query(text="Q", options=["Y", "N"]),
            "every row",
        ),
        (
            [{"state": "0", "label": 0, "metrics": ["missing"]}],
            Query(text="Q", options=["Y", "N"]),
            "Unknown metrics",
        ),
        (
            [{"state": "0", "label": 0, "query": "Another"}],
            Query(text="Q", options=["Y", "N"]),
            "not both",
        ),
        (
            [{"state": "0", "label": 0, "options": ["A"]}],
            Query(text="Q", options=["Y", "N"]),
            "not both",
        ),
        (
            [{"state": "0", "label": [0]}],
            [Query(text="Q", options=["Y", "N"])] * 2,
            "one label per query",
        ),
        ([{"state": "0", "label": -1}], Query(text="Q", options=["Y", "N"]), "option indices"),
        ([{"state": "0", "label": 2}], Query(text="Q", options=["Y", "N"]), "option indices"),
        ([{"state": "0", "label": [0.1]}], Query(text="Q", options=["Y", "N"]), "one value"),
        ([{"state": "0", "query": "Q", "options": ["Y", "N"], "label": 2}], None, "option indices"),
    ],
)
def test_invalid_setup_fails_before_inference(rows, query, message):
    model = EvaluationModel()
    with pytest.raises(ValueError, match=message):
        model.evaluate(data=rows, queries=query)
    assert model.calls == []


def test_duplicate_callable_names_require_explicit_keys():
    model = EvaluationModel()
    with pytest.raises(ValueError, match="distinct names"):
        model.evaluate(data=[], metrics=[accuracy, accuracy])
    assert model.calls == []


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_invalid_calibration_bins_fail_before_inference(bins):
    """Reject invalid bin counts without submitting model work."""
    model = EvaluationModel()
    with pytest.raises(ValueError, match="calibration_bins"):
        model.evaluate(data=[], calibration_bins=bins)
    assert model.calls == []


def test_evaluation_always_returns_calibration():
    """Default calibration reuses the single inference call and ignores metric selection."""
    model = EvaluationModel()
    result = model.evaluate(
        queries=Query(text="Relevant?", options=["Yes", "No"]),
        data=[{"state": "0", "label": 0}, {"state": "0", "label": 1}],
        metrics=[],
    )
    assert len(model.calls) == 1
    assert result.scores == {}
    assert len(result.calibration.bins) == 10
    assert result.calibration.ece == 0.5
    assert result.calibration.bins[-1] == {
        "lower": 0.9,
        "upper": 1.0,
        "observations": 2,
        "confidence": 1.0,
        "accuracy": 0.5,
    }
    assert result.calibration.bins[0] == {
        "lower": 0.0,
        "upper": 0.1,
        "observations": 0,
        "confidence": None,
        "accuracy": None,
    }


def test_calibration_boundaries_and_weighted_error():
    """Place exact boundaries in the next bin and weight gaps by observations."""
    probabilities = [[0.5, 0.5], [0.1, 0.9], [0.2, 0.8], [0.0, 1.0]]
    labels = [0, 1, 0, 1]
    result = compute_calibration(probabilities, labels, bins=10)
    assert [b["observations"] for b in result.bins] == [0, 0, 0, 0, 0, 1, 0, 0, 1, 2]
    # Gaps: .5 for one observation, .8 for one, .05 for two.
    assert result.ece == pytest.approx(0.35)
    assert result.bins[9]["confidence"] == pytest.approx(0.95)
    one_hot = [[float(i == label) for i in range(2)] for label in labels]
    assert compute_calibration(probabilities, one_hot, bins=10) == result


def test_distributional_calibration_averages_before_absolute_error():
    """Opposite errors within a bin cancel; this is not per-example absolute error."""
    result = compute_calibration([[0.1, 0.9], [0.1, 0.9]], [[0.2, 0.8], [0.0, 1.0]], bins=20)
    assert len(result.bins) == 20
    assert result.ece == pytest.approx(0.0)
    assert result.bins[18]["accuracy"] == pytest.approx(0.9)
    empty = compute_calibration([], [], bins=10)
    assert empty.ece is None
    assert sum(b["observations"] for b in empty.bins) == 0


def test_calibration_rounds_final_ece_only():
    """Round the aggregate while retaining full precision in bin measurements."""
    result = compute_calibration([[0.987654, 0.012346]], [0], bins=10)
    assert result.ece == 0.01
    assert result.bins[9]["confidence"] == 0.987654
