"""Verify evaluation input expansion and scoring without model downloads."""

import pytest
from datasets import Dataset

from social_jev import Query, Result, SocialJev
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
    result = model.evaluate(data=[row], metrics=distribution_score, **kwargs)
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
