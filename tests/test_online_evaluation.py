"""Online benchmarks use the standalone evaluation's data and score contract."""

from social_jev.evaluation import OnlineEvaluation
from social_jev.metrics import accuracy
from social_jev.types import Query


def test_online_dataset_factory_preserves_query_labels_and_metric_selection():
    rows = [{"state": "A", "label": [1, [0.75, 0.25]], "metrics": ["accuracy"]}]
    benchmark = OnlineEvaluation(
        data=lambda: iter(rows),
        queries=[
            Query(text="First?", options=["yes", "no"]),
            Query(text="Second?", options=["a", "b"]),
        ],
        metrics={"accuracy": accuracy},
    )
    first = benchmark.prepare()
    second = benchmark.prepare()
    assert first == second
    requests, labels, selected_metrics = first
    assert requests[0].state == "A"
    assert [query.text for query in requests[0].queries] == ["First?", "Second?"]
    assert labels == [1, [0.75, 0.25]]
    assert selected_metrics == [["accuracy"], ["accuracy"]]
