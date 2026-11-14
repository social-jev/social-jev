"""Exercise both optimizer paths with deterministic inference."""

import json
from importlib.util import find_spec

import httpx
import pytest

from social_jev import Query, Result, SocialJev
from social_jev.metrics import f1

class ParityModel(SocialJev):
    """Score the test query without loading a checkpoint."""

    def query(self, requests):
        rows = []
        for request in requests if isinstance(requests, list) else [requests]:
            (query,) = request.queries
            label = int(request.state) % 2
            predicted = label if query.text == "improved" else 1 - label
            choice = query.options[predicted]
            probabilities = {option: float(option == choice) for option in reversed(query.options)}
            rows.append([Result(choice=choice, probabilities=probabilities)])
        return rows

    async def aquery(self, requests):
        return self.query(requests)


def test_evaluation_f1_uses_first_option():
    """Distinguish first-option and second-option F1 on asymmetric predictions."""
    model = ParityModel(processor=None, engine=None)  # ty: ignore[invalid-argument-type]
    query = Query(text="improved", options=["Even", "Odd"])
    data = [
        {"state": "0", "label": 0},
        {"state": "2", "label": 1},
        {"state": "1", "label": 1},
        {"state": "3", "label": 1},
    ]

    assert model.evaluate(query=query, data=data, metric=f1).score == pytest.approx(2 / 3)


@pytest.mark.skipif(
    find_spec("async_gepa") is None,
    reason="Install the optim extra to exercise AsyncGEPA",
)
@pytest.mark.parametrize("continuous", [True, False])
def test_optimization_returns_the_winning_query_and_scores(monkeypatch, continuous):
    """Retry an invalid default rewrite and accept a valid rewrite through either loop."""
    original = Query(text="initial", options=["Even", "Odd"])
    improved = Query(text="improved", options=["Even", "Odd"])
    trainset = [{"state": "2", "label": 0}, {"state": "3", "label": 1}]
    valset = [{"state": "4", "label": 0}, {"state": "5", "label": [0.0, 1.0]}]
    option_counts = []

    def reflect(request):
        body = json.loads(request.content)
        assert request.url.host == "openrouter.ai"
        assert body["model"] == "z-ai/glm-5.3-flash"
        assert "A label of 1 means the second option." in body["messages"][0]["content"]
        options = improved.options[:1] if not option_counts else improved.options
        option_counts.append(len(options))
        payload = json.dumps({"text": improved.text, "options": options})
        message = {"role": "assistant", "content": payload}
        if "tools" in body:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "test",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": payload,
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    def propose(candidate, reflective_dataset, components_to_update):  # noqa: ARG001
        return {"query": improved.model_dump_json()}

    if continuous:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(httpx.Client, "send", lambda _, request, **_kwargs: reflect(request))
        kwargs = {"max_in_flight_proposals": 1}
    else:
        kwargs = {"scheduler": None, "custom_candidate_proposer": propose}

    model = ParityModel(processor=None, engine=None)  # ty: ignore[invalid-argument-type]
    optimization = model.optimize(
        query=original,
        trainset=trainset,
        valset=valset,
        reflection_minibatch_size=2,
        max_metric_calls=8,
        **kwargs,
    )

    assert optimization.query == improved
    assert optimization.result.val_aggregate_scores == [0.0, 1.0]
    assert optimization.result.total_metric_calls == 8
    assert option_counts == ([1, 2] if continuous else [])
