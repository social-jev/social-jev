import asyncio
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from transformers import AutoTokenizer

from social_jev import Query, Request, Result, SocialJev
from social_jev.integrations import Modal


@pytest.mark.parametrize(
    ("settings", "max_inputs"),
    [({}, None), ({"backend": "vllm"}, None), ({"backend": "vllm", "serve": True}, 2)],
)
def test_rented_model_queries_and_releases_compute_on_exception(
    checkpoint, monkeypatch, settings, max_inputs
):
    stop = Mock()
    ready = Mock()
    submit = Mock(return_value=[[Result(choice="B", probabilities={"A": 0.2, "B": 0.8})]])

    async def respond(request):
        await asyncio.sleep(0)
        return [[Result(choice=request.queries[0].options[0], probabilities={"A": 1.0})]]

    submit.aio = AsyncMock(side_effect=respond)
    worker = SimpleNamespace(
        ready=SimpleNamespace(remote=ready),
        run_query=SimpleNamespace(remote=submit),
    )

    @contextmanager
    def run_app():
        try:
            yield
        finally:
            stop()

    sdk = Mock()
    sdk.App.return_value.run.side_effect = run_app
    sdk.App.return_value.cls.return_value = Mock(return_value=Mock(return_value=worker))
    for decorator in ("concurrent", "enter", "method", "exit"):
        getattr(sdk, decorator).return_value = lambda decorated: decorated
    monkeypatch.setitem(sys.modules, "modal", sdk)

    request = Request(state="S", queries=[Query(text="Q", options=["A", "B"])])

    def run():
        with Modal(max_inputs=max_inputs), monkeypatch.context() as local:
            tokenizer = Mock(side_effect=AssertionError("Tokenizer loaded on the caller"))
            local.setattr(AutoTokenizer, "from_pretrained", tokenizer)
            model = SocialJev.from_pretrained(checkpoint, **settings)
            ready.assert_called_once_with()
            assert model.query(request)[0][0].choice == "B"
            if not settings:
                with pytest.raises(RuntimeError, match="aquery requires"):
                    asyncio.run(model.aquery(request))
                submit.aio.assert_not_awaited()
            else:
                check_overlap(model, request, max_inputs)
            submit.assert_called_once_with(request)
            message = "exit the rental"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="exit the rental"):
        run()

    stop.assert_called_once_with()
    local = SocialJev.from_pretrained(checkpoint)
    try:
        assert local.query(request)[0][0].choice in {"A", "B"}
    finally:
        local.stop()


@pytest.mark.parametrize("overrides", [False, True])
def test_modal_forwards_worker_options(monkeypatch, overrides):
    sdk = Mock()
    sdk.App.return_value.run.return_value = nullcontext()
    monkeypatch.setitem(sys.modules, "modal", sdk)
    image = object()
    options = {"image": image, "gpu": "H100:2", "cpu": 8, "max_inputs": 4} if overrides else {}

    with Modal(**options):
        SocialJev.from_pretrained("test/checkpoint", backend="vllm")

    sdk.concurrent.assert_called_once_with(max_inputs=4 if overrides else 1)
    worker_options = sdk.App.return_value.cls.call_args.kwargs
    if overrides:
        sdk.Image.debian_slim.assert_not_called()
        assert worker_options["image"] is image
        assert worker_options["gpu"] == "H100:2"
        assert worker_options["cpu"] == 8
    else:
        sdk.Image.debian_slim.assert_called_once()
        assert worker_options["gpu"] == "H100"


def test_transformers_rejects_multi_gpu_before_creating_modal_app(monkeypatch):
    sdk = Mock()
    monkeypatch.setitem(sys.modules, "modal", sdk)
    with Modal(gpu="H100:2"), pytest.raises(ValueError, match="Transformers on Modal uses one GPU"):
        SocialJev.from_pretrained("test/checkpoint")
    sdk.App.assert_not_called()


def check_overlap(model, request, max_inputs):
    async def overlap():
        return await asyncio.gather(model.aquery(request), model.aquery(request))

    warning = (
        pytest.warns(UserWarning, match="Multiple aquery") if max_inputs is None else nullcontext()
    )
    with warning:
        results = asyncio.run(overlap())
    assert [batch[0][0].choice for batch in results] == ["A", "A"]
