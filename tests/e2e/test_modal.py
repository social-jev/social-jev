import asyncio
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from social_jev import Query, Request, SocialJev
from social_jev.integrations import Modal


def test_rented_model_queries_and_releases_compute_on_exception(checkpoint, monkeypatch):
    stop = Mock()
    ready = Mock()
    submit = Mock(return_value=[0.0, 1.0])
    submit.aio = AsyncMock(return_value=[1.0, 0.0])
    worker = SimpleNamespace(
        ready=SimpleNamespace(remote=ready),
        score=SimpleNamespace(remote=submit),
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
        with Modal():
            model = SocialJev.from_pretrained(checkpoint, max_model_len=128)
            ready.assert_called_once_with()
            assert model.query(request)[0][0].choice == "B"
            assert asyncio.run(model.aquery(request))[0][0].choice == "A"
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
