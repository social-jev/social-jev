"""Public typed classification through Transformers or vLLM."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer

from . import config  # noqa: F401 -- Register the checkpoint configuration before loading.
from .data import Processor
from .integrations.context import active_provider

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .integrations.context import RemoteClient
    from .types import Request, Result


class SocialJev:
    """Load a SocialJev model for synchronous or asynchronous inference.

    Args:
        processor: Checkpoint request preparation and result decoding.
        engine: Selected execution engine.
        serve: Whether to launch a vLLM HTTP server.

    Examples:
        >>> model = SocialJev.from_pretrained(repo)  # doctest: +SKIP
        >>> results = model.query([request])  # doctest: +SKIP
        >>> results[0][0].choice  # doctest: +SKIP
        'Transport'
        >>> model.stop()  # doctest: +SKIP
    """

    def __init__(
        self,
        processor: Processor | None,
        engine,
        *,
        serve: bool = False,
        remote: RemoteClient | None = None,
    ):
        self.local_processor = processor
        self.engine = engine
        self.serve = serve
        self.remote = remote

    @property
    def processor(self) -> Processor:
        """Access request processing on the machine that loaded the model."""
        if self.local_processor is None:
            message = "Request processing runs on the remote worker"
            raise AttributeError(message)
        return self.local_processor

    @classmethod
    def from_pretrained(cls, repo, *, backend=None, serve=False, **kwargs):
        """Load a checkpoint and configure its inference backend.

        Args:
            repo: Hugging Face model repository or local checkpoint directory.
            backend: Inference backend: "transformers" or "vllm". Defaults to
                Transformers.
            serve: Launch a managed vLLM HTTP server. Requires backend="vllm".
                Otherwise, inference runs through the local backend.
            **kwargs: Arguments forwarded to the selected backend's model loader.
                Transformers accepts options such as device_map and dtype.
                vLLM accepts options such as tensor_parallel_size,
                gpu_memory_utilization, and max_model_len.

        Returns:
            SocialJev: Loaded model with synchronous query support.
                Both local and server-based vLLM also support asynchronous queries.

        Examples:
            Load with Transformers:

            >>> model = SocialJev.from_pretrained(repo)  # doctest: +SKIP

            Load with local vLLM:

            >>> model = SocialJev.from_pretrained(  # doctest: +SKIP
            ...     repo, backend="vllm", gpu_memory_utilization=0.6
            ... )

            Launch a vLLM server using two GPUs:

            >>> model = SocialJev.from_pretrained(  # doctest: +SKIP
            ...     repo, backend="vllm", serve=True, tensor_parallel_size=2
            ... )
        """
        backend = "transformers" if backend is None else backend
        if backend not in {"transformers", "vllm"} or (serve and backend != "vllm"):
            message = "Choose transformers offline, vllm offline, or vllm with serve=True"
            raise ValueError(message)
        provider = active_provider.get()
        if provider is not None:
            remote = provider.start(str(repo), backend=backend, serve=serve, **kwargs)
            return cls(None, None, serve=serve, remote=remote)
        if backend == "transformers":
            from .transformers import TransformersEngine  # noqa: PLC0415

            engine = TransformersEngine(repo, **kwargs)
            tokenizer = engine.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=kwargs.get("revision"))
            if serve:
                from .vllm.server import Server  # noqa: PLC0415

                engine = Server(repo, **kwargs)
            else:
                from .vllm.offline import VllmEngine  # noqa: PLC0415

                engine = VllmEngine(repo, **kwargs)
        processor = Processor(cast("PreTrainedTokenizerBase", tokenizer))
        return cls(processor, engine, serve=serve)

    def query(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Submit requests synchronously; results are ordered by document, then query."""
        if self.remote is not None:
            return self.remote.submit(requests)
        encoded = self.processor.encode(requests)
        scores = self.engine.process(encoded.inputs)
        return self.processor.decode(scores, encoded)

    async def aquery(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Submit requests asynchronously with vLLM; results are ordered by document, then query."""
        if self.remote is not None:
            return await self.remote.asubmit(requests)
        if not hasattr(self.engine, "aprocess"):
            message = "aquery requires backend='vllm'"
            raise RuntimeError(message)
        encoded = await asyncio.to_thread(self.processor.encode, requests)
        scores = await self.engine.aprocess(encoded.inputs)
        return await asyncio.to_thread(self.processor.decode, scores, encoded)

    def stop(self):
        """Release model resources and stop any server launched by this instance."""
        if self.remote is not None:
            self.remote.close()
        else:
            self.engine.stop()
