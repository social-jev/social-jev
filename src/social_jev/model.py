"""Public typed classification through Transformers or vLLM."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer

from . import config  # noqa: F401 -- Register the checkpoint configuration before loading.
from .data import Processor
from .evaluation import EvaluationResult
from .metrics import accuracy
from .types import Request

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .optim import OptimizationResult
    from .types import Query, Result


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

    def __init__(self, processor: Processor, engine, *, serve: bool = False):
        self.processor = processor
        self.engine = engine
        self.serve = serve

    @classmethod
    def from_pretrained(cls, repo, *, backend="transformers", serve=False, **kwargs):
        """Load a checkpoint and configure its inference backend.

        Args:
            repo: Hugging Face model repository or local checkpoint directory.
            backend: Inference backend: "transformers" (default) or "vllm".
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
        if backend not in {"transformers", "vllm"} or (serve and backend != "vllm"):
            message = "Choose transformers offline, vllm offline, or vllm with serve=True"
            raise ValueError(message)
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
        encoded = self.processor.encode(requests)
        scores = self.engine.process(encoded.inputs)
        return self.processor.decode(scores, encoded)

    async def aquery(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Submit requests asynchronously with vLLM; results are ordered by document, then query."""
        if not hasattr(self.engine, "aprocess"):
            message = "aquery requires backend='vllm'"
            raise RuntimeError(message)
        encoded = await asyncio.to_thread(self.processor.encode, requests)
        scores = await self.engine.aprocess(encoded.inputs)
        return await asyncio.to_thread(self.processor.decode, scores, encoded)

    def evaluate(self, *, query: Query, examples, metric=accuracy) -> EvaluationResult:
        """Score a batch of labeled examples with probability vectors in option order."""
        examples = list(examples)
        if not examples:
            msg = "Evaluation requires at least one labeled example."
            raise ValueError(msg)

        requests = [
            Request(state=example["state"], queries=[query]) for example in examples
        ]
        predictions = [prediction for (prediction,) in self.query(requests)]
        labels = [example["label"] for example in examples]
        probabilities = [
            [prediction.probabilities[option] for option in query.options]
            for prediction in predictions
        ]
        return EvaluationResult(metric(probabilities, labels), predictions, labels)

    def optimize(
        self, *, query: Query, trainset, valset=None, metric=None, **kwargs
    ) -> OptimizationResult:
        """Optimize a query and its options against labeled examples using AsyncGEPA.

        Pass GEPA lists or data loaders as trainset and valset. Continuous rollouts
        are enabled by default and use the async vLLM backend. Other optimizer
        arguments, including the sampling strategy and budget, go to AsyncGEPA.
        """
        from async_gepa import optimize  # noqa: PLC0415  # ty: ignore[unresolved-import]

        from social_jev.optim import (  # noqa: PLC0415
            GEPAAdapter,
            OptimizationResult,
            accuracy,
            get_proposer_kwargs,
            get_scheduler_kwargs,
        )

        adapter = GEPAAdapter(self, query, accuracy if metric is None else metric)
        scheduler_kwargs = get_scheduler_kwargs(kwargs)
        proposer_kwargs = get_proposer_kwargs(kwargs)
        result = optimize(
            seed_candidate={"query": query.model_dump_json()},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            **scheduler_kwargs,
            **proposer_kwargs,
            **kwargs,
        )
        return OptimizationResult(
            query=adapter.candidate_query(result.candidates[result.best_idx]),
            result=result,
        )

    def stop(self):
        """Release model resources and stop any server launched by this instance."""
        self.engine.stop()
