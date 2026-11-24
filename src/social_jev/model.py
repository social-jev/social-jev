"""Public typed classification through Transformers or vLLM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer

from .config import SocialJevConfig
from .data import Processor
from .evaluation import Metrics, compute_calibration, create_eval_requests, get_named_metrics
from .integrations.context import active_provider
from .metrics import accuracy
from .types import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    from .data import Sample
    from .integrations.context import RemoteClient
    from .optim import OptimizationResult
    from .training import TrainConfig
    from .types import Query, Request, Result


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
        source = Path(repo)
        if source.is_dir():
            repo = str(source.resolve())
        else:
            checkpoint_config = SocialJevConfig.from_pretrained(
                repo,
                revision=kwargs.get("revision") or "main",
                token=kwargs.get("hf_token") if backend == "vllm" else kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            kwargs["revision"] = getattr(checkpoint_config, "_commit_hash", None) or kwargs.get(
                "revision"
            )
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

    def evaluate(
        self,
        *,
        data,
        queries: Query | list[Query] | None = None,
        metrics: Metrics = accuracy,
        calibration_bins: int = 10,
    ) -> EvaluationResult:
        """Score shared or per-row queries with named metrics on selected decisions."""
        if type(calibration_bins) is not int or calibration_bins < 1:
            message = "calibration_bins must be a positive integer."
            raise ValueError(message)
        named_metrics = get_named_metrics(metrics)
        requests, labels, selected_metrics = create_eval_requests(
            data, queries=queries, metric_names=named_metrics
        )
        predictions = [prediction for results in self.query(requests) for prediction in results]
        submitted_queries = [query for request in requests for query in request.queries]
        probabilities = [
            [prediction.probabilities[option] for option in query.options]
            for prediction, query in zip(predictions, submitted_queries, strict=True)
        ]
        scores = {}
        for name, metric in named_metrics.items():
            # find the submitted queries that requested this metric
            queries_for_metric = [i for i, names in enumerate(selected_metrics) if name in names]
            if queries_for_metric:
                # apply metric to selected queries
                scores[name] = metric(
                    [probabilities[i] for i in queries_for_metric],
                    [labels[i] for i in queries_for_metric],
                )
        return EvaluationResult(
            scores=scores,
            predictions=predictions,
            labels=labels,
            calibration=compute_calibration(probabilities, labels, bins=calibration_bins),
        )

    def optimize(
        self, *, query: Query, trainset, valset=None, metric=None, **kwargs
    ) -> OptimizationResult:
        """Optimize a query and its options against labeled examples using AsyncGEPA.

        Pass GEPA lists or data loaders as trainset and valset. Continuous rollouts
        are enabled by default and use the async vLLM backend. Other optimizer
        arguments, including the sampling strategy and budget, go to AsyncGEPA.
        """
        from async_gepa import optimize  # noqa: PLC0415

        from social_jev.optim import (  # noqa: PLC0415
            AsyncGEPAAdapter,
            GEPAAdapter,
            OptimizationResult,
            accuracy,
            get_proposer_kwargs,
            get_scheduler_kwargs,
        )

        scheduler_kwargs = get_scheduler_kwargs(kwargs)
        adapter_class = (
            AsyncGEPAAdapter if scheduler_kwargs["scheduler"] is not None else GEPAAdapter
        )
        adapter = adapter_class(self, query, accuracy if metric is None else metric)
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
        if self.remote is not None:
            self.remote.close()
        else:
            self.engine.stop()

    @staticmethod
    def train(  # noqa: PLR0913 -- public training inputs and output destinations.
        *,
        repo: str | Path,
        revision: str | None = None,
        trainset: Callable[[], Iterable[Sample]],
        valset: Callable[[], Iterable[Sample]] | None = None,
        config: TrainConfig,
        run_dir: str | Path | None = None,
        final_save_to: str | Path | None = None,
    ) -> None:
        """Train from a checkpoint and save the requested outputs."""
        from .training import Trainer  # noqa: PLC0415 -- optional training dependencies.

        Trainer(
            config,
            repo=repo,
            revision=revision,
            trainset=trainset,
            valset=valset,
            run_dir=run_dir,
        ).run(final_save_to=final_save_to)
