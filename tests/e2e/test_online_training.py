"""Benchmark evaluation shares scoring and distributes real samples exactly once."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from social_jev import Query, SocialJev
from social_jev.data import Processor
from social_jev.evaluation import OnlineEvaluation
from social_jev.metrics import accuracy

pytest.importorskip("torchtitan")
pytest.importorskip("triton")

from social_jev.training import CrossEntropy, TrainConfig
from social_jev.training.evaluation import ValAndEvalSchedule
from social_jev.training.trainer import TitanTrainer


class OptionTokenModel(torch.nn.Module):
    """Score option text identically regardless of packing or position."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.1))

    @property
    def device(self):
        return self.scale.device

    def process(self, inputs):
        return [float(tokens[-2] * self.scale) for tokens in inputs]

    def forward(self, batch):
        rows, ends = batch.option_positions.unbind(-1)
        scores = batch.input_ids[rows, ends - 1] * self.scale
        return scores[batch.option_indices.clamp(min=0)]


def query_count(_probabilities, labels):
    """Expose duplicated or dropped queries in the logged metric."""
    return len(labels)


def order_score(probabilities, _labels):
    """Make swapping predictions between different rows change the reported score."""
    return sum(index * row[0] for index, row in enumerate(probabilities, start=1))


def benchmark(count):
    return OnlineEvaluation(
        data=[{"state": "S", "label": [i % 2, [0.2, 0.3, 0.5]]} for i in range(count)],
        queries=[Query(text="Q", options=["A", "B"]), Query(text="R", options=["A", "B", "C"])],
        metrics={"accuracy": accuracy, "queries": query_count},
    )


def schedule(tokenizer, evaluation, metrics, *, valset=None, group=None):
    dimensions = SimpleNamespace(
        dp_enabled=group is not None,
        get_mesh=lambda _name: SimpleNamespace(get_group=lambda: group),
    )
    return ValAndEvalSchedule.Config(
        enable=True,
        source=valset,
        loss_function=CrossEntropy(),
        val_steps=2,
        evaluations={"benchmark": evaluation},
        eval_steps=3,
        total_steps=5,
    ).build(
        dp_world_size=torch.distributed.get_world_size(group) if group is not None else 1,
        dp_rank=torch.distributed.get_rank(group) if group is not None else 0,
        tokenizer=SimpleNamespace(tokenizer=tokenizer),
        parallel_dims=dimensions,
        metrics_processor=metrics,
        seq_len=20,
        local_batch_size=1,
    )


def test_benchmarks_run_at_their_own_interval_and_match_evaluate(checkpoint):
    from social_jev.data import Sample  # noqa: PLC0415

    loaded = SocialJev.from_pretrained(checkpoint)
    tokenizer = loaded.processor.tokenizer
    model = OptionTokenModel()
    evaluation = benchmark(3)
    offline = SocialJev(Processor(tokenizer), model).evaluate(
        data=evaluation.data,
        queries=evaluation.queries,
        metrics=evaluation.metrics,
    )
    requests, _, _ = evaluation.prepare()
    validation = Sample.from_request(requests[0])
    validation.labels = [1, 2]
    metrics = Mock()
    validator = schedule(tokenizer, evaluation, metrics, valset=lambda: [validation])
    assert [step for step in range(6) if validator.should_validate(step)] == [0, 2, 3, 4, 5]
    for step in (0, 2, 3, 4, 5, 5):
        validator.validate([model], step)
    assert [call.args[1] for call in metrics.log_validation.call_args_list] == [0, 2, 4, 5]
    assert [call.args[1] for call in metrics.logger.log.call_args_list] == [0, 3, 5]
    for call in metrics.logger.log.call_args_list:
        assert call.args[0] == {
            "evaluation_metrics/benchmark/accuracy": offline.scores["accuracy"],
            "evaluation_metrics/benchmark/queries": 6,
            "evaluation_metrics/benchmark/calibration/ece": offline.calibration.ece,
        }
    assert model.training
    loaded.stop()


def run_distributed_benchmark(rank, checkpoint, directory, count):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
    )
    try:
        loaded = SocialJev.from_pretrained(checkpoint)
        model = OptionTokenModel()
        metrics = Mock()
        evaluation = OnlineEvaluation(
            data=[
                {"state": "S", "query": "Q", "options": options, "label": index % len(options)}
                for index, options in enumerate([["A", "B"], ["B", "A"], ["C", "A", "B"]][:count])
            ],
            metrics={"accuracy": accuracy, "queries": query_count, "order": order_score},
        )
        expected = SocialJev(loaded.processor, model).evaluate(
            data=evaluation.data,
            queries=evaluation.queries,
            metrics=evaluation.metrics,
        )
        validator = schedule(
            loaded.processor.tokenizer,
            evaluation,
            metrics,
            group=torch.distributed.group.WORLD,
        )
        forwards = []

        def synchronize_forward(_model, _inputs, _output):
            marker = torch.ones(1)
            torch.distributed.all_reduce(marker)
            forwards.append(marker.item())

        model.register_forward_hook(synchronize_forward)
        validator.validate([model], 0)
        if rank == 0:
            values = metrics.logger.log.call_args.args[0]
            assert values["evaluation_metrics/benchmark/accuracy"] == expected.scores["accuracy"]
            assert values["evaluation_metrics/benchmark/queries"] == count
            assert values["evaluation_metrics/benchmark/order"] == expected.scores["order"]
            assert (
                values["evaluation_metrics/benchmark/calibration/ece"] == expected.calibration.ece
            )
        else:
            metrics.logger.log.assert_not_called()
        (directory / f"rank-{rank}.json").write_text(json.dumps(forwards))
        assert model.training
        assert validator.placeholder is None
        loaded.stop()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("count", [1, 3])
def test_online_evaluation_partitions_data_and_excludes_placeholder_scores(
    checkpoint, tmp_path, count
):
    torch.multiprocessing.spawn(
        run_distributed_benchmark, args=(checkpoint, tmp_path, count), nprocs=2
    )
    calls = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert calls == [[2.0] * ((count + 1) // 2)] * 2


def test_training_enables_benchmarks_without_a_validation_set(monkeypatch, tmp_path):
    monkeypatch.setattr(TitanTrainer.Config, "build", lambda self: self)
    config = TrainConfig(steps=4, evaluations={"benchmark": benchmark(1)}, eval_steps=None)
    titan = TitanTrainer.from_checkpoint(tmp_path, config, list, None, "run")
    assert titan.validator.enable
    assert titan.validator.evaluations == config.evaluations
    assert titan.validator.eval_steps is None


def test_online_evaluation_rejects_partial_benchmark_scores(checkpoint):
    loaded = SocialJev.from_pretrained(checkpoint)
    evaluation = OnlineEvaluation(
        data=[{"state": "S " * 100, "label": 0}],
        queries=Query(text="Q", options=["A", "B"]),
    )
    metrics = Mock()
    validator = schedule(loaded.processor.tokenizer, evaluation, metrics)
    model = OptionTokenModel()
    with (
        pytest.warns(UserWarning, match="Skipped a query"),
        pytest.raises(ValueError, match="did not score every query"),
    ):
        validator.validate([model], 0)
    metrics.logger.log.assert_not_called()
    assert model.training
    loaded.stop()
