import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import AutoTokenizer

from social_jev import Query, Request, SocialJev
from social_jev.data import FormatStep, Sample, SampleStep, packed_batches


def requests():
    return [
        Request(
            state="S",
            queries=[Query(text="Q", options=["A", "B"]), Query(text="R", options=["C", "A", "B"])],
        ),
        Request(state="S S", queries=[Query(text="R", options=["B", "C"])]),
    ]


def test_checkpoint_classifies_and_preserves_nested_order(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    items = requests()
    actual = model.query(items)
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["model_type"] == "social_jev"
    assert saved_config["backbone_config"]["model_type"] == "qwen3"
    assert saved_config["backbone_config"]["hidden_size"] == 128
    assert "hidden_size" not in saved_config
    assert saved_config["decision_head"]["type"] in {"linear", "mlp"}
    assert [len(group) for group in actual] == [2, 1]
    for item, group in zip(items, actual, strict=True):
        single = model.query(item)
        assert len(single) == 1
        for query, result, one in zip(item.queries, group, single[0], strict=True):
            assert list(result.probabilities) == query.options
            assert result.probabilities == pytest.approx(one.probabilities, abs=1e-6)
            assert sum(result.probabilities.values()) == pytest.approx(1)
            assert result.choice == max(result.probabilities, key=result.probabilities.get)
    assert model.query([]) == []
    reloaded = SocialJev.from_pretrained(checkpoint)
    assert reloaded.query(items) == actual
    model.stop()
    model.stop()
    reloaded.stop()


def test_candidates_are_isolated_and_processor_is_reentrant(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    item = requests()[0]
    combined = model.query(item)[0]
    for query, expected in zip(item.queries, combined, strict=True):
        actual = model.query(Request(state=item.state, queries=[query]))[0][0]
        assert actual.probabilities == pytest.approx(expected.probabilities, abs=1e-6)
    processor = model.processor
    with ThreadPoolExecutor(max_workers=2) as executor:
        encoded = list(executor.map(processor.encode, requests()))
    assert [len(sample.inputs) for sample in encoded] == [5, 2]
    for sample in encoded:
        for ids in sample.inputs:
            text = processor.tokenizer.decode(ids, skip_special_tokens=False)
            assert text.count("state") == 1
            assert text.count("query") == 1
            assert text.count("option") == 1
            assert ids[-1] == processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert processor.decode([], processor.encode([])) == []
    model.stop()


def test_context_limit_and_execution_modes_fail_clearly(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    with pytest.raises(ValueError, match="capacity"):
        model.query(Request(state="S " * 300, queries=requests()[0].queries))
    with pytest.raises(RuntimeError, match="aquery requires"):
        asyncio.run(model.aquery(requests()))
    with pytest.raises(ValueError, match="Choose"):
        SocialJev.from_pretrained(checkpoint, serve=True)
    model.stop()


def test_format_tokenize_split_runs_once_per_document(checkpoint, monkeypatch):
    model = SocialJev.from_pretrained(checkpoint)
    formatted = []
    apply_template = model.processor.tokenizer.apply_chat_template

    def tokenize(messages, **kwargs):
        formatted.append(messages)
        return apply_template(messages, **kwargs)

    monkeypatch.setattr(model.processor.tokenizer, "apply_chat_template", tokenize)
    items = requests()
    try:
        results = model.query(items)
        assert [len(group) for group in results] == [2, 1]
        assert len(formatted) == 2
        assert [message["role"] for message in formatted[0]] == [
            "state",
            "query",
            "option",
            "option",
            "query",
            "option",
            "option",
            "option",
        ]
        assert [message["content"] for message in formatted[0]] == [
            "S",
            "Q",
            "A",
            "B",
            "R",
            "C",
            "A",
            "B",
        ]
    finally:
        model.stop()


def test_transformers_loading_kwargs_reach_model(checkpoint, monkeypatch):
    tokenizer_loads = []
    load_tokenizer = AutoTokenizer.from_pretrained

    def load(repo, **kwargs):
        tokenizer_loads.append(repo)
        return load_tokenizer(repo, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    model = SocialJev.from_pretrained(checkpoint, dtype=torch.float64, local_files_only=True)
    try:
        assert model.engine.model.dtype == torch.float64
        assert len(tokenizer_loads) == 1
        assert model.processor.tokenizer is model.engine.tokenizer
        assert model.engine.model.config.backbone_config.pad_token_id is None
        assert model.engine.pad_id == model.engine.tokenizer.pad_token_id == 0
        assert len(model.query(requests())) == 2
    finally:
        model.stop()


def test_packed_transformers_matches_independent_candidates(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    items = requests()
    encoded = model.processor.encode(items)
    expected = model.processor.decode(model.engine.process(encoded.inputs), encoded)

    batch = packed_batches(
        FormatStep(items), model.processor.tokenizer, seq_len=256, batch_size=1
    ).next()
    assert batch is not None
    with torch.no_grad():
        logits = model.engine.model(batch)
    actual = model.processor.decode(logits[batch.option_mask].tolist(), encoded)
    for packed, split in zip(actual, expected, strict=True):
        for result, reference in zip(packed, split, strict=True):
            assert result.probabilities == pytest.approx(reference.probabilities, abs=1e-6)
    model.stop()


def test_packed_training_targets_follow_queries(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    first, second = requests()

    def source():
        yield Sample(Sample.from_request(first).messages, [0, [0.1, 0.2, 0.7]])
        yield Sample(Sample.from_request(second).messages, [1])

    batches = packed_batches(
        SampleStep(source, 0, 1, infinite=False, shuffle_buffer_size=1),
        model.processor.tokenizer,
        seq_len=128,
        batch_size=2,
    )
    batch = batches.next()
    assert batch is not None
    assert batch.input_ids.shape[1] == 128
    for row, boundaries in zip(batch.input_ids, batch.message_boundaries, strict=True):
        assert (row[boundaries[-1] + 1 :] == model.processor.tokenizer.pad_token_id).all()
    assert batch.targets is not None
    assert batch.targets.flatten().tolist() == pytest.approx([1, 0, 0, 0.1, 0.2, 0.7, 0, 1, 0])
    logits = model.engine.model(batch)
    assert logits.shape == batch.targets.shape
    assert torch.equal(logits.isneginf(), batch.option_indices < 0)
    log_probs = logits.float().log_softmax(-1).masked_fill(batch.option_indices < 0, 0)
    loss = -(batch.targets * log_probs).sum() / len(batch.targets)
    loss.backward()
    assert model.engine.model.decision_head.network[-1].weight.grad.isfinite().all()
    model.stop()


def test_dcp_loading_and_hf_export_preserve_predictions(checkpoint, tmp_path):
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415 -- checkpoint exercise.

    from social_jev.transformers import SocialJevModel  # noqa: PLC0415 -- model export.

    original = SocialJev.from_pretrained(checkpoint, dtype=torch.bfloat16)
    expected = original.query(requests())
    saved = tmp_path / "step-10"
    state = dict(original.engine.model.state_dict())
    state["optimizer"] = {"exp_avg": torch.ones(4)}
    with pytest.warns(UserWarning, match="torch.distributed"):
        dcp.save(state, checkpoint_id=saved, no_dist=True)
    original.engine.model.config.save_pretrained(saved)
    original.processor.tokenizer.save_pretrained(saved)
    with pytest.warns(UserWarning, match="torch.distributed"):
        loaded = SocialJev.from_pretrained(saved)
    assert loaded.engine.model.dtype == torch.bfloat16
    assert loaded.query(requests()) == expected
    assert loaded.processor.tokenizer.chat_template == original.processor.tokenizer.chat_template

    destination = tmp_path / "hf-export"
    loaded.engine.model.save_pretrained(destination, max_shard_size="5GB")
    restored = SocialJevModel.from_pretrained(destination, dtype="auto")
    for name, value in original.engine.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert restored.tokenizer.get_vocab() == original.processor.tokenizer.get_vocab()
    original.stop()
    loaded.stop()
