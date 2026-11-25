from collections import Counter
from itertools import islice

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from social_jev.attention import AttentionMask, Positions
from social_jev.data import (
    CollateStep,
    PackAndPadStep,
    Sample,
    SampleStep,
    SplitStep,
    Step,
    TokenizedSample,
)


class Source(Step):
    def __init__(self, samples):
        self.samples = iter(samples)
        self.reads = 0

    def next(self):
        sample = next(self.samples, None)
        if sample is not None:
            self.reads += 1
        return sample


def sample(index):
    return Sample(
        [
            {"role": "state", "content": str(index)},
            {"role": "query", "content": "Q"},
            {"role": "option", "content": "A"},
            {"role": "option", "content": "B"},
        ],
        labels=[index % 2],
    )


def tokenized(lengths, roles, labels=None):
    boundaries = []
    for length in lengths:
        boundaries.append((boundaries[-1] + 1 if boundaries else 0) + length - 1)
    count = roles.count("query")
    return TokenizedSample(
        ids=list(range(sum(lengths))),
        message_boundaries=boundaries,
        roles=roles,
        labels=labels,
        metadata=[{"task": str(index)} for index in range(count)],
    )


def test_shuffle_is_lazy_disjoint_and_restarts():
    reads = []

    def source():
        for index in range(20):
            reads.append(index)
            yield sample(index)

    first = SampleStep(source, 0, 1, shuffle_buffer_size=4)
    assert reads == [0, 1, 2, 3]
    first.next()
    assert len(reads) == 5

    common = SampleStep(source, 0, 1, shuffle_buffer_size=4)
    ordered = [item.messages[0]["content"] for item in islice(iter(common.next, None), 40)]
    ranks = [SampleStep(source, rank, 2, shuffle_buffer_size=4) for rank in range(2)]
    for rank, step in enumerate(ranks):
        selected = [item.messages[0]["content"] for item in islice(iter(step.next, None), 20)]
        assert selected == ordered[rank::2]
    assert Counter(ordered[:20]) == Counter(map(str, range(20)))
    assert Counter(ordered[20:]) == Counter(ordered[:20])
    assert ordered[:20] != ordered[20:]


def test_finite_source_yields_each_sample_once():
    step = SampleStep(lambda: (sample(i) for i in range(7)), 0, 1, infinite=False)
    output = list(iter(step.next, None))
    assert Counter(item.messages[0]["content"] for item in output) == Counter(map(str, range(7)))
    assert step.next() is None


def test_query_split_preserves_tokens_labels_and_metadata():
    original = tokenized(
        [2, 1, 1, 1, 1, 1, 1],
        ["state", "query", "option", "option", "query", "option", "option"],
        [0, [0.25, 0.75]],
    )
    assert list(original.split(on="query", maximum=8)) == [original]
    split = SplitStep(Source([original]), split_on="query", seq_len=5)
    rows = list(iter(split.next, None))
    assert [row.ids for row in rows] == [[0, 1, 2, 3, 4], [0, 1, 5, 6, 7]]
    assert [row.labels for row in rows] == [[0], [[0.25, 0.75]]]
    assert [row.metadata for row in rows] == [[{"task": "0"}], [{"task": "1"}]]
    assert all(row.message_boundaries == [1, 2, 3, 4] for row in rows)


def test_option_split_keeps_query_and_option_order():
    original = tokenized(
        [2, 1, 1, 1, 1, 1],
        ["state", "query", "option", "option", "query", "option"],
    )
    split = SplitStep(Source([original]))
    rows = list(iter(split.next, None))
    assert [row.ids for row in rows] == [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 5, 6]]
    assert all(row.roles == ["state", "query", "option"] for row in rows)


def test_overlength_queries_warn_once_and_preserve_following_queries():
    original = tokenized(
        [2, 4, 1, 4, 1, 1, 1],
        ["state", "query", "option", "query", "option", "query", "option"],
        [0, 0, 0],
    )
    split = SplitStep(Source([original]), split_on="query", seq_len=4)
    with pytest.warns(UserWarning, match="Increase seq_len") as warnings:
        output = list(iter(split.next, None))
    assert len(warnings) == 1
    assert len(output) == 1
    assert output[0].ids == [0, 1, 12, 13]


def test_bucket_packing_is_first_fit_and_pads_only_when_requested():
    samples = [tokenized([length], ["state"]) for length in (6, 3, 4, 7)]
    padded = list(iter(PackAndPadStep(Source(samples), 10, 99).next, None))
    unpadded = list(iter(PackAndPadStep(Source(samples), 10, 99, pad=False).next, None))
    assert [len(row) for row in unpadded] == [9, 7, 4]
    assert [len(row) for row in padded] == [10, 10, 10]
    assert unpadded[0].message_boundaries == [5, 8]
    for row, reference in zip(padded, unpadded, strict=True):
        assert row.ids == reference.ids + [99] * (10 - len(reference))
        assert row.message_boundaries == reference.message_boundaries
    assert sorted(value for row in unpadded for value in row.ids) == sorted(
        value for row in samples for value in row.ids
    )


def test_bucket_limit_emits_before_source_exhaustion():
    source = Source(tokenized([6], ["state"]) for _ in range(40))
    packed = PackAndPadStep(source, 10, 99)
    first = packed.next()
    assert first is not None
    assert len(first) == 10
    assert source.reads == 33
    assert len(list(iter(packed.next, None))) == 39


def test_collation_aligns_labels_and_positions_including_partial_batch():
    samples = [
        tokenized([2, 1, 1, 1], ["state", "query", "option", "option"], [1]),
        tokenized(
            [2, 1, 1, 1, 1], ["state", "query", "option", "option", "option"], [[0.2, 0.3, 0.5]]
        ),
    ]
    packed = PackAndPadStep(Source(samples), 6, 99)
    batches = CollateStep(packed, local_batch_size=3)
    batch = batches.next()
    assert batch is not None
    assert batch.input_ids.shape == (2, 6)
    torch.testing.assert_close(batch.labels, torch.tensor([[0.2, 0.3, 0.5], [0, 1, 0]]))
    assert batch.option_indices.tolist() == [[0, 1, 2], [3, 4, -1]]
    assert Positions(batch).value.tolist() == [[0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 3, 0]]
    assert torch.equal(batch.input_ids[1, -1], torch.tensor(99))
    assert batches.next() is None


def test_packed_attention_preserves_examples_and_isolates_options_and_samples():
    roles = ["state", "query", "example", "example", "option", "option", "state", "query", "option"]
    source = Source([tokenized([1] * len(roles), roles)])
    batch = CollateStep(PackAndPadStep(source, 10, 99), local_batch_size=1).next()
    assert batch is not None
    mask = create_mask(AttentionMask(batch).value.mask_mod, 1, 1, 10, 10, device="cpu")
    visible = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5],
        [6],
        [6, 7],
        [6, 7, 8],
        [9],
    ]
    expected = torch.zeros((10, 10), dtype=torch.bool)
    for row, columns in enumerate(visible):
        expected[row, columns] = True
    torch.testing.assert_close(mask[0, 0], expected)
    assert Positions(batch).value.tolist() == [[0, 1, 2, 2, 3, 3, 0, 1, 2, 0]]
