"""Build packed CPU batches for Transformers and TorchTitan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING

import torch

from .data import Step, TokenizeStep

PACKING_LOOKAHEAD = 32

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    from .data import Sample, TokenizedSample


@dataclass
class PackedBatch:
    """Tokens, message boundaries, and optional query targets."""

    input_ids: torch.Tensor
    message_boundaries: list[list[int]]
    roles: list[list[str]]
    query_ids: list[int]
    metadata: list[dict[str, str]]
    option_indices: torch.Tensor
    option_positions: torch.Tensor
    targets: torch.Tensor | None

    @classmethod
    def from_samples(cls, samples: list[TokenizedSample], pad_id: int | None) -> PackedBatch:  # noqa: C901 -- collate tokens and query targets in the same order.
        """Collate token sequences and align target rows with query and option order."""
        rows = [sample.ids for sample in samples]
        if pad_id is not None:
            length = max(map(len, rows))
            rows = [row + [pad_id] * (length - len(row)) for row in rows]
        groups: list[list[int]] = []
        positions: list[tuple[int, int]] = []
        for row, sample in enumerate(samples):
            for end, role in zip(sample.message_boundaries, sample.roles, strict=True):
                if role == "query":
                    groups.append([])
                elif role == "option":
                    groups[-1].append(len(positions))
                    positions.append((row, end))
        width = max(map(len, groups))
        targets = None
        if samples[0].targets is not None:
            labels = [label for sample in samples for label in sample.targets or []]
            targets = torch.zeros((len(groups), width), dtype=torch.float32)
            for row, (label, group) in enumerate(zip(labels, groups, strict=True)):
                if isinstance(label, int):
                    targets[row, label] = 1
                else:
                    targets[row, : len(group)] = torch.tensor(label)
        return cls(
            input_ids=torch.tensor(rows),
            message_boundaries=[s.message_boundaries for s in samples],
            roles=[s.roles for s in samples],
            query_ids=[query for s in samples for query in s.query_ids],
            metadata=[metadata for s in samples for metadata in s.metadata],
            option_indices=torch.tensor([group + [-1] * (width - len(group)) for group in groups]),
            option_positions=torch.tensor(positions),
            targets=targets,
        )

    def to(self, device) -> PackedBatch:
        """Move the tensors used by the model to its device."""
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            option_indices=self.option_indices.to(device),
            option_positions=self.option_positions.to(device),
            targets=None if self.targets is None else self.targets.to(device),
        )


class SampleStep(Step["Sample"]):
    """Open a sample source on the machine consuming it."""

    def __init__(self, source: Callable[[], Iterable[Sample]], rank: int, ranks: int):
        self.samples = islice(source(), rank, None, ranks)

    def next(self) -> Sample | None:
        """Pull the next sample for this rank."""
        return next(self.samples, None)


class SplitByLengthStep(Step["TokenizedSample"]):
    """Split samples between queries to fit the sequence length."""

    def __init__(self, parent: Step[TokenizedSample], seq_len: int):
        self.parent = parent
        self.seq_len = seq_len
        self.remainder: TokenizedSample | None = None

    def next(self) -> TokenizedSample | None:
        """Return the next complete group of queries that fits."""
        sample = self.remainder if self.remainder is not None else self.parent.next()
        if sample is None:
            return None
        fitting, self.remainder = sample.split_at_length(self.seq_len)
        return fitting


class PackAndPadStep(Step["TokenizedSample"]):
    """Pack samples with mert's longest-fitting lookahead."""

    def __init__(self, parent: Step[TokenizedSample], seq_len: int, pad_id: int | None):
        self.parent = parent
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.buffer: list[TokenizedSample] = []

    def next(self) -> TokenizedSample | None:
        """Pack to the sequence limit and right-pad fixed-length rows."""
        while len(self.buffer) < PACKING_LOOKAHEAD:
            sample = self.parent.next()
            if sample is None:
                break
            self.buffer.append(sample)
        if not self.buffer:
            return None
        packed = self.buffer.pop(0)
        while fitting := [
            index
            for index, sample in enumerate(self.buffer)
            if len(packed) + len(sample) <= self.seq_len
        ]:
            index = max(fitting, key=lambda index: len(self.buffer[index]))
            packed = packed + self.buffer.pop(index)
        if self.pad_id is not None:
            packed = replace(packed, ids=packed.ids + [self.pad_id] * (self.seq_len - len(packed)))
        return packed


class CollateStep(Step[PackedBatch]):
    """Gather packed sequences into one CPU batch."""

    def __init__(
        self, parent: Step[TokenizedSample], batch_size: int, pad_id: int | None, *, drop_last: bool
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.drop_last = drop_last

    def next(self) -> PackedBatch | None:
        """Return a collated batch, including the tail when requested."""
        samples = []
        for _ in range(self.batch_size):
            sample = self.parent.next()
            if sample is None:
                break
            samples.append(sample)
        if not samples or (self.drop_last and len(samples) < self.batch_size):
            return None
        return PackedBatch.from_samples(samples, self.pad_id)


def packed_batches(  # noqa: PLR0913 -- configure packing and collation.
    parent: Step[Sample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    seq_len: int,
    batch_size: int,
    drop_last: bool,
    fixed_length: bool = False,
) -> CollateStep:
    """Use the same tokenization and packing for inference and training."""
    tokens = TokenizeStep(parent, tokenizer)
    splits = SplitByLengthStep(tokens, seq_len)
    pad_id = tokenizer.pad_token_id
    packed = PackAndPadStep(splits, seq_len, pad_id if fixed_length else None)
    return CollateStep(packed, batch_size, None if fixed_length else pad_id, drop_last=drop_last)
