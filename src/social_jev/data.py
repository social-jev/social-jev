"""Dataclass transformations exposed through a pull-based Processor chain."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from importlib.resources import files
from itertools import islice
from math import isclose
from random import Random
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import OptionLabels, QueryLabel, Request, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from transformers import PreTrainedTokenizerBase


@dataclass
class Sample:
    """A complete request represented as role-tagged messages."""

    messages: list[dict[str, str]]
    """Document, questions, and options in their original order."""

    labels: list[QueryLabel] | None = None
    """One reference label per query."""
    metadata: dict[str, str] = field(default_factory=dict)
    """Caller-supplied categories for reporting."""

    def __post_init__(self):  # noqa: C901 -- keep the label contract together.
        """Check that supplied labels match the query options."""
        if self.labels is None:
            return
        counts = []
        for entry in self.messages:
            if entry["role"] == "query":
                counts.append(0)
            elif entry["role"] == "option":
                counts[-1] += 1
        if len(self.labels) != len(counts):
            message = "Supply one label per query."
            raise ValueError(message)
        for label, count in zip(self.labels, counts, strict=True):
            if isinstance(label, int):
                if not 0 <= label < count:
                    message = "Label index is outside the query's options."
                    raise ValueError(message)
            elif (
                len(label) != count
                or any(value < 0 for value in label)
                or not isclose(sum(label), 1.0, abs_tol=1e-6)
            ):
                message = "Supply one nonnegative probability per option summing to one."
                raise ValueError(message)

    @classmethod
    def from_request(cls, request: Request) -> Sample:
        """Format a complete typed request into messages."""
        messages = [{"role": "state", "content": request.state}]
        for query in request.queries:
            messages.append({"role": "query", "content": query.text})
            messages.extend({"role": "option", "content": option} for option in query.options)
        return cls(messages)


@dataclass
class TokenizedSample:
    """A tokenized request with message boundaries and roles."""

    ids: list[int]
    """Token IDs for the complete request."""
    message_boundaries: list[int]
    """Inclusive terminal token index of each message, matching mert."""
    roles: list[str]
    """Message roles aligned with their token boundaries."""

    labels: list[QueryLabel] | None = None
    """Labels aligned with the query messages."""
    metadata: list[dict[str, str]] = field(default_factory=list)
    """Reporting categories for each query."""

    @classmethod
    def from_sample(cls, sample: Sample, tokenizer: PreTrainedTokenizerBase) -> TokenizedSample:
        """Apply the checkpoint template once to the complete message list."""
        ids = cast(
            "list[int]",
            tokenizer.apply_chat_template(
                sample.messages,
                tokenize=True,
                return_dict=False,
            ),
        )
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        boundaries = [index for index, token in enumerate(ids) if token == end_id]
        roles = [message["role"] for message in sample.messages]
        if len(boundaries) != len(roles) or not boundaries or boundaries[-1] != len(ids) - 1:
            message = "Checkpoint template must terminate each message with <|im_end|>"
            raise ValueError(message)
        return cls(
            ids,
            boundaries,
            roles,
            sample.labels,
            [sample.metadata] * roles.count("query"),
        )

    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.ids)

    def __getitem__(self, selection: slice) -> TokenizedSample:
        """Slice messages while preserving mert's token offsets and boundaries."""
        start, stop, _ = selection.indices(len(self.roles))
        stop = max(start, stop)
        offsets = [0, *(end + 1 for end in self.message_boundaries)]
        token_start, token_stop = offsets[start], offsets[stop]
        queries = slice(self.roles[:start].count("query"), self.roles[:stop].count("query"))
        return TokenizedSample(
            ids=self.ids[token_start:token_stop],
            message_boundaries=[end - token_start for end in self.message_boundaries[start:stop]],
            roles=self.roles[start:stop],
            labels=None if self.labels is None else self.labels[queries],
            metadata=self.metadata[queries],
        )

    def __add__(self, other: TokenizedSample) -> TokenizedSample:
        """Concatenate messages and adjust boundaries by the existing token length."""
        labels = None
        if self.labels is not None:
            assert other.labels is not None
            labels = self.labels + other.labels
        return TokenizedSample(
            ids=self.ids + other.ids,
            message_boundaries=self.message_boundaries
            + [end + len(self) for end in other.message_boundaries],
            roles=self.roles + other.roles,
            labels=labels,
            metadata=self.metadata + other.metadata,
        )

    def split(  # noqa: C901 -- two explicit split modes, sharing message slicing.
        self, *, on: Literal["option", "query"] = "option", maximum: int | None = None
    ) -> Iterator[TokenizedSample]:
        """Yield candidates, or groups of complete queries that fit the limit."""
        state = self[:1]
        if on == "option":
            query = None
            for index, role in enumerate(self.roles):
                if role == "query":
                    query = self[index : index + 1]
                elif role == "option":
                    assert query is not None
                    yield state + query + self[index : index + 1]
            return

        assert maximum is not None
        if len(self) <= maximum:
            yield self
            return

        starts = [index for index, role in enumerate(self.roles) if role == "query"]
        state = self[: starts[0]]
        chunk = state
        for start, end in zip(starts, [*starts[1:], len(self.roles)], strict=True):
            query = self[start:end]
            if len(chunk) > len(state) and len(chunk) + len(query) > maximum:
                yield chunk
                chunk = state
            chunk = chunk + query
        yield chunk


@dataclass
class Batch:
    """Padded candidate tensors for offline Transformers execution."""

    input_ids: torch.Tensor
    """Candidate token IDs with right padding."""
    attention_mask: torch.Tensor
    """One for real tokens and zero for padding."""

    @classmethod
    def from_inputs(cls, inputs: list[list[int]], pad_id: int) -> Batch:
        """Pad candidate sequences to the longest input in this submission."""
        length = max(map(len, inputs))
        return cls(
            torch.tensor([ids + [pad_id] * (length - len(ids)) for ids in inputs]),
            torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids in inputs]),
        )

    def to(self, device) -> Batch:
        """Transfer padded candidate tensors to the backbone input device."""
        return replace(
            self, input_ids=self.input_ids.to(device), attention_mask=self.attention_mask.to(device)
        )


@dataclass
class Encoded:
    """Candidate inputs and their document/question output grouping."""

    inputs: list[list[int]]
    """Unpadded candidate sequences in document, question, and option order."""
    requests: list[Request]
    """Original requests used to reconstruct nested results."""

    def results(self, scores: list[float]) -> list[list[Result]]:
        """Normalize candidate logits within each question and restore document order."""
        expected = sum(len(query.options) for item in self.requests for query in item.queries)
        if len(scores) != expected:
            message = "Engine returned a different number of candidate scores"
            raise ValueError(message)
        rows = iter(scores)
        documents = []
        for request in self.requests:
            answers = []
            for query in request.queries:
                logits = torch.tensor([next(rows) for _ in query.options], dtype=torch.float32)
                probabilities = logits.softmax(0)
                answers.append(
                    Result(
                        choice=query.options[int(probabilities.argmax())],
                        probabilities=dict(zip(query.options, probabilities.tolist(), strict=True)),
                    )
                )
            documents.append(answers)
        return documents


@dataclass
class PackedBatch:
    """Tokens, message boundaries, and optional query labels."""

    input_ids: torch.Tensor
    message_boundaries: list[list[int]]
    roles: list[list[str]]
    metadata: list[dict[str, str]]
    option_indices: torch.Tensor
    option_positions: torch.Tensor
    labels: OptionLabels | None

    @classmethod
    def from_samples(cls, samples: list[TokenizedSample]) -> PackedBatch:  # noqa: C901 -- collate aligned options and labels together.
        """Stack packed rows and align option coordinates, labels, and metadata."""
        groups: list[list[int]] = []
        positions = []
        for row, sample in enumerate(samples):
            for end, role in zip(sample.message_boundaries, sample.roles, strict=True):
                if role == "query":
                    groups.append([])
                elif role == "option":
                    groups[-1].append(len(positions))
                    positions.append((row, end))

        labels = None
        if samples[0].labels is not None:
            query_labels = []
            for sample in samples:
                assert sample.labels is not None
                query_labels.extend(sample.labels)
            distributions = []
            for label, options in zip(query_labels, groups, strict=True):
                if isinstance(label, int):
                    distribution = torch.zeros(len(options))
                    distribution[label] = 1.0
                else:
                    distribution = torch.tensor(label, dtype=torch.float32)
                distributions.append(distribution)
            labels = pad_sequence(distributions, batch_first=True)

        return cls(
            input_ids=torch.tensor([sample.ids for sample in samples]),
            message_boundaries=[sample.message_boundaries for sample in samples],
            roles=[sample.roles for sample in samples],
            metadata=[values for sample in samples for values in sample.metadata],
            option_indices=pad_sequence(
                [torch.tensor(group) for group in groups],
                batch_first=True,
                padding_value=-1,
            ),
            option_positions=torch.tensor(positions),
            labels=labels,
        )

    @property
    def option_mask(self):
        """Mark supplied options rather than padding columns."""
        return self.option_indices >= 0

    def to(self, device) -> PackedBatch:
        """Move the tensors used by the model to its device."""
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            option_indices=self.option_indices.to(device),
            option_positions=self.option_positions.to(device),
            labels=None if self.labels is None else self.labels.to(device),
        )


class Step[T](ABC):
    """Pull items from a preparation stage.

    Examples:
        >>> FormatStep([]).next() is None
        True
    """

    @abstractmethod
    def next(self) -> T | None:
        """Return the next item, or None when exhausted."""
        ...


class SampleStep(Step[Sample]):
    """Read and shuffle a bounded stream before selecting this rank's samples."""

    def __init__(
        self,
        source: Callable[[], Iterable[Sample]],
        rank: int,
        world_size: int,
        *,
        infinite: bool = True,
        shuffle_buffer_size: int = 256,
    ):
        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.infinite = infinite
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = 0
        self.index = 0
        self.reset()

    def reset(self):
        self.samples = iter(self.source())
        self.random = Random(self.epoch)
        self.buffer = list(islice(self.samples, self.shuffle_buffer_size))

    def next(self) -> Sample | None:  # noqa: C901 -- bounded shuffle, restart, and rank selection.
        while True:
            if not self.buffer:
                if not self.infinite:
                    return None
                self.epoch += 1
                self.reset()
                if not self.buffer:
                    return None
            position = self.random.randrange(len(self.buffer))
            sample = self.buffer[position]
            replacement = next(self.samples, None)
            if replacement is None:
                self.buffer.pop(position)
            else:
                self.buffer[position] = replacement
            selected = self.index % self.world_size == self.rank
            self.index += 1
            if selected:
                return sample


class FormatStep(Step[Sample]):
    """Pull requests and delegate message formatting to Sample.

    Args:
        requests: Documents to classify in output order.

    Examples:
        >>> FormatStep([]).next() is None
        True
    """

    def __init__(self, requests: list[Request]):
        """Initialize this call's request cursor."""
        self.requests = iter(requests)

    def next(self) -> Sample | None:
        """Format the next complete request."""
        request = next(self.requests, None)
        return None if request is None else Sample.from_request(request)


class TokenizeStep(Step[TokenizedSample]):
    """Delegate candidate tokenization to TokenizedSample.

    Args:
        parent: Formatted question source.
        tokenizer: Checkpoint tokenizer shared across calls.

    Examples:
        >>> isinstance(TokenizeStep, type)
        True
    """

    def __init__(self, parent: Step[Sample], tokenizer: PreTrainedTokenizerBase):
        """Attach the formatted source and tokenizer settings."""
        self.parent = parent
        self.tokenizer = tokenizer

    def next(self) -> TokenizedSample | None:
        """Pull and tokenize the next question's candidates."""
        sample = self.parent.next()
        if sample is None:
            return None
        return TokenizedSample.from_sample(sample, self.tokenizer)


class SplitStep(Step[TokenizedSample]):
    """Pull individual option sequences or length-limited groups of queries."""

    def __init__(
        self,
        parent: Step[TokenizedSample],
        *,
        split_on: Literal["option", "query"] = "option",
        seq_len: int | None = None,
    ):
        self.parent = parent
        self.split_on = split_on
        self.seq_len = seq_len
        self.pending: Iterator[TokenizedSample] = iter(())
        self.warned = False

    def next(self) -> TokenizedSample | None:
        while True:
            sample = next(self.pending, None)
            if sample is None:
                source = self.parent.next()
                if source is None:
                    return None
                self.pending = source.split(on=self.split_on, maximum=self.seq_len)
                continue
            if self.seq_len is None or len(sample) <= self.seq_len:
                return sample
            if not self.warned:
                warnings.warn(
                    "Skipped a query because its state, query, and options exceed seq_len. "
                    "Increase seq_len to include longer examples.",
                    stacklevel=2,
                )
                self.warned = True


class PackAndPadStep(Step[TokenizedSample]):
    """Pack samples into open buckets and optionally pad emitted rows to seq_len."""

    def __init__(self, parent: Step[TokenizedSample], seq_len: int, pad_id: int, *, pad=True):
        self.parent = parent
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.pad = pad
        self.buckets: list[TokenizedSample] = []
        self.exhausted = False

    @property
    def fullest_bucket_index(self) -> int:
        return max(range(len(self.buckets)), key=lambda index: len(self.buckets[index]))

    def next(self) -> TokenizedSample | None:  # noqa: C901 -- first-fit buckets and finite draining.
        while not self.exhausted and len(self.buckets) <= 32:  # noqa: PLR2004 -- reference bucket limit.
            sample = self.parent.next()
            if sample is None:
                self.exhausted = True
                break
            for index, bucket in enumerate(self.buckets):
                if len(bucket) + len(sample) <= self.seq_len:
                    self.buckets[index] = bucket + sample
                    break
            else:
                self.buckets.append(sample)
        if not self.buckets:
            return None
        packed = self.buckets.pop(self.fullest_bucket_index)
        if self.pad:
            packed = replace(packed, ids=packed.ids + [self.pad_id] * (self.seq_len - len(packed)))
        return packed


class CollateStep(Step[PackedBatch]):
    """Stack packed rows into a CPU batch, retaining a finite source's final rows."""

    def __init__(self, parent: Step[TokenizedSample], batch_size: int):
        self.parent = parent
        self.batch_size = batch_size

    def next(self) -> PackedBatch | None:
        samples = []
        for _ in range(self.batch_size):
            sample = self.parent.next()
            if sample is None:
                break
            samples.append(sample)
        return PackedBatch.from_samples(samples) if samples else None


def packed_batches(
    parent: Step[Sample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    seq_len: int,
    batch_size: int,
    pad: bool = True,
) -> CollateStep:
    """Build packed batches for training and validation."""
    tokens = TokenizeStep(parent, tokenizer)
    split = SplitStep(tokens, split_on="query", seq_len=seq_len)
    packed = PackAndPadStep(split, seq_len, tokenizer.pad_token_id, pad=pad)
    return CollateStep(packed, batch_size)


class Processor:
    """Expose synchronous preparation and result decoding.

    Args:
        tokenizer: Checkpoint tokenizer with its trained chat template.

    Examples:
        >>> processor = Processor(tokenizer)  # doctest: +SKIP
        >>> processor.decode([], processor.encode([]))  # doctest: +SKIP
        []
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Store immutable preparation settings; each encode builds its own chain."""
        self.tokenizer = tokenizer

    @staticmethod
    def template() -> str:
        """Read the bundled linear-head template used when exporting checkpoints."""
        return files("social_jev").joinpath("templates/messages.jinja").read_text()

    def encode(self, requests: Request | list[Request]) -> Encoded:
        """Run a fresh Step.next chain and collect unpadded candidate sequences."""
        items = [requests] if isinstance(requests, Request) else requests
        formatted = FormatStep(items)
        tokenized = TokenizeStep(formatted, self.tokenizer)
        split = SplitStep(tokenized)
        inputs = []
        while (sample := split.next()) is not None:
            inputs.append(sample.ids)
        return Encoded(inputs, items)

    @staticmethod
    def decode(scores: list[float], encoded: Encoded) -> list[list[Result]]:
        """Delegate nested result reconstruction to the encoded dataclass."""
        return encoded.results(scores)
