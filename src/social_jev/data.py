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

from .types import OptionLabels, Request, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase


type Target = int | list[float]


@dataclass
class Sample:
    """A complete request represented as role-tagged messages."""

    messages: list[dict[str, str]]
    """Document, questions, and options in their original order."""

    targets: list[Target] | None = None
    """One reference target per query."""
    metadata: dict[str, str] = field(default_factory=dict)
    """Caller-supplied categories for reporting."""
    query_ids: list[int] = field(default_factory=list)
    """Positions of the queries in the original inference request batch."""

    def __post_init__(self):  # noqa: C901 -- validate roles and their aligned targets.
        roles = [message["role"] for message in self.messages]
        groups: list[list[str]] = []
        if not roles or roles[0] != "state":
            message = "A sample must start with a state message."
            raise ValueError(message)
        for entry in self.messages[1:]:
            role = entry["role"]
            if role == "query":
                groups.append([])
            elif role == "option" and groups:
                groups[-1].append(entry["content"])
            elif role != "example" or not groups or groups[-1]:
                message = "Expected a query, optional examples, and then its options."
                raise ValueError(message)
        if not groups or any(
            not options or len(set(options)) != len(options) for options in groups
        ):
            message = "Each query requires distinct answer options."
            raise ValueError(message)
        if self.targets is None:
            return
        if len(self.targets) != len(groups):
            message = "Supply one target per query."
            raise ValueError(message)
        for target, options in zip(self.targets, groups, strict=True):
            if isinstance(target, int):
                if not 0 <= target < len(options):
                    message = "Target index is outside the query's options."
                    raise ValueError(message)
            elif (
                len(target) != len(options)
                or any(p < 0 for p in target)
                or not isclose(sum(target), 1.0, abs_tol=1e-6)
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

    targets: list[Target] | None = None
    """Targets aligned with the query messages."""
    metadata: list[dict[str, str]] = field(default_factory=list)
    """Reporting categories for each query."""
    query_ids: list[int] = field(default_factory=list)
    """Original inference positions for each query."""

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
            sample.targets,
            [sample.metadata] * roles.count("query"),
            sample.query_ids,
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
            targets=None if self.targets is None else self.targets[queries],
            metadata=self.metadata[queries],
            query_ids=self.query_ids[queries],
        )

    def __add__(self, other: TokenizedSample) -> TokenizedSample:
        """Concatenate messages and adjust boundaries by the existing token length."""
        return TokenizedSample(
            ids=self.ids + other.ids,
            message_boundaries=self.message_boundaries
            + [end + len(self) for end in other.message_boundaries],
            roles=self.roles + other.roles,
            targets=(
                None
                if self.targets is None and other.targets is None
                else (self.targets or []) + (other.targets or [])
            ),
            metadata=self.metadata + other.metadata,
            query_ids=self.query_ids + other.query_ids,
        )

    def select_queries(self, start: int, stop: int | None = None) -> TokenizedSample | None:
        queries = [index for index, role in enumerate(self.roles) if role == "query"]
        stop = len(queries) if stop is None else stop
        if start == stop:
            return None
        boundaries = [*queries, len(self.roles)]
        return self[: queries[0]] + self[boundaries[start] : boundaries[stop]]

    def split(
        self, *, on: Literal["option", "query"] = "option", maximum: int | None = None
    ) -> tuple[TokenizedSample | None, TokenizedSample | None]:
        """Extract one option sequence or a fitting group of complete queries."""
        queries = [index for index, role in enumerate(self.roles) if role == "query"]
        if on == "option":
            option = self.roles.index("option")
            fitting = self[:1] + self[queries[0] : queries[0] + 1] + self[option : option + 1]
            end = queries[1] if len(queries) > 1 else len(self.roles)
            remainder = (
                self[:option] + self[option + 1 :] if option + 1 < end else self.select_queries(1)
            )
            return fitting, remainder

        assert maximum is not None
        if len(self) <= maximum:
            return self, None
        starts = [0, *(end + 1 for end in self.message_boundaries)]
        query_ends = [starts[index] for index in queries[1:]] + [len(self)]
        count = 0
        for end in query_ends:
            if end > maximum:
                break
            count += 1
        return self.select_queries(0, count), self.select_queries(max(count, 1))


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
    """Tokens, message boundaries, and optional query targets."""

    input_ids: torch.Tensor
    message_boundaries: list[list[int]]
    roles: list[list[str]]
    query_ids: list[int]
    metadata: list[dict[str, str]]
    option_indices: torch.Tensor
    option_positions: torch.Tensor
    targets: OptionLabels | None

    @classmethod
    def from_samples(cls, samples: list[TokenizedSample]) -> PackedBatch:  # noqa: C901 -- collate tokens and query targets in the same order.
        """Collate token sequences and align target rows with query and option order."""
        rows = [sample.ids for sample in samples]
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
        self.query_id = 0

    def next(self) -> Sample | None:
        """Format the next complete request."""
        request = next(self.requests, None)
        if request is None:
            return None
        sample = Sample.from_request(request)
        sample.query_ids = list(range(self.query_id, self.query_id + len(request.queries)))
        self.query_id += len(request.queries)
        return sample


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
        self.remainder: TokenizedSample | None = None
        self.warned = False

    def next(self) -> TokenizedSample | None:
        while True:
            sample = self.remainder if self.remainder is not None else self.parent.next()
            if sample is None:
                return None
            fitting, self.remainder = sample.split(on=self.split_on, maximum=self.seq_len)
            if fitting is not None:
                return fitting
            if sample.targets is None:
                message = "Query and its options exceed model context capacity."
                raise ValueError(message)
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
    """Build the shared packed training and Transformers inference pipeline."""
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
