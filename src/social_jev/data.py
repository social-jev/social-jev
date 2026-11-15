"""Dataclass transformations exposed through a pull-based Processor chain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from importlib.resources import files
from math import isclose
from typing import TYPE_CHECKING, cast

import torch

from .types import Request, Result

if TYPE_CHECKING:
    from collections.abc import Iterator

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

    def split(self) -> Iterator[TokenizedSample]:
        """Extract one document-question-option sequence per candidate."""
        state = self[:1]
        query = None
        for index, role in enumerate(self.roles):
            if role == "query":
                query = self[index : index + 1]
            elif role == "option":
                assert query is not None
                yield state + query + self[index : index + 1]

    def split_at_length(self, maximum: int) -> tuple[TokenizedSample, TokenizedSample | None]:
        """Keep complete queries together and repeat the state when splitting."""
        if len(self) <= maximum:
            return self, None
        state_end = self.roles.index("query")
        ends = [
            index for index, role in enumerate(self.roles) if role == "query" and index > state_end
        ]
        fitting = [end for end in ends if self.message_boundaries[end - 1] < maximum]
        if not fitting:
            message = "Query and its options exceed model context capacity."
            raise ValueError(message)
        end = fitting[-1]
        return self[:end], self[:state_end] + self[end:]


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
    """Delegate candidate extraction to TokenizedSample.split().

    Args:
        parent: Tokenized complete-request source.

    Examples:
        >>> step = SplitStep(parent)  # doctest: +SKIP
        >>> candidate = step.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[TokenizedSample]):
        """Attach the source and initialize this call's pending candidates."""
        self.parent = parent
        self.pending: Iterator[TokenizedSample] = iter(())

    def next(self) -> TokenizedSample | None:
        """Pull the next candidate, splitting a request when needed."""
        while (candidate := next(self.pending, None)) is None:
            sample = self.parent.next()
            if sample is None:
                return None
            self.pending = iter(sample.split())
        return candidate


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
