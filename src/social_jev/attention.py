"""Packed visibility masks and logical token positions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.nn.attention.flex_attention import create_block_mask

if TYPE_CHECKING:
    from .packing import PackedBatch as Batch


ROLES = ("state", "query", "example", "option", "answer")


@dataclass
class AttentionConfig:
    mask: str = "causal"
    state: list[str] = field(default_factory=list)
    query: list[str] = field(default_factory=lambda: ["state"])
    example: list[str] = field(default_factory=lambda: ["state", "query"])
    option: list[str] = field(default_factory=lambda: ["state", "query", "example"])
    answer: list[str] = field(default_factory=lambda: ["state", "query", "example", "option"])

    def __post_init__(self):
        assert self.mask in {"causal", "bidirectional"}
        assert not self.state, "Shared states cannot attend to query-specific messages"
        assert all(set(getattr(self, role)) <= set(ROLES) for role in ROLES)

    @classmethod
    def resolve(cls, values=None, *, head_type="linear"):
        settings = {"option": []} if head_type == "clm" else {}
        settings.update(values or {})
        config = cls(**settings)
        assert head_type != "clm" or not config.option, "CLM requires separate option encoding"
        return config

    def visibility(self, device):
        return torch.tensor(
            [[source in getattr(self, role) for source in ROLES] for role in ROLES],
            dtype=torch.bool,
            device=device,
        )


def message_spans(boundaries, roles, answer_length):
    """Expose the appended answer prefix without changing stored message roles."""
    start = 0
    for index, (boundary, role) in enumerate(zip(boundaries, roles, strict=True)):
        end = boundary + 1
        final_option = role == "option" and (
            index + 1 == len(roles) or roles[index + 1] != "option"
        )
        if answer_length and final_option:
            yield start, end - answer_length, role
            yield end - answer_length, end, "answer"
        else:
            yield start, end, role
        start = end


class Positions:
    def __init__(self, batch: "Batch"):
        positions = []
        for boundaries, roles in zip(batch.message_boundaries, batch.roles, strict=True):
            sequence = []
            state_length = query_length = longest_example = 0
            for start, end, role in message_spans(boundaries, roles, 0):
                length = end - start
                if role == "state":
                    state_length = length
                    query_length = longest_example = 0
                    offset = 0
                elif role == "query":
                    query_length = length
                    longest_example = 0
                    offset = state_length
                elif role == "example":
                    offset = state_length + query_length
                    longest_example = max(longest_example, length)
                else:  # option
                    offset = state_length + query_length + longest_example
                sequence.extend(range(offset, offset + length))
            padding = batch.input_ids.shape[1] - len(sequence)
            positions.append(sequence + [0] * padding)
        self.value = torch.tensor(positions, dtype=torch.long, device=batch.input_ids.device)


def attention_metadata(batch, answer_length):
    metadata = torch.full((*batch.input_ids.shape, 4), -1, dtype=torch.long)
    for row, (boundaries, roles) in enumerate(
        zip(batch.message_boundaries, batch.roles, strict=True)
    ):
        problem = query = -1
        for message, (start, end, role) in enumerate(
            message_spans(boundaries, roles, answer_length)
        ):
            if role == "state":
                problem += 1
                query = -1
            elif role == "query":
                query += 1
            metadata[row, start:end] = torch.tensor([ROLES.index(role), message, problem, query])
    return metadata.to(batch.input_ids.device)


class AttentionMask:
    def __init__(self, batch: "Batch", *, config=None, option_context="full", answer_length=0):
        # Existing callers without explicit settings retain their head's defaults.
        config = config or AttentionConfig.resolve(
            head_type="clm" if option_context is None else "linear"
        )
        if isinstance(config, dict):
            config = AttentionConfig(**config)
        metadata = attention_metadata(batch, answer_length)
        visibility = config.visibility(batch.input_ids.device)
        bidirectional = config.mask == "bidirectional"
        batch_size, length = batch.input_ids.shape

        def allowed(b, _h, q, k):
            qr, qm, qp, qq = metadata[b, q.clamp(max=length - 1)].unbind()
            kr, km, kp, kq = metadata[b, k.clamp(max=length - 1)].unbind()
            valid = (q < length) & (k < length) & (qr >= 0) & (kr >= 0)
            same_message = qm == km
            within = same_message & (bidirectional | (k <= q))
            scope = (qp == kp) & ((kr == 0) | ((qr != 0) & (qq == kq)))
            between = ~same_message & scope & visibility[qr.clamp(min=0), kr.clamp(min=0)]
            padding_self = (q < length) & (k < length) & (qr < 0) & (q == k)
            return (valid & (within | between)) | padding_self

        self.value = create_block_mask(
            allowed, batch_size, None, length, length, device=batch.input_ids.device
        )
