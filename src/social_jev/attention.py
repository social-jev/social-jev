"""Packed visibility masks and logical token positions."""

from typing import TYPE_CHECKING

import torch
from torch.nn.attention.flex_attention import create_block_mask

if TYPE_CHECKING:
    from .data import PackedBatch as Batch


ROLES = ("state", "query", "example", "option")


def message_spans(boundaries, roles):
    """Yield each message's token span and role."""
    start = 0
    for boundary, role in zip(boundaries, roles, strict=True):
        end = boundary + 1
        yield start, end, role
        start = end


class Positions:
    def __init__(self, batch: "Batch"):
        positions = []
        for boundaries, roles in zip(batch.message_boundaries, batch.roles, strict=True):
            sequence = []
            state_length = query_length = longest_example = 0
            for start, end, role in message_spans(boundaries, roles):
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


def attention_metadata(batch):
    metadata = torch.full((*batch.input_ids.shape, 4), -1, dtype=torch.long)
    for row, (boundaries, roles) in enumerate(
        zip(batch.message_boundaries, batch.roles, strict=True)
    ):
        problem = query = -1
        for message, (start, end, role) in enumerate(message_spans(boundaries, roles)):
            if role == "state":
                problem += 1
                query = -1
            elif role == "query":
                query += 1
            metadata[row, start:end] = torch.tensor([ROLES.index(role), message, problem, query])
    return metadata.to(batch.input_ids.device)


class AttentionMask:
    def __init__(self, batch: "Batch"):
        metadata = attention_metadata(batch)
        batch_size, length = batch.input_ids.shape

        def allowed(b, _h, q, k):
            qr, qm, qp, qq = metadata[b, q.clamp(max=length - 1)].unbind()
            kr, km, kp, kq = metadata[b, k.clamp(max=length - 1)].unbind()
            valid = (q < length) & (k < length) & (qr >= 0) & (kr >= 0)
            same_message = qm == km
            within = same_message & (k <= q)
            scope = (qp == kp) & ((kr == 0) | ((qr != 0) & (qq == kq)))
            # role order encodes state → query → example → option visibility
            between = ~same_message & scope & (kr < qr)
            padding_self = (q < length) & (k < length) & (qr < 0) & (q == k)
            return (valid & (within | between)) | padding_self

        self.value = create_block_mask(
            allowed, batch_size, None, length, length, device=batch.input_ids.device
        )
