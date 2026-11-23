"""Offline Qwen3 execution with the trained scalar linear or MLP head."""

from typing import TYPE_CHECKING, cast

import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import AutoTokenizer, PreTrainedModel, Qwen3Model

from .attention import AttentionMask, Positions
from .config import SocialJevConfig
from .data import Batch, Encoded, FormatStep, packed_batches
from .head import DecisionHeadConfig, ScalarHead

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .data import PackedBatch
    from .types import Request, Result


class SocialJevModel(PreTrainedModel):
    """Load the backbone and scalar-head weights from a SocialJev checkpoint.

    Args:
        config: Qwen3 configuration for the scalar classifier.

    Examples:
        >>> model = SocialJevModel.from_pretrained(repo)  # doctest: +SKIP
    """

    config_class = SocialJevConfig
    """Configuration class for scalar-head checkpoints."""
    base_model_prefix = "backbone"
    """Checkpoint prefix for backbone parameters."""
    _supports_sdpa = True
    """Enable the backbone's scaled dot-product attention implementation."""

    def __init__(self, config: SocialJevConfig):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.backbone = Qwen3Model(config.backbone_config)
        self.decision_head = ScalarHead(
            DecisionHeadConfig(**config.decision_head), config.backbone_config.hidden_size
        )
        self.post_init()

    def save_pretrained(self, save_directory, **kwargs):
        """Save weights and configuration with the attached tokenizer and template."""
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask):
        """Return raw logits from each candidate's last nonpadding token."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        positions = attention_mask.sum(-1) - 1
        return self.decision_head(hidden[rows, positions])

    def forward_packed(self, batch: "PackedBatch"):
        """Read one scalar for each option using the checkpoint attention rules."""
        block_mask = AttentionMask(batch).value
        mask = block_mask
        if self.backbone.config._attn_implementation != "flex_attention":
            size, length = batch.input_ids.shape
            mask = create_mask(
                block_mask.mask_mod, size, 1, length, length, device=str(batch.input_ids.device)
            )
        hidden = self.backbone(
            input_ids=batch.input_ids,
            position_ids=Positions(batch).value,
            attention_mask={"full_attention": mask},
            use_cache=False,
        ).last_hidden_state
        rows, positions = batch.option_positions.unbind(dim=-1)
        scores = self.decision_head(hidden[rows, positions])
        return scores[batch.option_indices.clamp(min=0)].masked_fill(
            batch.option_indices < 0, -torch.inf
        )


class TransformersEngine:
    """Run a collection of candidate sequences as one padded batch.

    Args:
        repo: Self-contained checkpoint path or Hugging Face repository.

    Examples:
        >>> engine = TransformersEngine(repo)  # doctest: +SKIP
        >>> engine.process([])  # doctest: +SKIP
        []
    """

    def __init__(self, repo, **kwargs):
        self.model = SocialJevModel.from_pretrained(repo, **kwargs).eval()
        self.tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(repo, revision=kwargs.get("revision")),
        )
        self.model.tokenizer = self.tokenizer
        self.pad_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def process(self, inputs: list[list[int]]) -> list[float]:
        """Process token sequences synchronously and return scalar scores in input order."""
        if not inputs:
            return []
        assert self.model is not None
        maximum = self.model.config.backbone_config.max_position_embeddings
        if any(len(ids) > maximum for ids in inputs):
            message = "Candidate sequence exceeds model context capacity"
            raise ValueError(message)
        batch = Batch.from_inputs(inputs, self.pad_id).to(self.model.device)
        return (
            self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            .float()
            .tolist()
        )

    @torch.inference_mode()
    def query(self, requests: "Request | list[Request]") -> "list[list[Result]]":
        """Score packed requests and restore the original query order."""
        assert self.model is not None
        items = requests if isinstance(requests, list) else [requests]
        maximum = self.model.config.backbone_config.max_position_embeddings
        batches = packed_batches(
            FormatStep(items), self.tokenizer, seq_len=maximum, batch_size=1, pad=False
        )
        scores = {}
        while (batch := batches.next()) is not None:
            logits = self.model.forward_packed(batch.to(self.model.device)).float().tolist()
            counts = (batch.option_indices >= 0).sum(dim=-1).tolist()
            for index, row, count in zip(batch.query_ids, logits, counts, strict=True):
                scores[index] = row[:count]
        return Encoded([], items).results(
            [score for index in range(len(scores)) for score in scores[index]]
        )

    def stop(self):
        """Release the model weights."""
        self.model = None
