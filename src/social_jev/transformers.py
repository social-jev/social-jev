"""Offline Qwen3 execution with the trained scalar linear or MLP head."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import AutoTokenizer, PreTrainedModel, Qwen3Model

from .attention import AttentionMask, Positions
from .config import SocialJevConfig
from .data import Batch, PackedBatch
from .head import DecisionHeadConfig, ScalarHead

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def is_dcp_checkpoint(source: str | Path) -> bool:
    """Identify a local distributed checkpoint."""
    return isinstance(source, Path) and (source / ".metadata").is_file()


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

    def forward(self, batch: Batch | PackedBatch):
        """Return a tensor of scores arranged by query and option."""
        input_ids = batch.input_ids
        if isinstance(batch, PackedBatch):
            mask = AttentionMask(batch).value
            if self.backbone.config._attn_implementation != "flex_attention":
                size, length = input_ids.shape
                mask = create_mask(
                    mask.mask_mod, size, 1, length, length, device=str(input_ids.device)
                )
            attention_mask = {"full_attention": mask}
            position_ids = Positions(batch).value
            option_positions = batch.option_positions
            option_indices = batch.option_indices
        else:
            attention_mask = batch.attention_mask
            position_ids = None
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            positions = attention_mask.sum(dim=-1) - 1
            option_positions = torch.stack((rows, positions), dim=-1)
            option_indices = rows[:, None]

        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        rows, positions = option_positions.unbind(dim=-1)
        scores = self.decision_head(hidden[rows, positions])
        return scores[option_indices.clamp(min=0)].masked_fill(option_indices < 0, -torch.inf)


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
        return self.model(batch).squeeze(-1).float().tolist()

    def stop(self):
        """Release the model weights."""
        self.model = None
