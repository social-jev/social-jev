"""Optional rented compute integrations."""

from .huggingface import HuggingFaceJobs
from .modal import Modal
from .vast import VastAI

__all__ = ["HuggingFaceJobs", "Modal", "VastAI"]
