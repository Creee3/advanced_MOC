"""Replaceable routing features and an existing, local tokenizer adapter."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from .sampler import unit_vector


class LocalTokenizerCounter:
    """HF text tokens, not an assertion of Ollama chat/prefill token equivalence.

    Loading is strictly local. No model weights or tokenizers are downloaded.
    The production environment must have the lightweight tokenizers package.
    """

    def __init__(self, tokenizer_json, expected_sha256=None):
        path = Path(tokenizer_json).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("tokenizer SHA256 differs from the configured model tokenizer")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Install tokenizers in the runtime, or inject the model's actual token counter") from exc
        self.tokenizer = Tokenizer.from_file(str(path))
        self.name = f"HF-text:{path.name}:sha256:{digest}"

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


@dataclass(frozen=True)
class FeatureSpace:
    kind: str
    model: str
    layer: int | None = None
    position_basis: str = "text"


def pool_token_states(states, message_mask):
    """Pool caller-supplied [tokens,dimension] states on explicitly marked tokens.

    No backend extraction is implied. The caller must supply a consistent
    layer/model/basis on sender and receiver and account for any prefill cost.
    """
    states = np.asarray(states, dtype=np.float64)
    mask = np.asarray(message_mask, dtype=bool)
    if states.ndim != 2 or mask.shape != (len(states),) or not mask.any():
        raise ValueError("expected [tokens,dimension] states and a nonempty token mask")
    return unit_vector(states[mask].mean(axis=0))


def pool_kv_sketch(keys, values, message_mask, *, positions_aligned):
    """Caller supplies [tokens,heads,head_dim] K/V in a shared positional basis."""
    if not positions_aligned:
        raise ValueError("raw RoPE keys from unmatched positions are not a shared routing space")
    keys, values = np.asarray(keys), np.asarray(values)
    if keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError("K/V must share [tokens,heads,head_dim]")
    pooled_k = pool_token_states(keys.reshape(len(keys), -1), message_mask)
    pooled_v = pool_token_states(values.reshape(len(values), -1), message_mask)
    return unit_vector(np.concatenate((pooled_k, pooled_v)))
