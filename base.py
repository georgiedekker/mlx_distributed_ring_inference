"""
Basic support classes for distributed inference.

This module defines a very simple key/value cache and an identity block
placeholder. These can be used when composing models that expect a
``cache`` argument or when experimenting with custom layers. The cache
here does nothing except hold references to keys and values; it does not
implement any eviction or compression logic.
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn

class KVCache:
    """Simple key/value cache for attention layers.

    This minimal cache stores keys and values without any specific
    management strategy. It is intended as a placeholder for
    distributed inference where each rank might need to preserve its
    own cache entries. Real models usually implement more complex
    caching (e.g. ring buffers) directly within their attention
    modules.
    """

    def __init__(self) -> None:
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None


class IdentityBlock(nn.Module):
    """A no‑op neural network block.

    This module simply returns its input. It accepts optional mask and
    cache parameters for API compatibility with more complex blocks.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        return x