"""Neural KV cache compaction package."""

__all__ = [
    "CompactKVCache",
    "StillCompactor",
    "StillLayerCompactor",
]


def __getattr__(name: str):
    if name in {"StillCompactor", "StillLayerCompactor"}:
        from neural_kv.models import StillCompactor, StillLayerCompactor

        values = {
            "StillCompactor": StillCompactor,
            "StillLayerCompactor": StillLayerCompactor,
        }
        return values[name]
    if name == "CompactKVCache":
        from neural_kv.modules import CompactKVCache

        return CompactKVCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
