"""Helpers for deriving safe text lengths from processor metadata."""

from numbers import Integral

# Hugging Face uses a very large sentinel when a tokenizer has no limit.  It is
# not a useful truncation length and must not turn an unset config into a huge
# allocation.
_MAX_REASONABLE_LENGTH = 1_000_000


def infer_max_text_seq_len(processor) -> int | None:
    """Return the smallest finite text limit advertised by a processor.

    CLIP exposes its 77-token limit on ``processor.tokenizer``, while other
    processors may expose ``model_max_length`` directly.  Returning the minimum
    keeps a wrapper from selecting a length accepted by one component but
    rejected by another.  Hugging Face's large "unbounded" sentinel is ignored.
    """
    candidates = [processor, getattr(processor, "tokenizer", None)]
    lengths = []
    for candidate in candidates:
        value = getattr(candidate, "model_max_length", None)
        if isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < _MAX_REASONABLE_LENGTH:
            lengths.append(int(value))
    return min(lengths) if lengths else None
