import importlib.util
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = Path(__file__).parents[3] / "vla_foundry" / "data" / "processor" / "text_limits.py"
_SPEC = importlib.util.spec_from_file_location("text_limits", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
infer_max_text_seq_len = _MODULE.infer_max_text_seq_len


def test_uses_strictest_processor_or_tokenizer_limit():
    processor = SimpleNamespace(model_max_length=2048, tokenizer=SimpleNamespace(model_max_length=77))

    assert infer_max_text_seq_len(processor) == 77


def test_ignores_unbounded_tokenizer_sentinel():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=10**30))

    assert infer_max_text_seq_len(processor) is None


def test_does_not_treat_invalid_or_boolean_limits_as_lengths():
    processor = SimpleNamespace(model_max_length=True, tokenizer=SimpleNamespace(model_max_length=0))

    assert infer_max_text_seq_len(processor) is None
