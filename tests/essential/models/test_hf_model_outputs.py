"""Tests for HF model outputs to ensure they return standard HF objects."""

import pytest
import torch

from vla_foundry.models import create_model
from vla_foundry.params.model_params import TransformerHFParams, VLMHFParams


class TestHFModelOutputs:
    """Test that HF models return standard HF output objects instead of tuples."""

    @pytest.fixture
    def transformer_hf_config(self):
        return TransformerHFParams(
            hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
        )

    @pytest.fixture
    def vlm_hf_config(self):
        return VLMHFParams(hf_pretrained="microsoft/git-base", resume_from_checkpoint=None, resume_weights_only=False)

    def test_transformer_hf_returns_hf_object(self, transformer_hf_config):
        """Test that TransformerHF returns standard HF output objects."""
        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, use_cache=False
        )

        # Verify it returns a standard HF output object
        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "past_key_values")
        assert hasattr(outputs, "hidden_states")
        assert outputs.logits is not None
        assert outputs.past_key_values is None
        assert outputs.hidden_states is None

    def test_transformer_hf_with_hidden_states(self, transformer_hf_config):
        """Test that TransformerHF returns HF objects with hidden states."""
        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )

        # Verify it returns a standard HF output object
        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "past_key_values")
        assert hasattr(outputs, "hidden_states")
        assert outputs.logits is not None
        assert outputs.past_key_values is None
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) == 13  # DialoGPT-small has 13 layers

    def test_vlm_hf_returns_hf_object(self, vlm_hf_config):
        """Test that VLMHF returns standard HF output objects."""
        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = vlm.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        # Verify it returns a standard HF output object
        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "past_key_values")
        assert hasattr(outputs, "hidden_states")
        assert outputs.logits is not None
        assert outputs.past_key_values is None
        assert outputs.hidden_states is None

    def test_vlm_hf_with_hidden_states(self, vlm_hf_config):
        """Test that VLMHF returns HF objects with processed hidden states."""
        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = vlm.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Verify it returns a standard HF output object
        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "past_key_values")
        assert hasattr(outputs, "hidden_states")
        assert outputs.logits is not None
        assert outputs.past_key_values is None
        assert isinstance(outputs.hidden_states, tuple)
        for hidden_state in outputs.hidden_states:
            assert hidden_state[0].dtype == torch.float32
