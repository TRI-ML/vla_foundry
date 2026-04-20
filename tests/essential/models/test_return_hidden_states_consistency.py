import pytest
import torch

from vla_foundry.models import create_model
from vla_foundry.models.utils import compute_num_image_tokens
from vla_foundry.params.model_params import ModelParams, TransformerHFParams, TransformerParams, VLMHFParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml


class TestReturnHiddenStatesConsistency:
    """Test that all models consistently return 3 items when output_hidden_states=True"""

    @pytest.fixture
    def transformer_config(self):
        return load_params_from_yaml(
            TransformerParams, "tests/essential/params/dummy_configs/dummy_transformer_config.yaml"
        )

    @pytest.fixture
    def vlm_config(self):
        model_params = load_params_from_yaml(
            ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_config.yaml"
        )
        object.__setattr__(model_params, "image_token_id", 999)
        return model_params

    def test_transformer_return_format_consistency(self, transformer_config):
        """Test that Transformer consistently returns 3 items"""
        transformer = create_model(transformer_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is None

        # Test with hidden states
        result = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is not None
        assert isinstance(result.hidden_states, tuple)
        assert len(result.hidden_states) == 2  # n_layers = 2

    def test_transformer_hf_return_format_consistency(self, transformer_config):
        """Test that TransformerHF returns HF output objects"""

        transformer_hf_config = TransformerHFParams(
            hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
        )
        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states - HF models return HF objects
        result = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, use_cache=False
        )
        assert hasattr(result, "logits")
        assert hasattr(result, "past_key_values")
        assert hasattr(result, "hidden_states")
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is None

        # Test with hidden states
        result = transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
        assert hasattr(result, "logits")
        assert hasattr(result, "past_key_values")
        assert hasattr(result, "hidden_states")
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is not None
        assert isinstance(result.hidden_states, tuple)
        assert len(result.hidden_states) == 13  # DialoGPT-small has 13 layers

    def test_vlm_return_format_consistency(self, vlm_config):
        """Test that VLM consistently returns 3 items"""
        vlm = create_model(vlm_config)

        # Compute expected number of image tokens from ViT config
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set image tokens
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        pixel_values = torch.randn(batch_size, 1, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test without hidden states
        result = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=False
        )
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is None

        # Test with hidden states
        result = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=True
        )
        assert result.logits is not None
        assert result.past_key_values is None
        assert result.hidden_states is not None
        assert isinstance(result.hidden_states, tuple)
        assert len(result.hidden_states) == 2  # n_layers = 2

    def test_all_models_return_same_format(self, transformer_config, vlm_config):
        """Test that all models return the same format: (logits, past_key_values, hidden_states)"""
        # This test ensures the interface is consistent across all model types

        # Create all model types
        transformer = create_model(transformer_config)

        transformer_hf_config = TransformerHFParams(
            hf_pretrained="microsoft/git-base", resume_from_checkpoint=None, resume_weights_only=False
        )
        transformer_hf = create_model(transformer_hf_config)

        # Create VLM
        vlm = create_model(vlm_config)

        vlm_hf_config = VLMHFParams(
            hf_pretrained="microsoft/git-base", resume_from_checkpoint=None, resume_weights_only=False
        )
        vlm_hf = create_model(vlm_hf_config)

        transformer.train()
        transformer_hf.train()
        vlm.train()
        vlm_hf.train()

        # Test inputs
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        pixel_values = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)

        # Set image tokens for VLM
        input_ids_vlm = input_ids.clone()
        input_ids_vlm[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids_vlm[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        # Test all models return same format
        models_and_inputs = [
            (transformer, {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}),
            (transformer_hf, {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}),
            (
                vlm,
                {
                    "input_ids": input_ids_vlm,
                    "pixel_values": pixel_values,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                },
            ),
            (
                vlm_hf,
                {
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                },
            ),
        ]

        for model, inputs in models_and_inputs:
            # Test without hidden states
            result = model(**inputs, output_hidden_states=False)
            assert result.logits is not None
            assert result.past_key_values is None
            assert result.hidden_states is None

            # Test with hidden states
            result = model(**inputs, output_hidden_states=True)
            assert result.logits is not None
            assert result.past_key_values is None
            assert result.hidden_states is not None
            assert isinstance(result.hidden_states, tuple)
