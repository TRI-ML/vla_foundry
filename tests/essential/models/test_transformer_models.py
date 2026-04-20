from unittest.mock import Mock, patch

import pytest
import torch

from vla_foundry.models import create_model
from vla_foundry.params.model_params import TransformerParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml


class TestTransformer:
    @pytest.fixture
    def transformer_config(self):
        return load_params_from_yaml(
            TransformerParams, "tests/essential/params/dummy_configs/dummy_transformer_config.yaml"
        )

    @pytest.fixture
    def transformer(self, transformer_config):
        return create_model(transformer_config)

    def test_transformer_forward_basic(self, transformer):
        """Test basic forward pass without hidden states"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert output.hidden_states is None

    def test_transformer_forward_with_hidden_states(self, transformer):
        """Test forward pass with hidden states returned"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert isinstance(output.hidden_states, tuple)
        assert len(output.hidden_states) == 2  # n_layers = 2
        assert output.hidden_states[0].shape == (batch_size, seq_len, 128)  # hidden_dim = 128
        assert output.hidden_states[1].shape == (batch_size, seq_len, 128)

    def test_transformer_forward_with_inputs_embeds(self, transformer):
        """Test forward pass with input embeddings instead of input_ids"""
        batch_size, seq_len = 2, 10
        inputs_embeds = torch.randn(batch_size, seq_len, 128)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=False)

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert output.hidden_states is None

    def test_transformer_forward_with_past_key_values(self, transformer):
        """Test forward pass with past key values for caching"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Create mock past key values
        n_heads = transformer.model_params.n_heads
        head_dim = transformer.model_params.hidden_dim // n_heads
        past_key_values = [
            (
                torch.randn(batch_size, 5, n_heads, head_dim),
                torch.randn(batch_size, 5, n_heads, head_dim),
            )
            for _ in range(transformer.n_layers)
        ]

        output = transformer(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is not None
        assert len(output.past_key_values) == 2
        assert output.hidden_states is None

    def test_transformer_forward_error_no_input(self, transformer):
        """Test that error is raised when neither input_ids nor inputs_embeds provided"""
        attention_mask = torch.ones(2, 10, dtype=torch.bool)

        with pytest.raises(ValueError, match="Either input_ids or inputs_embeds must be provided"):
            transformer(attention_mask=attention_mask)

    def test_transformer_properties(self, transformer):
        """Test transformer properties"""
        assert transformer.hidden_dim == 128
        assert transformer.n_layers == 2
        assert transformer.vocab_size == 1000

    def test_transformer_grad_checkpointing(self, transformer):
        """Test gradient checkpointing setting"""
        assert transformer.grad_checkpointing is False

        transformer.set_grad_checkpointing(True)
        assert transformer.grad_checkpointing is True

        transformer.set_grad_checkpointing(False)
        assert transformer.grad_checkpointing is False

    def test_resize_token_embeddings_extend_vocab(self, transformer):
        """Test extending vocabulary with new tokens (HF semantics: arg = new total size)"""
        original_vocab_size = transformer.vocab_size
        original_embedding_weight = transformer.embeddings.weight.data.clone()
        original_output_weight = transformer.output.weight.data.clone()

        # Extend vocabulary by 100 tokens -> new total size is original + 100
        new_num_tokens = original_vocab_size + 100
        result = transformer.resize_token_embeddings(new_num_tokens)

        # Check that vocabulary size was updated to exactly new_num_tokens
        assert transformer.vocab_size == new_num_tokens
        assert result == new_num_tokens

        # Check that embedding layer was resized
        assert transformer.embeddings.num_embeddings == new_num_tokens
        assert transformer.embeddings.embedding_dim == transformer.hidden_dim

        # Check that output layer was resized
        assert transformer.output.out_features == new_num_tokens
        assert transformer.output.in_features == transformer.hidden_dim

        # Check that original weights were preserved
        torch.testing.assert_close(transformer.embeddings.weight.data[:original_vocab_size], original_embedding_weight)
        torch.testing.assert_close(transformer.output.weight.data[:original_vocab_size], original_output_weight)

        # Check that new weights were initialized
        assert transformer.embeddings.weight.data[original_vocab_size:].abs().sum() > 0
        assert transformer.output.weight.data[original_vocab_size:].abs().sum() > 0

    def test_resize_token_embeddings_without_token_id(self, transformer):
        """Test that calling without an argument queries current size and does not modify the model."""
        original_vocab_size = transformer.vocab_size
        original_embedding_weight = transformer.embeddings.weight.data.clone()
        original_output_weight = transformer.output.weight.data.clone()

        result = transformer.resize_token_embeddings()

        # Query-only semantics: no change, returns current vocab size
        assert result == original_vocab_size
        assert transformer.vocab_size == original_vocab_size
        assert transformer.embeddings.num_embeddings == original_vocab_size
        assert transformer.output.out_features == original_vocab_size

        torch.testing.assert_close(transformer.embeddings.weight.data, original_embedding_weight)
        torch.testing.assert_close(transformer.output.weight.data, original_output_weight)

    def test_resize_token_embeddings_existing_token(self, transformer):
        """Test that requesting a size <= current is a no-op and returns the current vocab size."""
        original_vocab_size = transformer.vocab_size
        original_embedding_weight = transformer.embeddings.weight.data.clone()
        original_output_weight = transformer.output.weight.data.clone()

        result = transformer.resize_token_embeddings(original_vocab_size - 1)

        assert result == original_vocab_size
        assert transformer.vocab_size == original_vocab_size
        assert transformer.embeddings.num_embeddings == original_vocab_size
        assert transformer.output.out_features == original_vocab_size

        torch.testing.assert_close(transformer.embeddings.weight.data, original_embedding_weight)
        torch.testing.assert_close(transformer.output.weight.data, original_output_weight)

    def test_resize_token_embeddings_with_weight_tying(self, transformer_config):
        """Test embedding resizing when weight tying is enabled"""
        # Create config with weight tying enabled
        config_dict = transformer_config.__dict__.copy()
        config_dict["weight_tying"] = True
        config_with_tying = TransformerParams(**config_dict)

        transformer = create_model(config_with_tying)

        # Verify weight tying is enabled
        assert transformer.weight_tying is True
        assert transformer.embeddings.weight is transformer.output.weight

        original_vocab_size = transformer.vocab_size
        original_weight = transformer.embeddings.weight.data.clone()

        # Extend vocabulary -> new total size is original + 50
        new_num_tokens = original_vocab_size + 50
        result = transformer.resize_token_embeddings(new_num_tokens)

        assert transformer.vocab_size == new_num_tokens
        assert result == new_num_tokens

        # Check that weight tying is maintained
        assert transformer.embeddings.weight is transformer.output.weight

        # Check that the shared weight tensor was resized
        assert transformer.embeddings.weight.shape == (new_num_tokens, transformer.hidden_dim)
        assert transformer.output.weight.shape == (new_num_tokens, transformer.hidden_dim)

        # Check that original weights were preserved
        torch.testing.assert_close(transformer.embeddings.weight.data[:original_vocab_size], original_weight)

    def test_resize_token_embeddings_forward_pass_after_resize(self, transformer):
        """Test that forward pass works correctly after resizing embeddings"""
        new_num_tokens = transformer.vocab_size + 25
        transformer.resize_token_embeddings(new_num_tokens)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, new_num_tokens, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        assert outputs.logits.shape == (batch_size, seq_len, new_num_tokens)
        assert outputs.past_key_values is None
        assert outputs.hidden_states is None

    def test_resize_token_embeddings_multiple_resizes(self, transformer):
        """Test multiple consecutive embedding resizes"""
        original_vocab_size = transformer.vocab_size

        first_new_size = original_vocab_size + 100
        transformer.resize_token_embeddings(first_new_size)
        assert transformer.vocab_size == first_new_size

        second_new_size = first_new_size + 200
        transformer.resize_token_embeddings(second_new_size)
        assert transformer.vocab_size == second_new_size

        third_new_size = second_new_size + 50
        transformer.resize_token_embeddings(third_new_size)
        assert transformer.vocab_size == third_new_size

        assert transformer.embeddings.num_embeddings == third_new_size
        assert transformer.output.out_features == third_new_size

        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, third_new_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        assert outputs.logits.shape == (batch_size, seq_len, third_new_size)

    def test_resize_token_embeddings_preserves_model_state(self, transformer):
        """Test that resizing preserves other model parameters and state"""
        # Store original model state
        original_hidden_dim = transformer.hidden_dim
        original_n_layers = transformer.n_layers
        original_n_heads = transformer.model_params.n_heads
        original_ffn_type = transformer.model_params.ffn_type

        # Extend vocabulary
        new_token_id = transformer.vocab_size + 75
        transformer.resize_token_embeddings(new_token_id)

        # Check that other model parameters are unchanged
        assert transformer.hidden_dim == original_hidden_dim
        assert transformer.n_layers == original_n_layers
        assert transformer.model_params.n_heads == original_n_heads
        assert transformer.model_params.ffn_type == original_ffn_type

        # Check that transformer layers are unchanged
        assert len(transformer.layers) == original_n_layers
        for layer in transformer.layers:
            assert layer.hidden_dim == original_hidden_dim
            assert layer.n_heads == original_n_heads


class TestTransformerHF:
    @pytest.fixture
    def transformer_hf_config(self):
        # Create a new config with transformer_hf type
        from vla_foundry.params.model_params import TransformerHFParams

        return TransformerHFParams(
            hf_pretrained="microsoft/DialoGPT-small", resume_from_checkpoint=None, resume_weights_only=False
        )

    @patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_forward_basic(self, mock_from_pretrained, transformer_hf_config):
        """Test basic forward pass without hidden states"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        def mock_forward(*args, **kwargs):
            mock_output = Mock()
            mock_output.logits = mock_model.logits
            mock_output.past_key_values = mock_model.past_key_values
            # Set hidden_states based on output_hidden_states parameter
            output_hidden_states = kwargs.get("output_hidden_states", False)
            mock_output.hidden_states = None if not output_hidden_states else mock_model.hidden_states
            return mock_output

        mock_model.return_value = mock_forward()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        assert outputs.logits.shape == (batch_size, seq_len, 1000)
        assert outputs.past_key_values is None
        assert outputs.hidden_states is None

    @patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_forward_with_hidden_states(self, mock_from_pretrained, transformer_hf_config):
        """Test forward pass with hidden states returned"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = tuple(torch.randn(2, 10, 128) for _ in range(2))

        # Mock the forward method to return the expected values
        def mock_forward(*args, **kwargs):
            mock_output = Mock()
            mock_output.logits = mock_model.logits
            mock_output.past_key_values = mock_model.past_key_values
            mock_output.hidden_states = mock_model.hidden_states
            return mock_output

        mock_model.return_value = mock_forward()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        assert outputs.logits.shape == (batch_size, seq_len, 1000)
        assert outputs.past_key_values is None
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) == 2

    @patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_properties(self, mock_from_pretrained, transformer_hf_config):
        """Test transformer HF properties"""
        # Mock the HF model with config
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        assert transformer.hidden_dim == 128
        assert transformer.num_hidden_layers == 2

    @patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained")
    def test_transformer_hf_set_grad_checkpointing(self, mock_from_pretrained, transformer_hf_config):
        """Test that set_grad_checkpointing raises NotImplementedError"""
        # Mock the HF model
        mock_model = Mock()
        mock_from_pretrained.return_value = mock_model

        transformer = create_model(transformer_hf_config)

        with pytest.raises(NotImplementedError):
            transformer.set_grad_checkpointing(True)
