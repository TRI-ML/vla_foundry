from unittest.mock import Mock, patch

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast as HFCausalLMOutputWithPast

from vla_foundry.models import create_model
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import compute_num_image_tokens
from vla_foundry.models.vlm import VLM, ModalityProjector
from vla_foundry.models.vlm_hf import VLMHF
from vla_foundry.params.model_params import ModelParams, ViTParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml


class TestModalityProjector:
    @pytest.fixture
    def vit_configs(self):
        return load_params_from_yaml(ViTParams, "tests/essential/params/dummy_configs/dummy_vit_config.yaml")

    @pytest.fixture
    def projector(self, vit_configs):
        output_dim = vit_configs.hidden_dim * (vit_configs.projector_pixel_shuffle_factor**2)
        return ModalityProjector(vit_configs, output_dim)

    def test_modality_projector_forward(self, projector):
        """Test modality projector forward pass"""
        batch_size, seq_len = 2, 16  # 4x4 sequence
        input_tensor = torch.randn(batch_size, seq_len, 100)  # hidden_dim = 100

        output = projector(input_tensor)

        # After pixel shuffle: 100 * 2^2 = 400, then projected to output_dim
        expected_seq_len = seq_len // 4  # 16 // 4 = 4
        assert output.shape == (batch_size, expected_seq_len, 400)

    def test_modality_projector_pixel_shuffle_perfect_square(self, projector):
        """Test pixel shuffle with perfect square sequence length"""
        batch_size, seq_len = 2, 16  # 4x4 sequence
        input_tensor = torch.randn(batch_size, seq_len, 100)

        # Should not raise error
        output = projector(input_tensor)
        assert output.shape[1] == 4  # 16 // 4

    def test_modality_projector_pixel_shuffle_error_not_perfect_square(self, vit_configs):
        """Test pixel shuffle error with non-perfect square sequence length"""
        output_dim = vit_configs.hidden_dim * (vit_configs.projector_pixel_shuffle_factor**2)
        projector = ModalityProjector(vit_configs, output_dim)
        batch_size, seq_len = 2, 15  # Not a perfect square

        input_tensor = torch.randn(batch_size, seq_len, 100)

        with pytest.raises(ValueError, match="perfect-square sequence length"):
            projector(input_tensor)

    def test_modality_projector_pixel_shuffle_error_not_divisible(self):
        """Test pixel shuffle error when sequence root not divisible by scale factor"""
        # Create a config with scale factor 3 (not divisible by 4)
        vit_cfg = load_params_from_yaml(ViTParams, "tests/essential/params/dummy_configs/dummy_vit_config.yaml")
        object.__setattr__(vit_cfg, "projector_pixel_shuffle_factor", 3)
        output_dim = vit_cfg.hidden_dim * (vit_cfg.projector_pixel_shuffle_factor**2)
        projector = ModalityProjector(vit_cfg, output_dim)
        batch_size, seq_len = 2, 16  # 4x4 sequence

        input_tensor = torch.randn(batch_size, seq_len, 100)

        with pytest.raises(ValueError, match="divisible by"):
            projector(input_tensor)


class TestVLM:
    @pytest.fixture
    def vlm_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_config.yaml")

    @pytest.fixture
    def vlm(self, vlm_config):
        vlm = create_model(vlm_config)
        # Use a safe in-vocab token id for image tokens to avoid embedding OOB
        import builtins

        builtins.object.__setattr__(vlm.model_params, "image_token_id", 0)
        return vlm

    def test_vlm_forward_basic(self, vlm):
        """Test basic forward pass without hidden states"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Compute expected number of image tokens based on ViT config
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        pixel_values = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=False
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)  # vocab_size = 1000
        assert output.past_key_values is None
        assert output.hidden_states is None

    def test_vlm_forward_with_hidden_states(self, vlm):
        """Test forward pass with hidden states returned"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set some tokens as image tokens
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        input_ids[0, 0:num_image_tokens] = vlm.model_params.image_token_id
        input_ids[1, 0:num_image_tokens] = vlm.model_params.image_token_id

        pixel_values = torch.randn(batch_size, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        vlm.eval()
        output = vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is not None
        assert isinstance(output.hidden_states, tuple)
        assert len(output.hidden_states) == 2  # n_layers = 2

    def test_vlm_forward_image_token_mismatch(self, vlm):
        """Test error when image token count doesn't match image embedding count"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set wrong number of image tokens
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        wrong = max(1, num_image_tokens // 2)
        input_ids[0, 0:wrong] = vlm.model_params.image_token_id  # mismatch count

        pixel_values = torch.randn(batch_size, 1, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with pytest.raises(AssertionError):
            vlm(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

    def test_vlm_properties(self, vlm):
        """Test VLM properties"""
        assert vlm.hidden_dim == 128  # transformer.hidden_dim = 128
        assert vlm.num_hidden_layers == 2  # transformer.n_layers = 2

    def test_vlm_generate(self, vlm):
        """Test VLM generation"""
        vit_cfg = vlm.model_params.vit
        num_image_tokens = compute_num_image_tokens(vit_cfg)
        batch_size, seq_len = 2, num_image_tokens + 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Make sure input_ids have no image tokens other than the first num_image_tokens
        input_ids = torch.where(
            input_ids == vlm.model_params.image_token_id, vlm.model_params.image_token_id + 1, input_ids
        )
        input_ids[:, 0:num_image_tokens] = vlm.model_params.image_token_id
        image = torch.randn(batch_size, 1, 3, vit_cfg.img_size, vit_cfg.img_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        generated = vlm.generate(input_ids, image, attention_mask, max_new_tokens=5)

        assert generated.shape == (batch_size, seq_len + 5)


class TestVLMHF:
    @pytest.fixture
    def vlm_hf_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_hf_config.yaml")

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_forward_basic(self, mock_from_pretrained, vlm_hf_config):
        """Test basic forward pass without hidden states"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        mock_output = HFCausalLMOutputWithPast(
            logits=torch.randn(2, 10, 1000),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=None,
        )
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=False
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert output.hidden_states is None

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_forward_with_hidden_states(self, mock_from_pretrained, vlm_hf_config):
        """Test forward pass with hidden states returned"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = tuple(torch.randn(2, 10, 128) for _ in range(2))

        # Mock the forward method to return the expected values
        mock_output = HFCausalLMOutputWithPast(
            logits=torch.randn(2, 10, 1000),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=None,
        )
        mock_output.hidden_states = mock_model.hidden_states
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=True
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert isinstance(output.hidden_states, tuple)
        assert len(output.hidden_states) == 2

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_forward_hidden_states_fallback(self, mock_from_pretrained, vlm_hf_config):
        """Test forward pass with hidden states fallback to last_hidden_state"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 10, 1000)
        mock_model.past_key_values = None
        mock_model.hidden_states = tuple(torch.randn(2, 10, 128) for _ in range(2))

        # Mock the forward method to return the expected values
        mock_output = HFCausalLMOutputWithPast(
            logits=torch.randn(2, 10, 1000),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=None,
        )
        mock_output.hidden_states = mock_model.hidden_states
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = vlm(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=True
        )

        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.past_key_values is None
        assert isinstance(output.hidden_states, tuple)
        # Should create hidden states based on num_hidden_layers

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_properties(self, mock_from_pretrained, vlm_hf_config):
        """Test VLM HF properties"""
        # Mock the HF model with config
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test that properties can be accessed
        assert hasattr(vlm, "hidden_dim")
        assert hasattr(vlm, "num_hidden_layers")
        assert isinstance(vlm.hidden_dim, int)
        assert isinstance(vlm.num_hidden_layers, int)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_grad_checkpointing(self, mock_from_pretrained, vlm_hf_config):
        """Test gradient checkpointing methods"""
        # Mock the HF model with gradient checkpointing methods
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model.gradient_checkpointing_disable = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test that gradient checkpointing methods exist and can be called
        vlm.set_grad_checkpointing(True)
        vlm.set_grad_checkpointing(False)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_grad_checkpointing_no_methods(self, mock_from_pretrained, vlm_hf_config):
        """Test gradient checkpointing when methods don't exist"""
        # Mock the HF model without gradient checkpointing methods
        mock_model = Mock()
        # Don't add gradient_checkpointing methods
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Should not raise error even if methods don't exist
        vlm.set_grad_checkpointing(True)
        vlm.set_grad_checkpointing(False)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_generate(self, mock_from_pretrained, vlm_hf_config):
        """Test VLM HF generation"""
        # Mock the HF model
        mock_model = Mock()
        mock_model.logits = torch.randn(2, 15, 1000)  # 10 + 5 new tokens
        mock_model.past_key_values = None

        # Mock the forward method to return the expected values
        mock_output = HFCausalLMOutputWithPast(
            logits=torch.randn(2, 10, 1000),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=None,
        )
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        image = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        generated = vlm.generate(input_ids, image, attention_mask, max_new_tokens=5)

        assert generated.shape == (batch_size, seq_len + 5)


class TestTransformerBase:
    """Test the TransformerBase abstract base class"""

    def test_transformer_base_instantiation_works(self):
        """Test that TransformerBase can be instantiated (it's not truly abstract)"""
        # TransformerBase can be instantiated but will raise NotImplementedError when methods are called
        transformer_base = TransformerBase(Mock())
        assert isinstance(transformer_base, TransformerBase)

    def test_transformer_base_abstract_methods_raise_not_implemented(self):
        """Test that TransformerBase abstract methods raise NotImplementedError when called"""
        transformer_base = TransformerBase(Mock())

        # Test that abstract methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            _ = transformer_base.hidden_dim

        with pytest.raises(NotImplementedError):
            _ = transformer_base.num_hidden_layers

        with pytest.raises(NotImplementedError):
            transformer_base.forward(torch.tensor([1]), torch.tensor([1]))

    def test_transformer_base_interface_methods(self):
        """Test that TransformerBase defines the expected interface methods"""
        # Check that required methods exist
        assert hasattr(TransformerBase, "hidden_dim")
        assert hasattr(TransformerBase, "num_hidden_layers")
        assert hasattr(TransformerBase, "forward")
        assert hasattr(TransformerBase, "resize_token_embeddings")

        # Check that these are properties or methods
        assert isinstance(TransformerBase.hidden_dim, property)
        assert isinstance(TransformerBase.num_hidden_layers, property)
        assert callable(TransformerBase.forward)
        assert callable(TransformerBase.resize_token_embeddings)


class TestVLMInheritance:
    """Test that VLM now properly inherits from TransformerBase"""

    @pytest.fixture
    def vlm_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_config.yaml")

    @pytest.fixture
    def vlm(self, vlm_config):
        vlm = create_model(vlm_config)
        # Use a safe in-vocab token id for image tokens to avoid embedding OOB
        import builtins

        builtins.object.__setattr__(vlm.model_params, "image_token_id", 0)
        return vlm

    def test_vlm_inherits_from_transformer_base(self, vlm):
        """Test that VLM now inherits from TransformerBase instead of BaseModel directly"""
        assert isinstance(vlm, TransformerBase)
        assert isinstance(vlm, BaseModel)  # Should inherit from BaseModel through TransformerBase
        # Check that the inheritance chain is correct
        assert VLM.__mro__.index(TransformerBase) < VLM.__mro__.index(BaseModel)

    def test_vlm_has_transformer_base_methods(self, vlm):
        """Test that VLM has all the methods from TransformerBase"""
        assert hasattr(vlm, "resize_token_embeddings")

        # Test that resize_token_embeddings returns the token_id (default implementation)
        token_id = 1000
        result = vlm.resize_token_embeddings(token_id)
        assert result == token_id


class TestVLMGradientCheckpointing:
    """Test the new gradient checkpointing functionality in VLM"""

    @pytest.fixture
    def vlm_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_config.yaml")

    @pytest.fixture
    def vlm(self, vlm_config):
        vlm = create_model(vlm_config)
        import builtins

        builtins.object.__setattr__(vlm.model_params, "image_token_id", 0)
        return vlm

    def test_vlm_set_grad_checkpointing_calls_underlying_models(self, vlm):
        """Test that set_grad_checkpointing calls the underlying transformer and vit models"""
        # Mock the underlying models' set_grad_checkpointing methods
        vlm.transformer.set_grad_checkpointing = Mock()
        vlm.vit.set_grad_checkpointing = Mock()

        # Call the method
        vlm.set_grad_checkpointing(True)

        # Verify both underlying models were called
        vlm.transformer.set_grad_checkpointing.assert_called_once_with(True)
        vlm.vit.set_grad_checkpointing.assert_called_once_with(True)

        # Test with False
        vlm.set_grad_checkpointing(False)
        assert vlm.transformer.set_grad_checkpointing.call_count == 2
        assert vlm.vit.set_grad_checkpointing.call_count == 2
        vlm.transformer.set_grad_checkpointing.assert_called_with(False)
        vlm.vit.set_grad_checkpointing.assert_called_with(False)


class TestVLMHFInheritance:
    """Test that VLMHF now properly inherits from TransformerBase"""

    @pytest.fixture
    def vlm_hf_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_hf_config.yaml")

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_inherits_from_transformer_base(self, mock_from_pretrained, vlm_hf_config):
        """Test that VLMHF now inherits from TransformerBase instead of BaseModel directly"""
        # Mock the HF model
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        assert isinstance(vlm, TransformerBase)
        assert isinstance(vlm, BaseModel)  # Should inherit from BaseModel through TransformerBase
        # Check that the inheritance chain is correct
        assert VLMHF.__mro__.index(TransformerBase) < VLMHF.__mro__.index(BaseModel)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_has_vlm_base_methods(self, mock_from_pretrained, vlm_hf_config):
        """Test that VLMHF has all the methods from TransformerBase"""
        # Mock the HF model
        mock_model = Mock()
        mock_config_obj = Mock()
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 2
        mock_model.config = mock_config_obj
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        assert hasattr(vlm, "resize_token_embeddings")
        assert hasattr(vlm, "set_grad_checkpointing")
        assert hasattr(vlm, "hidden_dim")
        assert hasattr(vlm, "num_hidden_layers")


class TestVLMHFVocabularyExtension:
    """Test the new vocabulary extension functionality in VLMHF"""

    @pytest.fixture
    def vlm_hf_config(self):
        return load_params_from_yaml(ModelParams, "tests/essential/params/dummy_configs/dummy_vlm_model_hf_config.yaml")

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_resize_token_embeddings_with_token_id(self, mock_from_pretrained, vlm_hf_config):
        """Test resize_token_embeddings with explicit token_id"""
        # Mock the HF model with embeddings
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.num_embeddings = 1000
        mock_model.get_input_embeddings.return_value = mock_embeddings
        mock_model.resize_token_embeddings = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test extending to a larger vocabulary
        new_num_tokens = 1500
        result = vlm.resize_token_embeddings(new_num_tokens)

        assert result == new_num_tokens - 1  # Returns ID of last token
        mock_model.resize_token_embeddings.assert_called_once_with(new_num_tokens, mean_resizing=False)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_resize_token_embeddings_without_token_id(self, mock_from_pretrained, vlm_hf_config):
        """Test resize_token_embeddings without explicit token_id (auto-increment)"""
        # Mock the HF model with embeddings
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.num_embeddings = 1000
        mock_model.get_input_embeddings.return_value = mock_embeddings
        mock_model.resize_token_embeddings = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test auto-increment
        result = vlm.resize_token_embeddings()

        assert result == 1000  # Returns ID of last token (1001 - 1)
        mock_model.resize_token_embeddings.assert_called_once_with(1001, mean_resizing=False)

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_resize_token_embeddings_no_resize_needed(self, mock_from_pretrained, vlm_hf_config):
        """Test resize_token_embeddings when no resize is needed"""
        # Mock the HF model with embeddings
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.num_embeddings = 1000
        mock_model.get_input_embeddings.return_value = mock_embeddings
        mock_model.resize_token_embeddings = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test with new_num_tokens that doesn't require resize
        new_num_tokens = 500  # Less than current vocab size
        result = vlm.resize_token_embeddings(new_num_tokens)

        assert result == new_num_tokens - 1  # Returns ID of last token
        mock_model.resize_token_embeddings.assert_not_called()

    @patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained")
    def test_vlm_hf_resize_token_embeddings_exact_size(self, mock_from_pretrained, vlm_hf_config):
        """Test resize_token_embeddings when token_id equals current vocab size"""
        # Mock the HF model with embeddings
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.num_embeddings = 1000
        mock_model.get_input_embeddings.return_value = mock_embeddings
        mock_model.resize_token_embeddings = Mock()
        mock_from_pretrained.return_value = mock_model

        vlm = create_model(vlm_hf_config)

        # Test with new_num_tokens equal to current vocab size
        new_num_tokens = 1000
        result = vlm.resize_token_embeddings(new_num_tokens)

        assert result == new_num_tokens - 1  # Returns ID of last token
        mock_model.resize_token_embeddings.assert_not_called()
