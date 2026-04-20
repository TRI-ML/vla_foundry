from unittest.mock import Mock, patch

import pytest
import torch

from vla_foundry.models import create_model
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.vlm_hf_backbone import VLMHFBackboneWrapper
from vla_foundry.params.model_params import (
    CLIPBackboneParams,
    DiffusionPolicyParams,
    VLMBackboneParams,
)
from vla_foundry.params.train_experiment_params import load_params_from_yaml


class TestDiffusionPolicy:
    @pytest.fixture
    def diffusion_policy_config(self):
        return load_params_from_yaml(
            DiffusionPolicyParams, "tests/essential/params/dummy_configs/dummy_diffusion_policy_config.yaml"
        )

    @pytest.fixture
    def diffusion_policy(self, diffusion_policy_config):
        with patch("vla_foundry.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_clip_pretrained:
            # Mock the HuggingFace CLIPModel with proper projection_dim
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_clip_pretrained.return_value = mock_hf_clip_model

            # Create the diffusion policy model
            model = create_model(diffusion_policy_config)
            return model

    def _mock_clip_output(self, batch_size, with_text=True, with_image=True):
        """Helper method to create mock CLIP output"""
        mock_clip_output = Mock()
        mock_clip_output.text_embeds = torch.randn(batch_size, 512) if with_text else None
        mock_clip_output.image_embeds = torch.randn(batch_size, 512) if with_image else None
        return mock_clip_output

    def test_diffusion_policy_forward_basic(self, diffusion_policy):
        """Test basic forward pass of diffusion policy"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_forward_without_image_input(self, diffusion_policy):
        """Test forward pass when image input is missing"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = None
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        mock_clip_output = self._mock_clip_output(batch_size, with_text=True, with_image=False)

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_clip_output):
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_forward_without_text_input(self, diffusion_policy):
        """Test forward pass when text input is missing"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        mock_clip_output = self._mock_clip_output(batch_size, with_text=False, with_image=True)

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_clip_output):
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_forward_with_past_actions(self, diffusion_policy):
        """Test forward pass with some past actions (mixed past/future)"""
        batch_size, seq_len = 2, 10
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with mixed past/future
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        # First 5 timesteps are past (no noise), last 5 are future (with noise)
        past_mask = torch.cat(
            [torch.ones(batch_size, 5, dtype=torch.bool), torch.zeros(batch_size, 5, dtype=torch.bool)], dim=1
        )
        future_mask = ~past_mask

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_generate_actions(self, diffusion_policy):
        """Test action generation using iterative denoising"""
        batch_size, seq_len = 2, 8
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        # First half are past actions, second half will be generated
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Generate actions with fewer inference steps for speed
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask_images=attention_mask_images,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=5,
                past_mask=past_mask,
            )

            # Check output shape and that past actions are preserved
            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            # Past actions should be preserved
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )

    def test_diffusion_policy_generate_actions_no_past_mask(self, diffusion_policy):
        """Test action generation without past mask (all actions generated)"""
        batch_size, seq_len = 2, 6
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Generate actions without past mask
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask_images=attention_mask_images,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=3,
            )

            # Check output shape
            assert generated_actions.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_generate_actions_without_image_input(self, diffusion_policy):
        """Test action generation when image input is missing"""
        batch_size, seq_len = 2, 6
        action_dim = diffusion_policy.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = None
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len - seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        mock_clip_output = self._mock_clip_output(batch_size, with_text=True, with_image=False)

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_clip_output):
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask_images=attention_mask_images,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=3,
                past_mask=past_mask,
            )

            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )

    def test_diffusion_policy_generate_actions_without_text_input(self, diffusion_policy):
        """Test action generation when text input is missing"""
        batch_size, seq_len = 2, 6
        action_dim = diffusion_policy.model_params.action_dim

        input_ids = None
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        attention_mask = None
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len - seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        mock_clip_output = self._mock_clip_output(batch_size, with_text=False, with_image=True)

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_clip_output):
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask_images=attention_mask_images,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=3,
                past_mask=past_mask,
            )

            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )

    def test_diffusion_policy_weight_initialization(self, diffusion_policy):
        """Test that weight initialization works correctly"""
        # Check that time encoding weights were initialized with sinusoidal embeddings
        time_encoding_weights = diffusion_policy.time_encoding.weight
        assert time_encoding_weights.shape == (100, 512)  # num_timesteps x projection_dim

        # Check that output layer was initialized with Xavier
        output_weights = diffusion_policy.output_layer.weight
        assert output_weights.shape == (
            diffusion_policy.model_params.action_dim,
            diffusion_policy.transformer.hidden_dim,
        )

        # Weights should not be zero (indicating initialization occurred)
        assert time_encoding_weights.abs().sum() > 0
        assert output_weights.abs().sum() > 0

    def test_diffusion_policy_model_components(self, diffusion_policy):
        """Test that all model components are properly initialized"""
        # Check that all components exist
        assert hasattr(diffusion_policy, "vision_language_backbone")
        assert hasattr(diffusion_policy, "transformer")
        assert hasattr(diffusion_policy, "scheduler")
        assert hasattr(diffusion_policy, "time_encoding")
        assert hasattr(diffusion_policy, "sinusoidal_position_embeddings")
        assert hasattr(diffusion_policy, "output_layer")
        assert hasattr(diffusion_policy, "action_encode")
        assert hasattr(diffusion_policy, "condition_encode")

        # Check dimensions
        assert diffusion_policy.time_encoding.num_embeddings == 100  # num_timesteps
        assert diffusion_policy.time_encoding.embedding_dim == 512  # clip projection_dim
        assert diffusion_policy.output_layer.in_features == 128  # transformer hidden_dim
        assert diffusion_policy.output_layer.out_features == 7  # action_dim
        assert diffusion_policy.action_encode.in_features == 7  # action_dim
        assert diffusion_policy.action_encode.out_features == 128  # transformer hidden_dim
        assert diffusion_policy.condition_encode.in_features == 512  # clip projection_dim
        assert diffusion_policy.condition_encode.out_features == 128  # transformer hidden_dim

    def test_diffusion_policy_conditional_embeddings_shape(self, diffusion_policy):
        """Test that conditional embeddings are created with correct shapes"""
        batch_size = 2

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Test time embeddings
            timesteps = torch.randint(0, 100, (batch_size,))
            time_embeddings = diffusion_policy.time_encoding(timesteps)
            assert time_embeddings.shape == (batch_size, 512)

            # Test that image embeddings can be handled in different shapes
            mock_output = self._mock_clip_output(batch_size)
            # 2D case (batch_size, features)
            if mock_output.image_embeds.ndim == 2:
                image_embeddings = mock_output.image_embeds.unsqueeze(1)
                assert image_embeddings.shape == (batch_size, 1, 512)

    def test_diffusion_policy_transformer_integration(self, diffusion_policy):
        """Test integration with transformer component"""
        batch_size, seq_len = 2, 8
        action_dim = diffusion_policy.model_params.action_dim

        # Create mock transformer input
        conditional_embeddings = torch.randn(batch_size, 3, 128)  # time + text + image
        noisy_action = torch.randn(batch_size, seq_len, 128)
        transformer_input = torch.cat([conditional_embeddings, noisy_action], dim=1)

        # Test transformer forward pass
        transformer_output = diffusion_policy.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
            is_causal=False,
        )

        # Check that transformer returns expected structure
        assert hasattr(transformer_output, "hidden_states")
        assert transformer_output.hidden_states is not None
        assert len(transformer_output.hidden_states) == 2  # n_layers

        # Check that we can extract action predictions
        action_seq_len = seq_len
        predicted_direction = diffusion_policy.output_layer(
            transformer_output.hidden_states[-1][:, -action_seq_len:, :]
        )
        assert predicted_direction.shape == (batch_size, seq_len, action_dim)

    def test_diffusion_policy_noise_scheduler_integration(self, diffusion_policy):
        """Test integration with noise scheduler"""
        batch_size, seq_len = 2, 5
        action_dim = diffusion_policy.model_params.action_dim

        # Create test data
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        timesteps = torch.randint(0, 100, (batch_size,))
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test noise addition
        noisy_action = diffusion_policy.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        assert noisy_action.shape == actions.shape

        # Test scheduler step (for generation)
        predicted_direction = torch.randn(batch_size, seq_len, action_dim)
        denoised_action = diffusion_policy.scheduler.step(predicted_direction, 50, actions, step_size=1)
        assert denoised_action.shape == actions.shape

    def test_diffusion_policy_edge_cases(self, diffusion_policy):
        """Test edge cases and error conditions"""
        batch_size = 1

        # Test with minimal sequence length
        input_ids = torch.randint(0, 1000, (batch_size, 1))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, 1, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, 1, 7)
        noise = torch.randn(batch_size, 1, 7)
        past_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        future_mask = torch.ones(batch_size, 1, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Should not raise error with minimal input
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )
            assert output.shape == (batch_size, 1, 7)

    def test_diffusion_policy_device_consistency(self, diffusion_policy):
        """Test that model handles device placement correctly"""
        batch_size, seq_len = 2, 4
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Test that timesteps are moved to correct device
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Output should be on same device as input
            assert output.device == actions.device

    def test_diffusion_policy_gradient_flow(self, diffusion_policy):
        """Test that gradients flow properly through the model"""
        batch_size, seq_len = 2, 4
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with gradient tracking
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim, requires_grad=True)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Mock the CLIP forward call
        with patch.object(
            diffusion_policy.vision_language_backbone._model,
            "forward",
            return_value=self._mock_clip_output(batch_size),
        ):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Compute loss and backpropagate
            loss = output.sum()
            loss.backward()

            # Check that gradients exist for key parameters
            assert diffusion_policy.output_layer.weight.grad is not None
            assert diffusion_policy.action_encode.weight.grad is not None
            assert diffusion_policy.condition_encode.weight.grad is not None

    def test_diffusion_policy_attention_mask_images_single_image(self, diffusion_policy):
        """Test attention_mask_images with single image (4D pixel_values)"""
        batch_size, seq_len = 2, 6
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with 4D pixel_values (single image per sample)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # attention_mask_images with shape [B, 1] for single image
        attention_mask_images = torch.tensor([[True], [False]], dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Create a mock that properly handles attention_mask_images
        def mock_clip_forward(**kwargs):
            mock_output = Mock()
            mock_output.text_embeds = torch.randn(batch_size, 512)
            # Simulate masking in image embeddings
            image_embeds = torch.randn(batch_size, 512)
            if kwargs.get("attention_mask_images") is not None:
                # Zero out embeddings where mask is False
                image_embeds = image_embeds * kwargs["attention_mask_images"].squeeze(-1).unsqueeze(-1)
            mock_output.image_embeds = image_embeds
            return mock_output

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", side_effect=mock_clip_forward):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_attention_mask_images_multiple_images(self, diffusion_policy):
        """Test attention_mask_images with multiple images per sample (5D pixel_values)"""
        batch_size, seq_len = 2, 6
        num_images = 3
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with 5D pixel_values (multiple images per sample)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, num_images, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Mask out the second image in first sample, and first image in second sample
        attention_mask_images = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Create a mock that properly handles multiple images and attention_mask_images
        def mock_clip_forward(**kwargs):
            mock_output = Mock()
            mock_output.text_embeds = torch.randn(batch_size, 512)
            # Simulate multiple image embeddings [B, N, D]
            image_embeds = torch.randn(batch_size, num_images, 512)
            if kwargs.get("attention_mask_images") is not None:
                # Zero out embeddings where mask is False
                image_embeds = image_embeds * kwargs["attention_mask_images"].unsqueeze(-1)
            mock_output.image_embeds = image_embeds
            return mock_output

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", side_effect=mock_clip_forward):
            # Forward pass
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_attention_mask_images_all_masked(self, diffusion_policy):
        """Test attention_mask_images when all images are masked out"""
        batch_size, seq_len = 2, 6
        num_images = 2
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with all images masked
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, num_images, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Mask out all images
        attention_mask_images = torch.zeros(batch_size, num_images, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Create a mock that properly handles all masked images
        def mock_clip_forward(**kwargs):
            mock_output = Mock()
            mock_output.text_embeds = torch.randn(batch_size, 512)
            # Simulate multiple image embeddings [B, N, D] - all zeros after masking
            image_embeds = torch.randn(batch_size, num_images, 512)
            if kwargs.get("attention_mask_images") is not None:
                # Zero out embeddings where mask is False (all of them)
                image_embeds = image_embeds * kwargs["attention_mask_images"].unsqueeze(-1)
            mock_output.image_embeds = image_embeds
            return mock_output

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", side_effect=mock_clip_forward):
            # Forward pass should work even when all images are masked
            output = diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            # Check output shape
            assert output.shape == (batch_size, seq_len, action_dim)
            assert output.dtype == torch.float32

    def test_diffusion_policy_generate_actions_with_attention_mask_images(self, diffusion_policy):
        """Test action generation with attention_mask_images"""
        batch_size, seq_len = 2, 6
        num_images = 2
        action_dim = diffusion_policy.model_params.action_dim

        # Create input tensors with multiple images and attention mask
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, num_images, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Only use first image in both samples
        attention_mask_images = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        # Create a mock that properly handles multiple images and attention_mask_images
        def mock_clip_forward(**kwargs):
            mock_output = Mock()
            mock_output.text_embeds = torch.randn(batch_size, 512)
            # Simulate multiple image embeddings [B, N, D]
            image_embeds = torch.randn(batch_size, num_images, 512)
            if kwargs.get("attention_mask_images") is not None:
                # Zero out embeddings where mask is False
                image_embeds = image_embeds * kwargs["attention_mask_images"].unsqueeze(-1)
            mock_output.image_embeds = image_embeds
            return mock_output

        with patch.object(diffusion_policy.vision_language_backbone._model, "forward", side_effect=mock_clip_forward):
            # Generate actions with attention_mask_images
            generated_actions = diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask_images=attention_mask_images,
                actions=actions,
                attention_mask=attention_mask,
                num_inference_steps=3,
                past_mask=past_mask,
            )

            # Check output shape and that past actions are preserved
            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )

    # VLM specific test fixtures.
    @pytest.fixture
    def vlm_diffusion_policy_config(self):
        return load_params_from_yaml(
            DiffusionPolicyParams, "tests/essential/params/dummy_configs/dummy_vla_diffusion_policy_config.yaml"
        )

    @pytest.fixture
    def vlm_diffusion_policy(self, vlm_diffusion_policy_config):
        with patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained") as mock_vlm_hf_pretrained:
            mock_vlm_hf_model = Mock()
            mock_vlm_hf_model.get_input_embeddings = Mock(
                return_value=Mock(num_embeddings=1000, weight=torch.randn(1000, 2048))
            )
            mock_vlm_hf_model.resize_token_embeddings = lambda new_num_tokens, mean_resizing: new_num_tokens
            mock_vlm_hf_model.language_model = Mock(config=Mock(hidden_size=2048))
            mock_vlm_hf_model.config = Mock(model_type="", vision_config=None)
            mock_vlm_hf_pretrained.return_value = mock_vlm_hf_model

            # Create the diffusion policy model
            model = create_model(vlm_diffusion_policy_config)
            return model

    def _mock_vlm_hf_output(self, batch_size, seq_len, hidden_dim=2048, num_layers=4):
        mock_vlm_hf_output = Mock()
        mock_vlm_hf_output.hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)]
        return mock_vlm_hf_output

    # VLM-specific tests
    def test_vlm_diffusion_policy_forward_basic(self, vlm_diffusion_policy):
        """Test forward pass with VLM backbone."""
        batch_size, seq_len = 2, 10
        action_dim = vlm_diffusion_policy.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask_images = None
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        mock_output = self._mock_vlm_hf_output(batch_size, seq_len + 1)
        with patch.object(vlm_diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_output):
            output = vlm_diffusion_policy(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=attention_mask_images,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

        assert output.shape == (batch_size, seq_len, action_dim)

    def test_vlm_diffusion_policy_generate_actions(self, vlm_diffusion_policy):
        """Test action generation with VLM backbone."""
        batch_size, seq_len = 2, 6
        action_dim = vlm_diffusion_policy.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        actions = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        mock_output = self._mock_vlm_hf_output(batch_size, seq_len + 1)
        with patch.object(vlm_diffusion_policy.vision_language_backbone._model, "forward", return_value=mock_output):
            generated_actions = vlm_diffusion_policy.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                actions=actions,
                num_inference_steps=5,
                past_mask=past_mask,
            )

        assert generated_actions.shape == (batch_size, seq_len, action_dim)
        torch.testing.assert_close(
            generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
        )

    def test_vlm_diffusion_policy_model_components(self, vlm_diffusion_policy):
        """Test VLM policy uses backbone embedding dimensions."""
        backbone = vlm_diffusion_policy.vision_language_backbone
        expected_dim = backbone.get_conditioning_embeddings_dim()

        assert vlm_diffusion_policy.time_encoding.embedding_dim == expected_dim
        assert vlm_diffusion_policy.condition_encode.in_features == expected_dim

    def test_build_transformer_input_concat(self, diffusion_policy):
        """Test CONCAT time conditioning build path."""
        batch_size, seq_len = 2, 6
        original_diffusion_step_conditioning = diffusion_policy.diffusion_step_conditioning
        diffusion_policy.diffusion_step_conditioning = "concat"
        backbone_embeddings = VisionLanguageBackboneOutput(
            embeddings=torch.randn(batch_size, 2, diffusion_policy.time_encoding.embedding_dim)
        ).embeddings
        timesteps = torch.randint(0, diffusion_policy.scheduler.num_timesteps, (batch_size,))
        time_embeddings = diffusion_policy.time_encoding(timesteps).unsqueeze(1)
        noisy_action = torch.randn(batch_size, seq_len, diffusion_policy.transformer.hidden_dim)

        diffusion_policy._build_transformer_input(
            backbone_embeddings=backbone_embeddings,
            time_embeddings=time_embeddings,
            noisy_action=noisy_action,
        )
        diffusion_policy.diffusion_step_conditioning = original_diffusion_step_conditioning

    # ------------------------------------------------------------------
    # num_action_head_repeats tests
    # ------------------------------------------------------------------

    @pytest.fixture
    def diffusion_policy_with_repeats(self, diffusion_policy_config):
        """Diffusion policy fixture with num_action_head_repeats=3."""
        object.__setattr__(diffusion_policy_config, "num_action_head_repeats", 3)
        with patch("vla_foundry.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_clip_pretrained:
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_clip_pretrained.return_value = mock_hf_clip_model
            return create_model(diffusion_policy_config)

    def test_forward_with_num_action_head_repeats_output_shape(self, diffusion_policy_with_repeats):
        """forward() with [B*N] action inputs should return [B*N, T, A] and call the backbone once."""
        num_repeats = 3
        vlm_batch_size, seq_len = 2, 10
        action_batch_size = vlm_batch_size * num_repeats
        action_dim = diffusion_policy_with_repeats.model_params.action_dim
        backbone_dim = 512  # projection_dim from mock

        # VLM inputs at [B]
        input_ids = torch.randint(0, 1000, (vlm_batch_size, seq_len))
        pixel_values = torch.randn(vlm_batch_size, 3, 224, 224)
        attention_mask = torch.ones(vlm_batch_size, seq_len, dtype=torch.bool)

        # Action-side inputs tiled to [B*N] (as the batch handler would produce)
        actions = torch.randn(action_batch_size, seq_len, action_dim)
        noise = torch.randn(action_batch_size, seq_len, action_dim)
        past_mask = torch.zeros(action_batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(action_batch_size, seq_len, dtype=torch.bool)

        mock_backbone_output = VisionLanguageBackboneOutput(embeddings=torch.randn(vlm_batch_size, 2, backbone_dim))
        with patch.object(
            diffusion_policy_with_repeats.vision_language_backbone,
            "get_action_conditioning",
            return_value=mock_backbone_output,
        ) as mock_get_conditioning:
            output = diffusion_policy_with_repeats(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=None,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

        assert output.shape == (action_batch_size, seq_len, action_dim)
        # Backbone must be called exactly once regardless of num_repeats
        mock_get_conditioning.assert_called_once()

    def test_forward_with_num_action_head_repeats_asserts_batch_size_mismatch(self, diffusion_policy_with_repeats):
        """forward() should raise AssertionError when action inputs are not tiled to [B*N]."""
        vlm_batch_size, seq_len = 2, 10
        action_dim = diffusion_policy_with_repeats.model_params.action_dim
        backbone_dim = 512

        # VLM inputs at [B], action inputs NOT tiled (still [B] — wrong)
        input_ids = torch.randint(0, 1000, (vlm_batch_size, seq_len))
        actions = torch.randn(vlm_batch_size, seq_len, action_dim)
        noise = torch.randn(vlm_batch_size, seq_len, action_dim)
        past_mask = torch.zeros(vlm_batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(vlm_batch_size, seq_len, dtype=torch.bool)

        mock_backbone_output = VisionLanguageBackboneOutput(embeddings=torch.randn(vlm_batch_size, 2, backbone_dim))
        with (
            patch.object(
                diffusion_policy_with_repeats.vision_language_backbone,
                "get_action_conditioning",
                return_value=mock_backbone_output,
            ),
            pytest.raises(AssertionError, match="Expected actions batch size"),
        ):
            diffusion_policy_with_repeats(
                input_ids=input_ids,
                pixel_values=None,
                attention_mask=None,
                attention_mask_images=None,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )


class TestBuildTransformerInput:
    """Test _build_transformer_input with different time conditioning strategies."""

    @pytest.fixture
    def diffusion_policy_concat(self):
        config = load_params_from_yaml(
            DiffusionPolicyParams, "tests/essential/params/dummy_configs/dummy_diffusion_policy_config.yaml"
        )
        assert config.diffusion_step_conditioning == "concat"
        with patch("vla_foundry.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_clip:
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_clip.return_value = mock_hf_clip_model
            return create_model(config)

    @pytest.fixture
    def diffusion_policy_add(self):
        config = load_params_from_yaml(
            DiffusionPolicyParams, "tests/essential/params/dummy_configs/dummy_diffusion_policy_config.yaml"
        )
        object.__setattr__(config, "diffusion_step_conditioning", "add")
        with patch("vla_foundry.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_clip:
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_clip.return_value = mock_hf_clip_model
            return create_model(config)

    def test_concat_prepends_time_token(self, diffusion_policy_concat):
        """CONCAT strategy: time is prepended as a separate token, increasing sequence length by 1."""
        model = diffusion_policy_concat
        batch_size, num_backbone_tokens, backbone_dim = 2, 3, 512
        action_seq_len, transformer_dim = 5, 128

        backbone_embeddings = torch.randn(batch_size, num_backbone_tokens, backbone_dim)
        time_embeddings = torch.randn(batch_size, 1, backbone_dim)
        noisy_action = torch.randn(batch_size, action_seq_len, transformer_dim)

        result = model._build_transformer_input(backbone_embeddings, time_embeddings, noisy_action)

        # CONCAT: (1 + N) conditioning tokens + T action tokens
        expected_seq_len = 1 + num_backbone_tokens + action_seq_len
        assert result.shape == (batch_size, expected_seq_len, transformer_dim)

    def test_add_broadcasts_time(self, diffusion_policy_add):
        """ADD strategy: time is added element-wise, preserving backbone sequence length."""
        model = diffusion_policy_add
        batch_size, num_backbone_tokens, backbone_dim = 2, 3, 512
        action_seq_len, transformer_dim = 5, 128

        backbone_embeddings = torch.randn(batch_size, num_backbone_tokens, backbone_dim)
        time_embeddings = torch.randn(batch_size, 1, backbone_dim)
        noisy_action = torch.randn(batch_size, action_seq_len, transformer_dim)

        result = model._build_transformer_input(backbone_embeddings, time_embeddings, noisy_action)

        # ADD: N conditioning tokens + T action tokens (no extra time token)
        expected_seq_len = num_backbone_tokens + action_seq_len
        assert result.shape == (batch_size, expected_seq_len, transformer_dim)

    def test_concat_with_proprioception(self, diffusion_policy_concat):
        """CONCAT with proprioception: (1+N) conditioning + P proprio + T action tokens."""
        model = diffusion_policy_concat
        batch_size, num_backbone_tokens, backbone_dim = 2, 2, 512
        proprio_seq_len, action_seq_len, transformer_dim = 3, 5, 128

        backbone_embeddings = torch.randn(batch_size, num_backbone_tokens, backbone_dim)
        time_embeddings = torch.randn(batch_size, 1, backbone_dim)
        noisy_action = torch.randn(batch_size, action_seq_len, transformer_dim)
        proprio_embeddings = torch.randn(batch_size, proprio_seq_len, transformer_dim)

        result = model._build_transformer_input(
            backbone_embeddings, time_embeddings, noisy_action, proprio_embeddings=proprio_embeddings
        )

        expected_seq_len = 1 + num_backbone_tokens + proprio_seq_len + action_seq_len
        assert result.shape == (batch_size, expected_seq_len, transformer_dim)

    def test_add_with_proprioception(self, diffusion_policy_add):
        """ADD with proprioception: N conditioning + P proprio + T action tokens."""
        model = diffusion_policy_add
        batch_size, num_backbone_tokens, backbone_dim = 2, 2, 512
        proprio_seq_len, action_seq_len, transformer_dim = 3, 5, 128

        backbone_embeddings = torch.randn(batch_size, num_backbone_tokens, backbone_dim)
        time_embeddings = torch.randn(batch_size, 1, backbone_dim)
        noisy_action = torch.randn(batch_size, action_seq_len, transformer_dim)
        proprio_embeddings = torch.randn(batch_size, proprio_seq_len, transformer_dim)

        result = model._build_transformer_input(
            backbone_embeddings, time_embeddings, noisy_action, proprio_embeddings=proprio_embeddings
        )

        expected_seq_len = num_backbone_tokens + proprio_seq_len + action_seq_len
        assert result.shape == (batch_size, expected_seq_len, transformer_dim)

    def test_add_single_token_backbone(self, diffusion_policy_add):
        """ADD with single-token backbone (typical VLM case): time adds to single embedding."""
        model = diffusion_policy_add
        batch_size, backbone_dim = 2, 512
        action_seq_len, transformer_dim = 5, 128

        backbone_embeddings = torch.randn(batch_size, 1, backbone_dim)
        time_embeddings = torch.randn(batch_size, 1, backbone_dim)
        noisy_action = torch.randn(batch_size, action_seq_len, transformer_dim)

        result = model._build_transformer_input(backbone_embeddings, time_embeddings, noisy_action)

        # ADD with 1 backbone token: 1 conditioning + T action tokens
        expected_seq_len = 1 + action_seq_len
        assert result.shape == (batch_size, expected_seq_len, transformer_dim)

    def test_add_forward_pass(self, diffusion_policy_add):
        """Test full forward pass with ADD time conditioning."""
        model = diffusion_policy_add
        batch_size, seq_len = 2, 10
        action_dim = model.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        future_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        mock_output = Mock()
        mock_output.text_embeds = torch.randn(batch_size, 512)
        mock_output.image_embeds = torch.randn(batch_size, 512)

        with patch.object(model.vision_language_backbone._model, "forward", return_value=mock_output):
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=None,
                actions=actions,
                noise=noise,
                past_mask=past_mask,
                future_mask=future_mask,
            )

            assert output.shape == (batch_size, seq_len, action_dim)

    def test_add_generate_actions(self, diffusion_policy_add):
        """Test action generation with ADD time conditioning."""
        model = diffusion_policy_add
        batch_size, seq_len = 2, 8
        action_dim = model.model_params.action_dim

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, seq_len, action_dim)
        past_mask = torch.cat(
            [
                torch.ones(batch_size, seq_len // 2, dtype=torch.bool),
                torch.zeros(batch_size, seq_len // 2, dtype=torch.bool),
            ],
            dim=1,
        )

        mock_output = Mock()
        mock_output.text_embeds = torch.randn(batch_size, 512)
        mock_output.image_embeds = torch.randn(batch_size, 512)

        with patch.object(model.vision_language_backbone._model, "forward", return_value=mock_output):
            generated_actions = model.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                attention_mask_images=None,
                actions=actions,
                num_inference_steps=3,
                past_mask=past_mask,
            )

            assert generated_actions.shape == (batch_size, seq_len, action_dim)
            torch.testing.assert_close(
                generated_actions[:, : seq_len // 2], actions[:, : seq_len // 2], rtol=1e-5, atol=1e-5
            )


class TestVisionLanguageBackbones:
    """Test the backbone wrapper interface."""

    @pytest.fixture
    def clip_backbone(self):
        from vla_foundry.models.vision_language_backbones import CLIPBackboneWrapper

        clip_params = CLIPBackboneParams(type="clip_backbone", hf_pretrained="openai/clip-vit-base-patch32")
        with patch("vla_foundry.models.diffusion_policy.clip_hf.CLIPModel.from_pretrained") as mock_pretrained:
            mock_hf_clip_model = Mock()
            mock_hf_clip_model.projection_dim = 512
            mock_pretrained.return_value = mock_hf_clip_model
            return CLIPBackboneWrapper(clip_params, load_pretrained=True)

    @pytest.fixture
    def vlm_backbone(self):
        with patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained") as mock_vlm:
            mock_hf_model = Mock()
            mock_hf_model.get_input_embeddings = Mock(
                return_value=Mock(num_embeddings=1000, weight=torch.randn(1000, 2048))
            )
            mock_hf_model.resize_token_embeddings = lambda new_num_tokens, mean_resizing: new_num_tokens
            mock_hf_model.language_model = Mock(config=Mock(hidden_size=2048))
            mock_hf_model.config = Mock(model_type="", vision_config=None)
            mock_vlm.return_value = mock_hf_model

            vlm_params = VLMBackboneParams(
                hf_pretrained="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", num_vlm_layers_to_use=2
            )
            return VLMHFBackboneWrapper(vlm_params, load_pretrained=True)

    def test_clip_backbone_get_action_conditioning(self, clip_backbone):
        """Test that CLIP backbone returns correct conditioning embeddings."""
        batch_size = 2
        mock_output = Mock()
        mock_output.text_embeds = torch.randn(batch_size, 512)
        mock_output.image_embeds = torch.randn(batch_size, 512)

        with patch.object(clip_backbone._model, "forward", return_value=mock_output):
            result = clip_backbone.get_action_conditioning(
                input_ids=torch.randint(0, 100, (batch_size, 5)),
                pixel_values=torch.randn(batch_size, 3, 224, 224),
            )

            # With text+image: [B, 2, D] (text token + image token)
            assert result.embeddings.shape == (batch_size, 2, 512)

    def test_clip_backbone_disable_text(self, clip_backbone):
        """Test that disable_text removes text from conditioning."""
        clip_backbone.disable_text = True
        batch_size = 2
        mock_output = Mock()
        mock_output.text_embeds = torch.randn(batch_size, 512)
        mock_output.image_embeds = torch.randn(batch_size, 512)

        with patch.object(clip_backbone._model, "forward", return_value=mock_output):
            output = clip_backbone.get_action_conditioning(
                input_ids=torch.randint(0, 100, (batch_size, 5)),
                pixel_values=torch.randn(batch_size, 3, 224, 224),
            )

        assert output.embeddings.shape == (batch_size, 1, 512)

    def test_vlm_backbone_action_token_and_concat(self, vlm_backbone):
        """Test VLM backbone appends action token and concatenates layers."""
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        captured = {}

        def mock_vlm_forward(*args, **kwargs):
            captured["input_ids"] = kwargs["input_ids"]
            seq_len = captured["input_ids"].shape[1]
            mock_output = Mock()
            mock_output.hidden_states = [
                torch.randn(batch_size, seq_len, 2048) for _ in range(vlm_backbone.num_vlm_layers_to_use)
            ]
            return mock_output

        with patch.object(vlm_backbone._model, "forward", side_effect=mock_vlm_forward):
            output = vlm_backbone.get_action_conditioning(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=None,
                attention_mask_images=None,
            )

        assert captured["input_ids"].shape[1] == input_ids.shape[1] + 1
        assert output.embeddings.shape == (batch_size, 1, 2048 * vlm_backbone.num_vlm_layers_to_use)
