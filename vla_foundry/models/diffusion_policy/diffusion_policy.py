import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.models.diffusion.unet import SinusoidalPositionEmbeddings
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.models.vision_language_backbones import BaseBackboneWrapper
from vla_foundry.params.model_params import DiffusionPolicyParams


class DiffusionPolicy(BaseModel):
    def __init__(
        self,
        model_params: DiffusionPolicyParams,
        vision_language_backbone: BaseBackboneWrapper,
        transformer: Transformer | TransformerHF,
        noise_scheduler: NoiseScheduler,
    ):
        super().__init__(model_params)
        self.vision_language_backbone = vision_language_backbone
        self.transformer = transformer
        self.scheduler = noise_scheduler
        self.proprioception_dim = model_params.proprioception_dim

        backbone_dim = vision_language_backbone.get_conditioning_embeddings_dim()
        self.time_encoding = torch.nn.Embedding(noise_scheduler.num_timesteps, backbone_dim)
        self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(backbone_dim)
        self.output_layer = torch.nn.Linear(transformer.hidden_dim, model_params.action_dim)
        self.action_encode = torch.nn.Linear(model_params.action_dim, transformer.hidden_dim)
        self.condition_encode = torch.nn.Linear(backbone_dim, transformer.hidden_dim)
        self.proprioception_encode = (
            torch.nn.Linear(self.proprioception_dim, transformer.hidden_dim) if self.proprioception_dim > 0 else None
        )

        self.diffusion_step_conditioning = model_params.diffusion_step_conditioning
        self.input_noise_std = model_params.input_noise_std
        self.num_action_head_repeats = model_params.num_action_head_repeats
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize time encoding weights with sinusoidal position embeddings
        timesteps = torch.arange(self.time_encoding.weight.shape[0])
        with torch.no_grad():
            self.time_encoding.weight.copy_(self.sinusoidal_position_embeddings.forward(timesteps))

        # Initialize output layer weights with Xavier initialization
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        if self.proprioception_encode is not None:
            torch.nn.init.xavier_uniform_(self.proprioception_encode.weight)

    def _build_transformer_input(self, backbone_embeddings, time_embeddings, noisy_action, proprio_embeddings=None):
        """Build transformer input by combining conditioning, time, and action embeddings.

        Supports two time conditioning strategies:
        - CONCAT: Prepend time as a separate token [time, backbone] → [B, 1+N, D]
        - ADD: Add time to backbone embeddings element-wise → [B, N, D]

        Args:
            backbone_embeddings: [B, N, backbone_dim] from vision-language backbone
            time_embeddings: [B, 1, backbone_dim] from time encoding
            noisy_action: [B, T, transformer_dim] encoded noisy actions
            proprio_embeddings: Optional [B, P, transformer_dim] encoded proprioception

        Returns:
            transformer_input: [B, C+P+T, transformer_dim]
        """
        if self.diffusion_step_conditioning == "add":
            conditional_embeddings = backbone_embeddings + time_embeddings
        elif self.diffusion_step_conditioning == "concat":
            conditional_embeddings = torch.cat([time_embeddings, backbone_embeddings], dim=1)
        else:
            raise ValueError(f"Unknown diffusion_step_conditioning: {self.diffusion_step_conditioning}")

        conditional_embeddings = self.condition_encode(conditional_embeddings)

        parts = [conditional_embeddings]
        if proprio_embeddings is not None:
            parts.append(proprio_embeddings)
        parts.append(noisy_action)

        return torch.cat(parts, dim=1)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        attention_mask_images,
        actions,
        noise,
        past_mask,
        future_mask,
        proprioception=None,
        **kwargs,
    ):
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (actions.shape[0],)).to(actions.device)  # [bsz]

        # Sample action to denoise
        noisy_action = self.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        noisy_action = torch.where(future_mask.unsqueeze(-1), noisy_action, actions)
        if self.input_noise_std > 0:
            noisy_action = noisy_action + torch.randn_like(noisy_action) * self.input_noise_std
        noisy_action = self.action_encode(noisy_action)

        # Get backbone embeddings (handles text+image concatenation)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        backbone_embeddings = backbone_output.embeddings
        num_repeats = self.num_action_head_repeats
        if num_repeats is not None and num_repeats > 1:
            # Verify action-side inputs were tiled to [B*N] by the batch handler
            vlm_batch_size = input_ids.shape[0]
            assert actions.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected actions batch size {vlm_batch_size * num_repeats} (vlm_batch={vlm_batch_size} * "
                f"num_repeats={num_repeats}), got {actions.shape[0]}"
            )
            assert noise.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected noise batch size {vlm_batch_size * num_repeats}, got {noise.shape[0]}"
            )
            assert future_mask.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected future_mask batch size {vlm_batch_size * num_repeats}, got {future_mask.shape[0]}"
            )
            if proprioception is not None:
                assert proprioception.shape[0] == vlm_batch_size * num_repeats, (
                    f"Expected proprioception batch size {vlm_batch_size * num_repeats}, got {proprioception.shape[0]}"
                )
            # Tile backbone embeddings to match the action batch size [B*N]
            backbone_embeddings = backbone_embeddings.repeat_interleave(num_repeats, dim=0)

        # Time embeddings (batch, 1, backbone_dim) — batch is [B*N] when repeating, else [B]
        time_embeddings = self.time_encoding(timesteps).unsqueeze(1)

        # Proprioception embeddings (already tiled to [B*N] by the batch handler when num_repeats > 1)
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
            if self.input_noise_std > 0:
                proprio_embeddings = proprio_embeddings + torch.randn_like(proprio_embeddings) * self.input_noise_std

        # Build transformer input using time conditioning strategy
        transformer_input = self._build_transformer_input(
            backbone_embeddings=backbone_embeddings,
            time_embeddings=time_embeddings,
            noisy_action=noisy_action,
            proprio_embeddings=proprio_embeddings,
        )

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract predicted direction to denoise the action (B, 1+N+P+T, D) -> (B, T, D)
        action_seq_len = noise.shape[1]
        predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

        return predicted_direction

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        pixel_values,
        actions,
        attention_mask=None,
        attention_mask_images=None,
        num_inference_steps=None,
        past_mask=None,
        proprioception=None,
        **kwargs,  # Ignore extra params like point_cloud (used by other models)
    ):
        """
        Generate actions using iterative denoising through the diffusion process.

        Args:
            input_ids: Text input token IDs
            pixel_values: Input images/pixel values
            actions: Input actions (past timesteps are given in the same sequence, others can be noise)
            attention_mask: Optional attention mask for text
            attention_mask_images: Optional attention mask for camera images
            num_inference_steps: Number of denoising steps (defaults to scheduler.num_timesteps)
            past_mask: Optional mask indicating which actions are from past (1) vs future (0)
            proprioception: Optional proprioception input
            **kwargs: Model-specific args

        Returns:
            Generated actions
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        batch_size = actions.shape[0]
        device = actions.device

        # Precompute backbone embeddings (reused across all denoising steps)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        # Initialize actions with noise (preserve past actions if past_mask provided)
        if past_mask is not None:
            original_past_actions = actions.clone() * past_mask[:, :, None].float()
            # Keep past actions, add noise to future actions
            actions = actions * past_mask[:, :, None].float() + torch.randn_like(actions) * (
                1 - past_mask[:, :, None].float()
            )
        else:
            # All actions are noise
            actions = torch.randn_like(actions)

        # Precompute proprioception embedding
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)

        # Iterative denoising loop
        step_size = max(1, self.scheduler.num_timesteps // num_inference_steps)
        for step in range(self.scheduler.num_timesteps - 1, 0, -step_size):
            # Create timesteps for current step
            timesteps = torch.tensor([step] * batch_size, device=device)
            time_embeddings = self.time_encoding(timesteps).unsqueeze(1)

            # Encode current actions
            action_encoding = self.action_encode(actions)

            # Build transformer input using time conditioning strategy
            transformer_input = self._build_transformer_input(
                backbone_embeddings=backbone_output.embeddings,
                time_embeddings=time_embeddings,
                noisy_action=action_encoding,
                proprio_embeddings=proprio_embeddings,
            )

            # Pass through transformer
            transformer_output = self.transformer(
                inputs_embeds=transformer_input,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract predicted direction to denoise the action
            action_seq_len = actions.shape[1]
            predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

            # Denoise actions using scheduler step
            predicted_actions = self.scheduler.step(predicted_direction, step, actions, step_size=step_size)

            # Preserve past actions if mask provided
            if past_mask is not None:
                past_mask_expanded = past_mask[:, :, None].to(actions.dtype)
                actions = original_past_actions + predicted_actions * (1 - past_mask_expanded)
            else:
                actions = predicted_actions

        return actions
