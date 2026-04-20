import torch

from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

EXPECTED_OUTPUT_TEXT = ["hi[PAD][PAD][PAD]....................", "This is a batch...................."]


def test_inference_text():
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")
    model = create_model(model_params)
    ckpt = "tests/essential/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    ins = tokenizer(["hi", "This is a batch"], return_tensors="pt", padding=True)
    out = model.generate(ins["input_ids"], ins["attention_mask"], temperature=0)  # Greedy for deterministic output
    assert tokenizer.batch_decode(out) == EXPECTED_OUTPUT_TEXT


def test_kv_cache_transformer():
    """Test that KV cache produces identical output to non-cached generation."""
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")
    model = create_model(model_params)
    ckpt = "tests/essential/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    ins = tokenizer(["hello world"], return_tensors="pt", padding=True)

    with torch.no_grad():
        # Generate with KV cache
        out_cached = model.generate(
            ins["input_ids"],
            ins["attention_mask"],
            max_new_tokens=10,
            temperature=0,
            use_cache=True,
        )

        # Generate without KV cache
        out_no_cache = model.generate(
            ins["input_ids"],
            ins["attention_mask"],
            max_new_tokens=10,
            temperature=0,
            use_cache=False,
        )

    assert torch.equal(out_cached, out_no_cache), "KV cache output should match non-cached output"


def test_kv_cache_forward_consistency():
    """Test that forward pass with use_cache returns valid past_key_values."""
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")
    model = create_model(model_params)
    ckpt = "tests/essential/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    ins = tokenizer(["test input"], return_tensors="pt", padding=True)

    with torch.no_grad():
        # First forward pass - full sequence
        outputs1 = model(
            input_ids=ins["input_ids"],
            attention_mask=ins["attention_mask"],
            use_cache=True,
        )

        assert outputs1.past_key_values is not None, "past_key_values should not be None when use_cache=True"
        assert len(outputs1.past_key_values) == model.n_layers, "Should have KV cache for each layer"

        # Each layer's cache should have [keys, values]
        for layer_cache in outputs1.past_key_values:
            assert len(layer_cache) == 2, "Each layer cache should have [keys, values]"
            keys, values = layer_cache
            assert keys.shape[1] == ins["input_ids"].shape[1], "Cached keys should match sequence length"
            assert values.shape[1] == ins["input_ids"].shape[1], "Cached values should match sequence length"

        # Second forward pass - single token with cache
        next_token = torch.tensor([[tokenizer.eos_token_id]])
        extended_mask = torch.cat([ins["attention_mask"], torch.ones(1, 1)], dim=1)

        outputs2 = model(
            input_ids=next_token,
            attention_mask=extended_mask,
            past_key_values=outputs1.past_key_values,
            use_cache=True,
        )

        # Check cache grew by 1 token
        for layer_cache in outputs2.past_key_values:
            keys, values = layer_cache
            assert keys.shape[1] == ins["input_ids"].shape[1] + 1, "Cached keys should grow by 1"
            assert values.shape[1] == ins["input_ids"].shape[1] + 1, "Cached values should grow by 1"


### Need to find a way to make this faster. Probably load a smaller model.
# def test_inference_vlm():
#     cfg = load_experiment_params_from_yaml("s3://your-bucket/your-path/scratch/user/experiment/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/config.json")
#     object.__setattr__(cfg.model, 'processor', 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct')
#     model = create_model(cfg.model)

#     ckpt = "s3://your-bucket/your-path/scratch/user/experiment/vlm_paligemma_3b/2025_06_09-01_54_30-model_vlm-lr_0.0001-bsz_64/checkpoints/checkpoint_1.pt"
#     load_model_checkpoint(model, ckpt, cfg.distributed)


#     import requests
#     from PIL import Image
#     from transformers import AutoProcessor
#     processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")

#     prompt = "<image>"
#     url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
#     image = Image.open(requests.get(url, stream=True).raw)
#     inputs = processor(image, prompt, return_tensors="pt")
#     print(inputs)

#     out = model.generate(input_ids=inputs['input_ids'],
#                           image=inputs['pixel_values'], attention_mask=inputs['attention_mask'])
#     print(out)
#     print(processor.decode(out[0], skip_special_tokens=True))
