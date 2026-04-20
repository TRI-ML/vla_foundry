import torch
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from transformers import CLIPTokenizer


class StableDiffusionProcessor:
    def __init__(self, image_size=512, max_length=64, tokenizer_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.image_size = image_size
        self.max_length = max_length
        self.image_token_id = None

        # Transform for PIL images (legacy path)
        self.pil_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Transform for tensors (CHW uint8 from decode_and_augment pipeline)
        self.tensor_transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((self.image_size, self.image_size), antialias=True),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, **sample):
        # Process text
        text_inputs = self.tokenizer(
            sample["text"], padding="max_length", max_length=self.max_length + 1, truncation=True, return_tensors="pt"
        )

        # Process images
        images = sample["images"]
        # Flatten nested lists (e.g. [[img1], [img2]] -> [img1, img2]) from ImageCaptionPipeline
        flat_images = []
        for img in images:
            if isinstance(img, list):
                flat_images.extend(img)
            else:
                flat_images.append(img)

        processed = []
        for img in flat_images:
            if isinstance(img, torch.Tensor):
                processed.append(self.tensor_transform(img))
            else:
                processed.append(self.pil_transform(img))
        pixel_values = torch.stack(processed)

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }
