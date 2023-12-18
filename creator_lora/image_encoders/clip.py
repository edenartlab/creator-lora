import clip
from PIL import Image
from typing import List
import torchvision.transforms as transforms
import torch

default_clip_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

class CLIPImageEncoder:
    def __init__(self, name: str = "RN50", device: str = "cpu"):
        clip_model, preprocess = clip.load(
            name,
            device=device,
        )
        self.model = clip_model.visual.float().to(device).eval()
        self.pil_image_preprocess_fn = preprocess
        self.device=device

    @torch.no_grad()
    def encode(self, pil_images: List[Image]):
        """
        obtain embeddings from a list of pil images
        """
        image_tensors = [
            self.pil_image_preprocess_fn(pil_image).unsqueeze(0)
            for pil_image in pil_images
        ]
        
        image_batch = torch.cat(image_tensors, dim = 0)

        embeddings = self.model(image_batch.to(self.device))
        return embeddings