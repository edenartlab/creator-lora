import clip
from PIL import Image
from typing import List
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import os

default_clip_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def chunk_list(list_to_be_chunked, chunk_size):
    """
    splits a list into chunks of size n
    """
    chunk_size = max(1, chunk_size)
    return list(
        (
            list_to_be_chunked[i : i + chunk_size]
            for i in range(0, len(list_to_be_chunked), chunk_size)
        )
    )


class CLIPImageEncoder:
    def __init__(self, name: str = "RN50", device: str = "cpu"):
        clip_model, preprocess = clip.load(
            name,
            device=device,
        )
        self.model = clip_model.visual.float().to(device).eval()
        self.pil_image_preprocess_fn = preprocess
        self.device = device

    @torch.no_grad()
    def encode(self, pil_images: List[Image], batch_size: int, progress=True):
        """
        obtain embeddings from a list of pil images
        """

        chunked_pil_images = chunk_list(
            list_to_be_chunked=pil_images, chunk_size=batch_size
        )
        all_embeddings = []
        for pil_images in tqdm(
            chunked_pil_images, disable=not (progress), total=len(chunked_pil_images)
        ):
            image_tensors = [
                self.pil_image_preprocess_fn(pil_image).unsqueeze(0)
                for pil_image in pil_images
            ]
            image_batch = torch.cat(image_tensors, dim=0)
            embeddings = self.model(image_batch.to(self.device))
            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def encode_and_save_batchwise(
        self,
        pil_images,
        output_filenames: List[str],
        batch_size: int,
        progress=True,
        skip_if_exists=True,
    ):
        chunked_pil_images = chunked_pil_images = chunk_list(
            list_to_be_chunked=pil_images, chunk_size=batch_size
        )

        count = 0
        for pil_images in tqdm(
            chunked_pil_images,
            disable=not (progress),
            total=len(chunked_pil_images),
            desc="Saving CLIP embeddings",
        ):
            ## assume all images exist if first and last images in batch exist
            if (
                os.path.exists(output_filenames[0])
                and os.path.exists(output_filenames[-1])
                and skip_if_exists
            ):
                print(f"Skipping batch...")
                count += len(pil_images)
                continue

            all_embeddings = self.encode(
                pil_images=pil_images, batch_size=batch_size, progress=False
            )

            for idx in range(all_embeddings.shape[0]):
                torch.save(
                    all_embeddings[idx, :].unsqueeze(0), f=output_filenames[count]
                )
                count += 1
