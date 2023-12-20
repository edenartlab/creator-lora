from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.dataset import (
    PickAPicV2Subset,
    UserContextDataset,
    UserContextCLIPEmbeddingsDataset,
)
import os
from creator_lora.utils import create_new_clean_folder
from creator_lora.dataset import CLIPEmbeddingsDataset
from torch.utils.data import DataLoader
from creator_lora.models.gpt import GPT
import torch
import numpy as np

device = "cuda:0"

parquet_urls = [
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00000-of-00014-387db523fa7e7121.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00001-of-00014-b4d27779c32b8591.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00002-of-00014-5a7a40ba35ff5c70.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00003-of-00014-dddaaa6cb97e4056.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00004-of-00014-bda08a373518160d.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00005-of-00014-823c2ee536bc1a39.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00006-of-00014-a245286301c6ed4a.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00007-of-00014-114287665955838d.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00008-of-00014-e56083416912a7da.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00009-of-00014-d7dfcc392b334773.parquet?download=true",
]

for i in range(0, len(parquet_urls)):
    parquet_filename = os.path.join("downloaded_dataset", f"{i}.pth")
    if i == 0:
        dataset = PickAPicV2Subset.from_url(
            url=parquet_urls[i].replace("?download=true", ""),
            parquet_filename=parquet_filename,
        )
    else:
        dataset.append(
            PickAPicV2Subset.from_url(
                url=parquet_urls[i].replace("?download=true", ""),
                parquet_filename=parquet_filename,
            )
        )

user_context_dataset = UserContextDataset(pick_a_pic_v2_subset=dataset)

image_0_uids = np.unique(dataset.pandas_dataframe.image_0_uid.values)
image_1_uids = np.unique(dataset.pandas_dataframe.image_1_uid.values)

pil_images = []
uids = []

for uid in image_0_uids:
    if uid not in uids:
        pil_images.append(dataset.get_image_from_image_0_uid(image_0_uid=uid))
        uids.append(uid)

for uid in image_1_uids:
    if uid not in uids:
        pil_images.append(dataset.get_image_from_image_1_uid(image_1_uid=uid))
        uids.append(uid)

image_encoder = CLIPImageEncoder(name="RN50", device="cuda:0")
# ## TODO: encode images based on image uid and not dataset index

image_embeddings_folder = "./clip_embeddings"
clip_embeddings_filenames = [
    os.path.join(image_embeddings_folder, f"{uids[count]}.pth")
    for count in range(len(pil_images))
]

# create_new_clean_folder(image_embeddings_folder)

# image_encoder.encode_and_save_batchwise(
#     pil_images=pil_images,
#     output_filenames=clip_embeddings_filenames,
#     batch_size=128,
# )

train_dataset = UserContextCLIPEmbeddingsDataset(
    user_context_dataset=user_context_dataset,
    clip_embeddings=CLIPEmbeddingsDataset(
        filenames=clip_embeddings_filenames, parent_folder=image_embeddings_folder
    ),
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=UserContextCLIPEmbeddingsDataset.collate_fn_with_padding,
)

# from mingpt import GPT
model_config = GPT.get_default_config()
model_config.model_type = None
model_config.vocab_size = 2 
model_config.block_size = int(768 * 2)
model_config.n_layer = 2
model_config.n_embd = 1024
model_config.n_head = 4
model = GPT(model_config)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)

step_idx = 0

for epoch_idx in range(10):
    for batch in train_dataloader:
        optimizer.zero_grad()
        logits = model.forward(
            image_embeddings=batch["image_embeddings"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = model.get_loss(
            logits=logits,
            sequence_lengths=batch["sequence_lengths"],
            labels=batch["labels"].to(device),
        )

        if step_idx % 5 == 0:
            accuracy = model.get_accuracy(
                logits=logits,
                sequence_lengths=batch["sequence_lengths"],
                labels=batch["labels"].to(device),
            )
            print(f"Step: {step_idx + 1} Loss: {loss.item()} Accuracy: {accuracy}")

        loss.backward()
        optimizer.step()
        step_idx += 1
