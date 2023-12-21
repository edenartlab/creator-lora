from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.dataset import (
    PickAPicV2Subset,
    UserContextDataset,
    UserContextCLIPEmbeddingsDataset,
)
import os
from creator_lora.utils import get_filenames_in_a_folder
from creator_lora.dataset import CLIPEmbeddingsDataset
from torch.utils.data import DataLoader
from creator_lora.models.gpt import GPT
import torch
import numpy as np
from creator_lora.dataset import save_all_unique_images_from_pick_a_pic_v2_subset
from tqdm import tqdm
from creator_lora.utils.json_stuff import load_json

device = "cuda:0"

for i in tqdm(range(0, 55), desc = "Loading data"):
    parquet_filename = os.path.join("downloaded_dataset", f"{i}.pth")
    if i == 0:
        dataset = PickAPicV2Subset(
            parquet_filename=parquet_filename,
        )
    else:
        dataset.append(
            PickAPicV2Subset(
                parquet_filename=parquet_filename,
            )
        )

user_context_dataset = UserContextDataset(pick_a_pic_v2_subset=dataset)

image_embeddings_folder = "./clip_embeddings"

full_dataset = UserContextCLIPEmbeddingsDataset(
    user_context_dataset=user_context_dataset,
    clip_embeddings=CLIPEmbeddingsDataset.from_folder(
        folder=image_embeddings_folder
    ),
)
train_test_split = 0.5
num_train_samples = int(len(full_dataset) * train_test_split)
train_dataset, validation_dataset = torch.utils.data.random_split(
    full_dataset, 
    [num_train_samples, len(full_dataset)-num_train_samples]
)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=UserContextCLIPEmbeddingsDataset.collate_fn_with_padding,
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=UserContextCLIPEmbeddingsDataset.collate_fn_with_padding,
)

# from mingpt import GPT
model_config = GPT.get_default_config()
model_config.model_type = None
model_config.vocab_size = 2 
model_config.block_size = int(768 * 2)
model_config.n_layer = 2
model_config.n_embd = 512
model_config.n_head = 2
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

        if step_idx % 10 == 0:
            accuracy = model.get_accuracy(
                logits=logits,
                sequence_lengths=batch["sequence_lengths"],
                labels=batch["labels"].to(device),
            )
            print(f"Step: {step_idx + 1} Epoch: {epoch_idx + 1} Training Loss: {loss.item()} Training Accuracy: {accuracy}")

        loss.backward()
        optimizer.step()
        step_idx += 1

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    total_val_accuracy = 0.0
    val_batches = 0

    with torch.no_grad():
        for val_batch in validation_dataloader:
            val_logits = model.forward(
                image_embeddings=val_batch["image_embeddings"].to(device),
                labels=val_batch["labels"].to(device),
            )
            val_loss = model.get_loss(
                logits=val_logits,
                sequence_lengths=val_batch["sequence_lengths"],
                labels=val_batch["labels"].to(device),
            )
            total_val_loss += val_loss.item()

            val_accuracy = model.get_accuracy(
                logits=val_logits,
                sequence_lengths=val_batch["sequence_lengths"],
                labels=val_batch["labels"].to(device),
            )
            total_val_accuracy += val_accuracy
            val_batches += 1

    average_val_loss = total_val_loss / val_batches
    average_val_accuracy = total_val_accuracy / val_batches

    print(f"Epoch: {epoch_idx + 1} Validation Loss: {average_val_loss} Validation Accuracy: {average_val_accuracy}")

    model.train()  # Set the model back to training mode
