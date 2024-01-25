import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from creator_lora.dataset.eden import EdenDataset
from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.utils.json_stuff import load_json, save_as_json
from creator_lora.utils.sampler_weights import compute_sampler_weights

config = load_json("config.json")

dataset = EdenDataset(
    filename = config["dataset_filename"],
    image_transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    ),
)

dataset.shuffle(seed=0)

model = CLIPImageEncoder(name = "RN50", device = config["device"]).model
model.attnpool = nn.Sequential(
    model.attnpool,
    nn.Linear(1024, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
model.to(config["device"])

print(f"Length of dataset: {len(dataset)} images")

num_train_samples = int(len(dataset) * config["train_test_split"])

# Created using indices from 0 to train_size.
train_dataset = torch.utils.data.Subset(dataset, range(num_train_samples))
# Created using indices from train_size to train_size + test_size.
validation_dataset = torch.utils.data.Subset(
    dataset, range(num_train_samples, len(dataset))
)

if not os.path.exists(config["sampler_weights_filename"]):
    print(f"Computing sampler_weights...")
    all_train_labels = [
        train_dataset[idx]["label"] for idx in range(len(train_dataset))
    ]
    sampler_weights = compute_sampler_weights(all_train_labels)
    save_as_json(sampler_weights,config["sampler_weights_filename"])
else:
    print(f"Loading existing sampler weights: {config['sampler_weights_filename']}")
    sampler_weights = load_json(config["sampler_weights_filename"])

sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    sampler=sampler
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
)

optimizer = torch.optim.Adam(
    model.attnpool[1].parameters(),
    lr=config['lr'],
)

def loss_function(logits, labels):
    return nn.MSELoss()(logits, labels)


def validation_run(config, model, validation_dataloader, epoch: int, loss_function: callable):
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_valid_loss = 0.0
        total_samples = 0

        for valid_batch in validation_dataloader:
            logits_valid = model.forward(valid_batch["image"].to(config["device"]))
            valid_loss = loss_function(
                logits_valid,
                valid_batch["label"].to(config["device"]).float().unsqueeze(-1),
            )

            total_valid_loss += valid_loss.item()

            
            total_samples += valid_batch["label"].size(0)

        # Calculate and print validation loss
        average_valid_loss = total_valid_loss / len(validation_dataloader)


        if config["wandb_log"]:
            wandb.log(
                {"validation_loss": average_valid_loss}
            )
        print(
            f"Epoch {epoch + 1}, Validation Loss: {average_valid_loss}"
        )

    model.train()  # Set the model back to training mode


def train_one_epoch(config, model, train_dataloader, loss_function, optimizer):
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        logits = model.forward(batch["image"].to(config["device"]))

        loss = loss_function(
            logits,
            batch["label"].to(config["device"]).float().unsqueeze(-1),
        )

        loss.backward()

        total_loss += loss.item()

        if (batch_idx + 1) % config["num_gradient_accumulation_steps"] == 0:
            # Update the model parameters after accumulating gradients for config["num_gradient_accumulation_steps"] batches
            optimizer.step()
            optimizer.zero_grad()

            # Print the average loss over config["num_gradient_accumulation_steps"] batches
            average_loss = total_loss / config["num_gradient_accumulation_steps"]

            if config["wandb_log"]:
                wandb.log(
                    {
                        "training_loss": average_loss,
                        "pred": wandb.Histogram(logits.reshape(-1).tolist())

                    }
                )
            else:
                print(f"Batch Stats: \nlabel mean: {batch['label'].float().mean()} var: {batch['label'].float().var()}\npred mean: {logits.mean().item()} var: {logits.var().item()}")
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Average Loss: {average_loss}")

            total_loss = 0.0  # Reset the total loss for the next accumulation

    # If there are remaining batches, update the model parameters
    if batch_idx % config["num_gradient_accumulation_steps"] != 0:
        optimizer.step()
        optimizer.zero_grad()

        # Print the average loss for the remaining batches
        average_loss = total_loss / (
            batch_idx % config["num_gradient_accumulation_steps"]
        )

        if config["wandb_log"]:
            wandb.log({"training_loss": average_loss})
        # print(f"Epoch {epoch + 1}, Remaining Batches, Average Loss: {average_loss}")


if config["wandb_log"]:
    wandb.init(project="eden-aesthetic-classifier", config=config)

for epoch in range(10):
    if epoch == 0:
        validation_run(
            config=config,
            model=model,
            validation_dataloader=validation_dataloader,
            loss_function=loss_function,
            epoch=epoch,
        )

    train_one_epoch(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    validation_run(
        config=config,
        model=model,
        validation_dataloader=validation_dataloader,
        loss_function=loss_function,
        epoch=epoch,
    )