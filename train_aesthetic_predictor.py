import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
# from creator_lora.dataset.eden import EdenDataset
from creator_lora.dataset.ava import AvaDataset
import torchvision.models as models
from creator_lora.utils.json_stuff import load_json, save_as_json
from creator_lora.utils.sampler_weights import compute_sampler_weights
from creator_lora.utils.image import crop_center

config = load_json("config.json")

dataset = AvaDataset(
    csv_filename = "ava_dataset.csv", 
    image_transform=transforms.Compose(
        [
            crop_center,
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    ),
)


model = models.resnet50(weights="DEFAULT")
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(64, 1),
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
        train_dataset[idx]["label"] for idx in tqdm(range(len(train_dataset)))
    ]
    sampler_weights = compute_sampler_weights(all_train_labels)
    save_as_json(sampler_weights,config["sampler_weights_filename"])
else:
    print(f"Loading existing sampler weights: {config['sampler_weights_filename']}")
    sampler_weights = load_json(config["sampler_weights_filename"])

sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"]["train"],
    shuffle=False,
    # sampler=sampler
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=config["batch_size"]["validation"],
    shuffle=False,
)

optimizer = torch.optim.Adam(
    model.fc.parameters(),
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

        for valid_batch in tqdm(validation_dataloader, desc = f"Validation Run"):
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
    return average_valid_loss


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
    wandb.init(project=config["wandb_project_name"], config=config)


validation_losses = []
for epoch in range(config["num_epochs"]):
    if epoch == 0:
        loss = validation_run(
            config=config,
            model=model,
            validation_dataloader=validation_dataloader,
            loss_function=loss_function,
            epoch=epoch,
        )
        validation_losses.append(loss)

    train_one_epoch(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    loss = validation_run(
        config=config,
        model=model,
        validation_dataloader=validation_dataloader,
        loss_function=loss_function,
        epoch=epoch,
    )
    if loss < min(validation_losses):
        print(f"Best val loss: {loss}")
        torch.save(
            model.state_dict(),
            config["checkpoint_filename"]
        )
    validation_losses.append(loss)