import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from lightning.pytorch import seed_everything
from creator_lora.dataset import MidJourneyDataset
from creator_lora.utils.json_stuff import load_json, save_as_json
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import wandb

seed_everything(0)

config = load_json("config.json")

images_folder = os.path.join(config["dataset_root_folder"], "images")

output_json_file = os.path.join(config["dataset_root_folder"], "data.json")
dataset = MidJourneyDataset(
    images_folder=images_folder,
    output_json_file=output_json_file,
    image_transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees = 5, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    ),
)

print(f"Length of dataset: {len(dataset)} images")

num_train_samples = int(len(dataset) * config["train_test_split"])

# Created using indices from 0 to train_size.
train_dataset = torch.utils.data.Subset(dataset, range(num_train_samples))

# Created using indices from train_size to train_size + test_size.
validation_dataset = torch.utils.data.Subset(
    dataset, range(num_train_samples, len(dataset))
)


# if not os.path.exists(config["sampler_weights_filename"]):
#     print(f"Computing sampler_weights...")
#     sampler_weights = [
#         config["class_weights"][train_dataset[i]["label"]] for i in tqdm(range(len(train_dataset)))
#     ]
#     save_as_json(sampler_weights,config["sampler_weights_filename"])
# else:
#     print(f"Loading existing sampler weights: {config['sampler_weights_filename']}")
#     sampler_weights = load_json(config["sampler_weights_filename"])

# sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    # sampler=sampler
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
)
from creator_lora.image_encoders import CLIPImageEncoder

# model = CLIPImageEncoder(name = "RN50", device = config["device"]).model
# model.attnpool = nn.Sequential(
#     model.attnpool,
#     nn.Linear(1024, 1),
#     nn.Sigmoid(),
#     # nn.Linear(512, 256),
#     # nn.ReLU(),
#     # nn.Linear(256, 1),
# )

model = models.resnet50(weights="DEFAULT")
model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

model.to(config["device"])

optimizer = torch.optim.Adam(
    # [
    #     *model.attnpool.parameters(),
    # ],
    model.parameters(),
    lr=1e-3,
)


def loss_function(logits, labels, prompts):
    """
    the problem right now is that even if we're reformulating the
    loss as a 4 way classification task, the model still thinks it
    has to do binary classification since it is give only one image per batch item.

    the potential solution would be an alternative architecture.
    something that takes in a sequence images as input and then


    xander plot
    sample ranodm images from the dataset
    histogram of their clip similarities
    do the same for random pairs of images of the same batch

    analyse intra prompt diversity and inter prompt diversity of images
    """

    assert labels.shape[0] == 4
    assert logits.shape[0] == 4
    assert logits.shape[1] == 1

    ## making sure all images are generates from the same prompt
    for i in range(1, len(prompts)):
        assert prompts[0] == prompts[i]

    logits = logits.reshape(1, -1)
    labels = labels.reshape(-1)
    cross_entropy_label = (labels == 1).nonzero()[0]

    import random

    if random.random() < 0.01:
        print(logits)
    return nn.CrossEntropyLoss()(logits, cross_entropy_label)


def validation_run(config, model, validation_dataloader, epoch: int):
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_valid_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for valid_batch in validation_dataloader:
            logits_valid = model.forward(valid_batch["image"].to(config["device"]))
            valid_loss = loss_function(
                logits_valid,
                valid_batch["label"].to(config["device"]).float().unsqueeze(-1),
                prompts=valid_batch["prompt"],
            )

            total_valid_loss += valid_loss.item()

            # Calculate accuracy
            predictions = (
                logits_valid > config["validation_accuracy_threshold"]
            ).float()
            correct_predictions += (
                (
                    predictions
                    == valid_batch["label"].to(config["device"]).float().unsqueeze(-1)
                )
                .sum()
                .item()
            )
            total_samples += valid_batch["label"].size(0)

        # Calculate and print validation loss and accuracy
        average_valid_loss = total_valid_loss / len(validation_dataloader)

        accuracy = correct_predictions / total_samples

        if config["wandb_log"]:
            wandb.log(
                {"validation_loss": average_valid_loss, "validation_accuracy": accuracy}
            )
        print(
            f"Epoch {epoch + 1}, Validation Loss: {average_valid_loss}, Accuracy: {accuracy}"
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
            prompts=batch["prompt"],
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
                wandb.log({"training_loss": average_loss})

            # print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Average Loss: {average_loss}")

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
    wandb.init(project="eden-creator-lora", config=config)

for epoch in range(10):
    if epoch == 0:
        validation_run(
            config=config,
            model=model,
            validation_dataloader=validation_dataloader,
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
        epoch=epoch,
    )
