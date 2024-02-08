import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from creator_lora.utils.image import crop_center
import os

class ResNet50MLP(nn.Module):
    def __init__(
        self, 
        model_path: str = None, 
        device: str = "cpu", 
        weights: str ="DEFAULT"
    ):
        """
        Aesthetic scoring model
        The output of the forward pass is a score between 1 and 10
        Note that divide that score by 10 during inference to match Xander's model
        """
        super().__init__()
        self.transforms = transforms.Compose(
            [
                crop_center,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = models.resnet50(weights=weights)
        ## usually we would only finetune this for aesthetic scoring
        self.model.fc = nn.Sequential(
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

        if model_path is not None:
            assert os.path.exists(model_path), f"Invalid model_path: {model_path}"
            print(f"Loading checkpoint: {model_path}")
            self.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)
        self.device = device

    def forward(self, x):
        return self.model(x.to(self.device))
 
    ## inference
    @torch.no_grad()
    def predict_score(self, pil_image):
        self.model.eval()
        image_tensor = self.transforms(pil_image).unsqueeze(0)
        logits = self.forward(image_tensor)
        assert logits.shape[1]== 1, f"Expected a score of size 1 but got: {logits.shape[1]}"
        ## we divide by 10 to match xander's model
        score = logits[0][0].item()/10
        return score

    @torch.no_grad()
    def predict_score_batch(self, pil_images: list):
        self.model.eval()
        image_tensor = torch.stack(
            [
                self.transforms(image).unsqueeze(0)
                for image in pil_images
            ]
        )
        logits = self.forward(image_tensor)
        assert logits.shape[1]== 1, f"Expected a score of size 1 but got: {logits.shape[1]}"
        ## we divide by 10 to match xander's model
        scores = (logits/10).tolist()
        return scores