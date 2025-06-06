import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel
import torch

class MultimodalFabricClassifier(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased", embedding_dim=512, hidden_dim=256, num_classes=3):
        super().__init__()

        # 1. Image encoder (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
        self.image_fc = nn.Linear(resnet.fc.in_features, embedding_dim)

        # 2. Text encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_encoder.config.dim, embedding_dim)

        # 3. Classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * 8)  # 3 labels × 8 classes
        )

    def forward(self, images, input_ids, attention_mask):
        # Encode image
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)
        img_feat = self.image_fc(img_feat)

        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(text_outputs.last_hidden_state[:, 0, :])  # CLS token

        # Combine
        fused = torch.cat((img_feat, text_feat), dim=1)
        logits = self.classifier(fused)
        logits = logits.view(-1, 3, 8)  # shape: (B, 3, 8)

        return logits
