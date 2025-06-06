import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging as hf_logging
from imagetodescribe.pipeline.fabric_dataset import FashionDataset
from imagetodescribe.pipeline.multimodal_fabric_classifier import MultimodalFabricClassifier

hf_logging.set_verbosity_error()  # suppress tokenizer/model loading logs

def train():
    # ----- Config -----
    image_dir = "images"
    fabric_label_path = "labels/texture/fabric_ann.txt"
    caption_path = "captions.json"
    batch_size = 32
    num_epochs = 5
    learning_rate = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # ----- Data -----
    dataset = FashionDataset(image_dir, fabric_label_path, caption_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----- Model -----
    model = MultimodalFabricClassifier()
    model.to(device)

    # ----- Loss & Optimizer -----
    criterion = nn.CrossEntropyLoss(ignore_index=7)  # ignore NA labels
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("✅ Model initialized. Using device:", device)
    # ----- Training Loop -----
    model.train()
    print("🚀 Starting training loop...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(images, input_ids, attention_mask)  # shape: [B, 3, 8]

            loss = 0
            for i in range(3):  # for each of the 3 labels
                loss += criterion(outputs[:, i, :], labels[:, i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(dataloader):.4f}")

    # ----- Save Model -----
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/multimodal_fabric_model.pt")
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    train()
