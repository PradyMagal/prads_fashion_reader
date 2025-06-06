import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from transformers import DistilBertTokenizer

class FashionDataset(Dataset):
    def __init__(self, image_dir, fabric_label_path, caption_path, tokenizer_name="distilbert-base-uncased", max_length=64, transform=None):
        self.image_dir = image_dir
        self.fabric_labels = self._load_fabric_labels(fabric_label_path)
        self.captions = self._load_captions(caption_path)
        self.image_names = list(self.fabric_labels.keys())
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_fabric_labels(self, path):
        labels = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                label = list(map(int, parts[1:4]))  # [upper, lower, outer]
                labels[filename] = label
        return labels

    def _load_captions(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(self.fabric_labels[image_name], dtype=torch.long)

        caption = self.captions.get(image_name, "")
        encoding = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
