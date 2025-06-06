import os
import torch
from torch.utils.data import DataLoader
from imagetodescribe.pipeline.fabric_dataset import FashionDataset

def test_dataset():
    image_dir = "images"
    fabric_label_path = "labels/texture/fabric_ann.txt"
    caption_path = "captions.json"

    dataset = FashionDataset(
        image_dir=image_dir,
        fabric_label_path=fabric_label_path,
        caption_path=caption_path
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))

    print("Image batch shape:", batch["image"].shape)           # [4, 3, 224, 224]
    print("Label batch shape:", batch["label"].shape)           # [4, 3]
    print("Input IDs shape:", batch["input_ids"].shape)         # [4, 64]
    print("Attention mask shape:", batch["attention_mask"].shape)  # [4, 64]

    # Optional: Decode a caption to confirm
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("Decoded caption example:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

if __name__ == "__main__":
    test_dataset()
