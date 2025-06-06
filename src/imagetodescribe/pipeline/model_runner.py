import torch
from torchvision import transforms
from PIL import Image
from transformers import DistilBertTokenizer
from .multimodal_fabric_classifier import MultimodalFabricClassifier

class FabricModelRunner:
    def __init__(self, model_path=None, device=None):
        if model_path is None:
            raise FileNotFoundError("Model path must be provided.")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultimodalFabricClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image: Image.Image, caption: str):
        # Process image
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Process text
        inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor, input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()  # [upper, lower, outer]

        return {
            "upper_fabric": predictions[0],
            "lower_fabric": predictions[1],
            "outer_fabric": predictions[2]
        }
