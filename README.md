# Fashion Vision Bot

A multimodal fashion classification system that combines computer vision and natural language processing to analyze fashion images and predict fabric types for different clothing categories.
Includes REST Server to host easily at an endpoint.

## How It Works

This system uses a dual-encoder architecture that processes both images and text:

**Image Processing**: A ResNet18 backbone extracts visual features from fashion images. The model looks at clothing items and understands their visual characteristics like texture, patterns, and structure.

**Text Processing**: A DistilBERT encoder processes text captions or descriptions. This helps the model understand context about the clothing item that might not be obvious from the image alone.

**Fusion**: The visual and text features are combined and fed through a classifier that predicts fabric types for three clothing categories:
- Upper body garments (shirts, tops, etc.)
- Lower body garments (pants, skirts, etc.) 
- Outer garments (jackets, coats, etc.)

The model can distinguish between 8 different fabric types for each category: denim, cotton, leather, furry, knitted, chiffon, and other materials. This gives you detailed material predictions for complete outfits.

## Dataset

This project uses the [DeepFashion-MultiModal dataset](https://github.com/yumingj/DeepFashion-MultiModal/tree/main), a large-scale high-quality human dataset with rich multi-modal annotations including 44,096 high-resolution fashion images with manual fabric, shape, and color annotations.

## Quick Start

1. Install dependencies:
   ```bash
   make install
   ```

2. Start the API server:
   ```bash
   make serve
   ```

The server will be available at `http://localhost:8000`

## API Endpoints

- `POST /predict/` - Upload an image and caption to get a prediction
- `GET /health/` - Health check endpoint

## Usage Example

```bash
curl -X POST "http://localhost:8000/predict/" \
  -F "image=@path/to/your/image.jpg" \
  -F "caption=your caption here"
```

## Development

- `make test` - Run tests
- `make lint` - Run linting checks
- `make format` - Format code
- `make help` - Show all available commands
