from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import io
from imagetodescribe.pipeline.model_runner import FabricModelRunner
from loguru import logger

# initialize server
app = FastAPI()
logger.info("Starting FastAPI server...")

# initialize model_runner
model_runner = FabricModelRunner(
    model_path="checkpoints/multimodal_fabric_model.pt"
)
logger.info("Model loaded successfully - runner initialized")


@app.post("/predict/")
async def predict(image: UploadFile = File(...), caption: str = Form(...)):
    try:
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        prediction = model_runner.predict(image_pil, caption)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed - please check the logs for more details.") 

    return {"prediction": prediction}


@app.get("/health/")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}