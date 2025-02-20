from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load YOLO model
model = YOLO("./best_v11-60epochs.pt")  # Replace with your custom model path if needed

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change "*" to specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        results = model(image)

        class_names = []
        for box in results[0].boxes:
            class_id = box.cls.item()
            class_name = model.names[int(class_id)]
            class_names.append(class_name)

        return JSONResponse(content={"classes": list(set(class_names))})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def home():
    return {"message": "Welcome to YOLO Inference API"}
