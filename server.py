from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import gdown

output_path = "final_epochs-60-yolov11x.pt"
url = f"https://drive.google.com/uc?id={'19KazqWcoAeTS48tmLF12orW51sqzOOSI'}"
gdown.download(url, output_path, quiet=False)

app = FastAPI()

# Load YOLO model
# model1 = YOLO("https://drive.google.com/file/d/1nnXc4k6yPUEurM3LTuJXst193TXnMbNW/view?usp=drive_link")  # Replace with your custom model path if needed
model = YOLO("./final_epochs-60-yolov11x.pt")

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
