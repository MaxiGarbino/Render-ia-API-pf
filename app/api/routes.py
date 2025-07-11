# app/api/routes.py
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.predictor import predict_image
import shutil
import uuid
import os

router = APIRouter()

TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.get("/hola")
def say_hello():
    return {"message": "Hola desde el router"}

@router.post("/diagnosticar")
async def diagnosticar(file: UploadFile = File(...)):
    # Guardar imagen temporal
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, unique_name)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await predict_image(temp_path)
    os.remove(temp_path)  # Limpieza

    return JSONResponse(content=result)
