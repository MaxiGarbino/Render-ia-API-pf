import os
import shutil
import uuid
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.utils import load_img, img_to_array  # Updated imports
import numpy as np
from typing import List

app = FastAPI()

# Directory for saving temporary images
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define file paths
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'models')

# Load the TensorFlow model
model_path = os.path.join(save_dir, 'melanoma-vs-bening.h5')
model = tf.keras.models.load_model(model_path)

async def diagnosticar(img_path: str) -> str:
    # Load and preprocess the image
    img = load_img(img_path, target_size=(150, 150))  # Load and resize the image
    x = img_to_array(img)  # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension (TensorFlow 2.x still expects batch)
    x /= 255.0  # Normalize pixel values (if needed, based on how the model was trained)

    # Perform prediction
    classes = model.predict(x)  # Predict the class probabilities

    # Interpret the prediction
    if classes[0] > 0.5:
        return "No hay enfermedad detectada"
    else:
        return "Melanoma"

@app.post("/")
async def create_upload_file(files: List[UploadFile] = File(..., description='Descripción')):
    diagnoses = []

    for file in files:
        # Create a unique name for each file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Diagnose the image
        diagnosis = await diagnosticar(file_path)
        diagnoses.append(diagnosis)

    return JSONResponse(content={"diagnoses": diagnoses})

# Endpoint for a single file
@app.post("/uploadfile/")
async def create_upload_file_single(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Diagnose the image
    diagnosis = await diagnosticar(file_path)

    return JSONResponse(content={"diagnosis": diagnosis})
