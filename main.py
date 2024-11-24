import os
import shutil
import uuid
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import List

app = FastAPI()

# Directorio temporal para guardar imágenes
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Directorio donde se encuentran los modelos
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'models')

# Cargar los tres modelos
model_paths = [
    os.path.join(save_dir, 'clasificador_4_clases.h5'),
    os.path.join(save_dir, 'clasificador_4_clases_2.h5'),
    os.path.join(save_dir, 'clasificador_4_clases_3.h5')
]
models = [tf.keras.models.load_model(model_path) for model_path in model_paths]

# Etiquetas de las 4 clases (modificar según las clases reales del modelo)
class_labels = ['piel-normal', 'lunar', 'melanoma', 'acne']

async def diagnosticar(img_path: str) -> dict:
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Obtener predicciones de todos los modelos
    all_predictions = [model.predict(x)[0] for model in models]
    
    # Promediar las probabilidades de las predicciones
    averaged_predictions = np.mean(all_predictions, axis=0)

    # Ordenar las probabilidades y sus índices
    sorted_indices = np.argsort(averaged_predictions)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_probabilities = [averaged_predictions[i] * 100 for i in sorted_indices]

    # Diagnóstico inicial
    primary_diagnosis = sorted_labels[0]
    primary_probability = sorted_probabilities[0]

    # Verificar si el diagnóstico principal necesita ajustes
    if primary_diagnosis == "melanoma" and primary_probability < 80:
        secondary_diagnosis = sorted_labels[1]
        secondary_probability = sorted_probabilities[1]
        return {
            "diagnosis": secondary_diagnosis,
            "probability": f"{secondary_probability:.2f}%",
            "original_diagnosis": primary_diagnosis,
            "original_probability": f"{primary_probability:.2f}%",
            "all_probabilities": {class_labels[i]: f"{averaged_predictions[i] * 100:.2f}%" for i in range(len(class_labels))}
        }

    return {
        "diagnosis": primary_diagnosis,
        "probability": f"{primary_probability:.2f}%",
        "all_probabilities": {class_labels[i]: f"{averaged_predictions[i] * 100:.2f}%" for i in range(len(class_labels))}
    }

@app.post("/")
async def create_upload_file(files: List[UploadFile] = File(...)):
    diagnoses = []

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        diagnosis = await diagnosticar(file_path)
        diagnoses.append(diagnosis)

        os.remove(file_path)

    return JSONResponse(content={"diagnoses": diagnoses})

@app.post("/uploadfile/")
async def create_upload_file_single(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis = await diagnosticar(file_path)
    os.remove(file_path)

    return JSONResponse(content={"diagnosis": diagnosis})
