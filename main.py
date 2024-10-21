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

# Define las rutas de los archivos
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'models')

# Carga el modelo TensorFlow
model_path = os.path.join(save_dir, 'clasificador_4_clases.h5')
model = tf.keras.models.load_model(model_path)

# Etiquetas de las 4 clases (modificar según las clases reales del modelo)
class_labels = ['piel-normal', 'lunar', 'melanoma', 'acne']

async def diagnosticar(img_path: str) -> dict:
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar la imagen al tamaño esperado por el modelo
    x = image.img_to_array(img)  # Convertir la imagen a un array numpy
    x = np.expand_dims(x, axis=0)  # Añadir la dimensión del lote
    x = x / 255.0  # Normalizar la imagen dividiendo por 255

    # Realizar la predicción
    predictions = model.predict(x)

    # Obtener el índice de la clase con mayor probabilidad
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Obtener la probabilidad de la clase predicha
    predicted_probability = predictions[0][predicted_class_index] * 100  # Convertir a porcentaje

    # Retornar el nombre de la clase predicha y su probabilidad
    return {
        "diagnosis": predicted_class_label,
        "probability": f"{predicted_probability:.2f}%",  # Formato de porcentaje con dos decimales
        "all_probabilities": {class_labels[i]: f"{predictions[0][i] * 100:.2f}%" for i in range(len(class_labels))}
    }

@app.post("/")
async def create_upload_file(files: List[UploadFile] = File(..., description='Subir imágenes para diagnóstico')):
    diagnoses = []

    for file in files:
        # Crear un nombre único para cada archivo
        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

        # Guardar el archivo temporalmente
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Diagnosticar la imagen
        diagnosis = await diagnosticar(file_path)
        diagnoses.append(diagnosis)

        # Eliminar la imagen después de procesar (opcional para liberar espacio)
        os.remove(file_path)

    return JSONResponse(content={"diagnoses": diagnoses})

# Endpoint adicional para manejar un solo archivo y devolver el diagnóstico
@app.post("/uploadfile/")
async def create_upload_file_single(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Diagnosticar la imagen
    diagnosis = await diagnosticar(file_path)

    # Eliminar la imagen después de procesar (opcional)
    os.remove(file_path)

    return JSONResponse(content={"diagnosis": diagnosis})
