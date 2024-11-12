import os
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import shutil
from PIL import Image
import io
import base64
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import torch.nn.functional as F
import requests

app = FastAPI()
app.mount("/files", StaticFiles(directory="files"), name="files")

origins = [
    "http://localhost:3000",  # URL de tu aplicación Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
POCKETBASE_URL = "http://127.0.0.1:8090"

# app.mount("/files", StaticFiles(directory="files"), name="files")
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained("../checkpoint/larbz-chestxray-classifierv2",output_attentions=True)

def predict_image(image):
    # image = (
    #     Image.open(image_path)
    #     .convert("RGB")
    #     .resize((512, 512))
    # )
    # image = dataset['validation'][1]['image'].convert("RGB").resize((512,512))
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_label_id = probs.argmax(-1).item()
    # print(model.config)
    predicted_label = model.config.id2label[predicted_label_id]
    confidence = probs.max().item()
    return {
        'predicted_label':predicted_label,
        'confidence':confidence
    }


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    os.makedirs("files", exist_ok=True)
    file_location = f"files/{file.filename}"
    print(file_location)
    #Medic ID qzsuham5mpjmm2v
    #Patient ID bvyaauikgveoeqs
    create_record('','radiologies', data = {
    "patient": "bvyaauikgveoeqs",
    "medic": "qzsuham5mpjmm2v",
    "confidence": 23,
    "label": "test"
    },files={
        "img":file
    })

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"url": f"http://localhost:8000/files/{file.filename}"}
    # return JSONResponse(content={"report": "hello"})
    
def create_record(token: str=Form(...),confidence:float= Form(...), collection: str=Form(...), patient:str=Form(...),medic:str=Form(...),file:UploadFile=File(...),label:str=Form(...)):
    url = f"{POCKETBASE_URL}/api/collections/{collection}/records"
    headers = {
        # "Content-Type":"application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "patient": patient,
        "medic": medic,
        "confidence": confidence,
        "label": label
    }
    file.file.seek(0)
    file_content = file.file.read()  # Lee el contenido del archivo
    files = {"img": (file.filename, file_content, file.content_type)}
    response = requests.post(url, data=data,files=files, headers=headers)
    print("called")
    print(response.json())
    return response.json()
    



@app.get("/get_report/")
async def get_report(image_url: str):
    radiological_report = """

    Paciente: Juan Pérez
    Fecha del Estudio: 12 de junio de 2024
    Modalidad: Radiografía de Tórax

    """
    # Hallazgos:
    # - Pulmones: Volúmenes pulmonares dentro de los límites normales. No se observan infiltrados, consolidaciones
    #   ni masas.
    # - Corazón: Tamaño y silueta cardíaca normales.
    # - Mediastino: No se observan adenopatías ni masas mediastínicas.
    # - Diafragma: Contornos diafragmáticos normales.
    # - Pleura: No se observa derrame pleural ni neumotórax.

    # Impresión:
    # - Radiografía de tórax normal. No se identifican hallazgos patológicos significativos.

    # Recomendaciones:
    # - Continuar con controles rutinarios según indicación médica.
    return JSONResponse(content={"report": radiological_report})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo subido
        contents = await file.read()
        # Abrir la imagen con PIL
        image = Image.open(io.BytesIO(contents))

        image_to_analize = image.convert("RGB").resize((512,512))
        response = predict_image(image_to_analize)
        print(response)
        label = response['predicted_label']
        confidence = response['confidence']
        # Mostrar información básica de la imagen
        # return {"filename": file.filename, "format": image.format, "size": image.size, "mode": image.mode}
        # return image
        # Guardar la imagen temporalmente
        temp_image_path = "temp_image.png"
        # image.save(temp_image_path, format="PNG")
        
        # # Devolver la imagen como respuesta HTTP
        # return FileResponse(temp_image_path, media_type="image/png", filename=file.filename)
        # Convertir la imagen a base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        create_record("",confidence,"radiologies",patient="bvyaauikgveoeqs",
                      medic="qzsuham5mpjmm2v",file=file,label=label
                      )
        # Crear la respuesta JSON
        response = {
            "filename": file.filename,
            "label": label,
            "confidence":confidence,
            "image": img_str
        }

        return JSONResponse(content=response)
    except Exception as e:
        return {"error": str(e)}