"""
Backend ALPR - Deteccion y lectura de placas colombianas
YOLOv8 + EasyOCR + FastAPI

Endpoints:
  GET  /         -> Health check
  POST /predict/ -> Recibe imagen, devuelve placas detectadas

Uso local:
  python app.py

Uso en AWS EC2:
  source venv/bin/activate
  python app.py
"""

import base64
import io
import logging
import os
import re

import cv2
import easyocr
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# --- Configuracion -----------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.3"))
PORT = int(os.getenv("PORT", "8080"))

# --- Inicializacion del servidor ---------------------------------------------

app = FastAPI(
    title="ALPR Placas Colombianas",
    description="Detecta y lee placas vehiculares colombianas con YOLOv8 + EasyOCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Carga de modelos --------------------------------------------------------

logger.info(f"Cargando modelo YOLOv8 desde: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
logger.info("Modelo YOLOv8 cargado")

logger.info("Inicializando EasyOCR (espanol + ingles)...")
reader = easyocr.Reader(["es", "en"], gpu=False)
logger.info("EasyOCR listo")

# --- Funciones auxiliares ----------------------------------------------------

# Prefijos de placas colombianas por ciudad/departamento
PREFIJOS_CIUDADES = {
    "AAA": "Bogota", "BAA": "Bogota", "CAA": "Bogota", "DAA": "Bogota",
    "EAA": "Bogota", "FAA": "Bogota", "GAA": "Bogota", "HAA": "Bogota",
    "HIM": "Medellin", "MEP": "Medellin", "OCA": "Medellin",
    "BXR": "Cali", "CXR": "Cali", "DXR": "Cali",
    "OBR": "Bucaramanga", "OBS": "Bucaramanga", "OBT": "Bucaramanga",
    "OBU": "Bucaramanga", "OBV": "Bucaramanga",
    "BCH": "Barranquilla", "BCJ": "Barranquilla",
    "OHG": "Cundinamarca", "OHH": "Cundinamarca",
}


def preprocesar_placa(imagen_bgr):
    """Preprocesa la region de la placa para mejorar el OCR."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    gris = cv2.resize(gris, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    return gris


def extraer_ciudad(texto_placa):
    """Extrae numero y ciudad de una placa colombiana (formato ABC-123)."""
    texto_limpio = re.sub(r"[^A-Za-z0-9-]", "", texto_placa).upper()
    patron = re.search(r"([A-Z]{3})[-]?([0-9]{3})", texto_limpio)

    if patron:
        letras = patron.group(1)
        numeros = patron.group(2)
        placa = f"{letras}-{numeros}"

        ciudad = PREFIJOS_CIUDADES.get(letras)
        if not ciudad:
            for prefijo, c in PREFIJOS_CIUDADES.items():
                if letras[:2] == prefijo[:2]:
                    ciudad = c
                    break
        ciudad = ciudad or "Desconocida"
        return placa, ciudad

    return texto_limpio, "No identificada"


def imagen_a_base64(imagen_bgr):
    """Convierte imagen OpenCV (BGR) a string base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", imagen_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def decodificar_imagen(data):
    """Decodifica bytes o base64 a imagen OpenCV (BGR)."""
    if isinstance(data, str):
        data = base64.b64decode(data)
    imagen = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)


def detectar_placas(imagen_bgr):
    """
    Pipeline completo: YOLO detecta -> OpenCV recorta -> EasyOCR lee.
    Retorna lista de placas detectadas y la imagen anotada.
    """
    alto, ancho = imagen_bgr.shape[:2]
    imagen_anotada = imagen_bgr.copy()

    # Deteccion con YOLOv8
    resultados = model.predict(source=imagen_bgr, conf=CONFIDENCE, verbose=False)

    if len(resultados) == 0 or len(resultados[0].boxes) == 0:
        return [], imagen_anotada

    placas_detectadas = []

    for box in resultados[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confianza = float(box.conf[0])

        # Recorte con margen
        margen = 5
        x1m = max(0, x1 - margen)
        y1m = max(0, y1 - margen)
        x2m = min(ancho, x2 + margen)
        y2m = min(alto, y2 + margen)
        recorte = imagen_bgr[y1m:y2m, x1m:x2m]

        # Preprocesamiento + OCR
        placa_procesada = preprocesar_placa(recorte)
        ocr_resultados = reader.readtext(
            placa_procesada,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )

        texto_raw = " ".join([r[1] for r in ocr_resultados])
        placa_texto, ciudad = extraer_ciudad(texto_raw)

        placas_detectadas.append(placa_texto)

        # Anotar imagen
        cv2.rectangle(imagen_anotada, (x1, y1), (x2, y2), (0, 255, 0), 3)
        etiqueta = f"{placa_texto} ({confianza:.0%})"
        cv2.putText(
            imagen_anotada, etiqueta, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        logger.info(f"Placa: {placa_texto} | Ciudad: {ciudad} | YOLO: {confianza:.1%}")

    return placas_detectadas, imagen_anotada


# --- Endpoints ---------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "modelo": MODEL_PATH,
        "descripcion": "ALPR Placas Colombianas - YOLOv8 + EasyOCR",
    }


@app.post("/predict/")
async def predict(
    file: UploadFile | None = File(None),
    image_base64: str | None = Form(None),
):
    """
    Recibe una imagen (como archivo o base64) y devuelve las placas detectadas.

    La app movil envia base64 via form-data.
    Postman/curl pueden enviar archivo multipart.
    """
    # Decodificar imagen desde archivo o base64
    if file is not None:
        data = await file.read()
        imagen_bgr = decodificar_imagen(data)
    elif image_base64 is not None:
        imagen_bgr = decodificar_imagen(image_base64)
    else:
        return {"error": "Envia una imagen como 'file' (multipart) o 'image_base64' (form)"}

    # Ejecutar pipeline de deteccion
    placas, imagen_anotada = detectar_placas(imagen_bgr)

    # Respuesta JSON
    return {
        "placas": placas,
        "cantidad": len(placas),
        "image": imagen_a_base64(imagen_anotada),
    }


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Iniciando servidor en puerto {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
