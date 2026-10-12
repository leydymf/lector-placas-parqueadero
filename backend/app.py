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

# Confusiones tipicas del OCR entre letras y numeros
A_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
A_NUMERO = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "T": "1", "Z": "2",
            "A": "4", "S": "5", "G": "6", "B": "8"}

# Formatos colombianos: carro ABC123 y moto ABC12D (L = letra, N = numero)
FORMATOS = ["LLLNNN", "LLLNNL"]


def preprocesar_placa(imagen_bgr):
    """Preprocesa la region de la placa para mejorar el OCR."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    gris = cv2.resize(gris, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    return gris


def variantes_placa(imagen_bgr):
    """
    Genera tres versiones del recorte para el OCR: la preprocesada, una en gris
    escalada a 160 px de alto y una binarizada con Otsu.
    """
    procesada = preprocesar_placa(imagen_bgr)
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    escala = 160 / max(1, gris.shape[0])
    gris = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    _, binaria = cv2.threshold(cv2.GaussianBlur(gris, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [procesada, gris, binaria]


def leer_recorte(recorte_bgr, lector):
    """Prueba el OCR con cada variante y devuelve la primera lectura valida (placa, texto crudo)."""
    texto = ""
    for variante in variantes_placa(recorte_bgr):
        resultados = lector.readtext(
            variante, detail=1, paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )
        texto = texto_principal(resultados)
        placa = leer_placa(texto)
        if placa:
            return placa, texto
    return None, texto


def texto_principal(ocr_resultados):
    """
    Une solo los textos grandes de la placa (los caracteres) y descarta los
    pequenos, como el nombre de la ciudad que va impreso abajo.
    """
    if not ocr_resultados:
        return ""
    alturas = [max(p[1] for p in caja) - min(p[1] for p in caja) for caja, _, _ in ocr_resultados]
    alto_max = max(alturas)
    grandes = [
        (min(p[0] for p in caja), texto)
        for (caja, texto, _), alto in zip(ocr_resultados, alturas)
        if alto >= 0.6 * alto_max
    ]
    # Ordena de izquierda a derecha y une
    return "".join(texto for _, texto in sorted(grandes))


def corregir_con_formato(ventana, formato):
    """Corrige una ventana de 6 caracteres segun el formato; devuelve (texto, correcciones)."""
    salida, correcciones = [], 0
    for caracter, tipo in zip(ventana, formato):
        if tipo == "L" and caracter.isdigit():
            if caracter not in A_LETRA:
                return None, 99
            caracter, correcciones = A_LETRA[caracter], correcciones + 1
        elif tipo == "N" and caracter.isalpha():
            if caracter not in A_NUMERO:
                return None, 99
            caracter, correcciones = A_NUMERO[caracter], correcciones + 1
        salida.append(caracter)
    return "".join(salida), correcciones


def leer_placa(texto_ocr):
    """
    Busca en el texto del OCR la secuencia que mejor encaja con una placa
    colombiana y la devuelve en formato ABC-123 (o ABC-12D para motos).
    """
    limpio = re.sub(r"[^A-Z0-9]", "", texto_ocr.upper())
    # Una racha de 5 o mas letras es texto de la ciudad (BOGOTADC, MEDELLIN), nunca parte de la placa
    limpio = re.sub(r"[A-Z]{5,}", "", limpio)
    mejor, menos_correcciones = None, 99
    for inicio in range(max(0, len(limpio) - 5)):
        ventana = limpio[inicio:inicio + 6]
        for formato in FORMATOS:
            # El formato de moto solo se acepta si el ultimo caracter es una letra que no se confunde con un numero
            if formato == "LLLNNL" and (not ventana[-1:].isalpha() or ventana[-1] in A_NUMERO):
                continue
            corregido, correcciones = corregir_con_formato(ventana, formato)
            if corregido and correcciones < menos_correcciones:
                mejor, menos_correcciones = corregido, correcciones
    # Con mas de 2 correcciones la lectura no es confiable
    if mejor is None or menos_correcciones > 2:
        return None
    return f"{mejor[:3]}-{mejor[3:]}"


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

        # OCR sobre varias versiones del recorte
        placa_texto, texto_raw = leer_recorte(recorte, reader)

        # Si el texto no encaja con una placa colombiana no se reporta como lectura
        if placa_texto is None:
            logger.info(f"Placa detectada pero ilegible | OCR: {texto_raw!r} | YOLO: {confianza:.1%}")
            placa_texto = "ilegible"
        else:
            placas_detectadas.append(placa_texto)

        # Anotar imagen
        cv2.rectangle(imagen_anotada, (x1, y1), (x2, y2), (0, 255, 0), 3)
        etiqueta = f"{placa_texto} ({confianza:.0%})"
        cv2.putText(
            imagen_anotada, etiqueta, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        logger.info(f"Placa: {placa_texto} | OCR: {texto_raw!r} | YOLO: {confianza:.1%}")

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
