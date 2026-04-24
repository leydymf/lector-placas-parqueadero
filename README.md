# Lector de Placas Vehiculares Colombianas - ALPR

Sistema de reconocimiento automatico de placas vehiculares colombianas (ALPR) para control de ingreso en parqueaderos.

**Estudiante:** Leydy Yohana Macareo Fuentes
**Curso:** Ciencia de Datos - UNAB 
**Fecha:** Abril 2026

---

## Descripcion

Aplicacion movil que detecta y lee placas de vehiculos colombianos usando inteligencia artificial. El sistema funciona en dos etapas:

1. **Deteccion**: Un modelo YOLOv8n localiza la placa en la imagen
2. **Lectura (OCR)**: EasyOCR extrae los caracteres de la placa (formato ABC-123)

Adicionalmente, el sistema identifica la ciudad o departamento colombiano segun el prefijo de la placa.

## Arquitectura del Sistema

```
+---------------------+          +------------------------+
|    App Movil         |   HTTP   |    Backend AWS EC2     |
|    (Expo / React     | -------> |    (FastAPI)           |
|     Native)          |          |                        |
|                      | <------- |  YOLOv8n + EasyOCR     |
|  1. Abre camara      |   JSON   |  4. Detecta placa      |
|  2. Toma foto        |          |  5. Lee caracteres     |
|  3. Envia al server  |          |  6. Retorna resultado  |
|  7. Muestra resultado|          |                        |
|  8. Lee en voz alta  |          |                        |
+---------------------+          +------------------------+
```

## Estructura del Proyecto

```
lector-placas-parqueadero/
  backend/
    app.py                 <- API FastAPI (YOLOv8 + EasyOCR)
    requirements.txt       <- Dependencias de Python
    best.pt                <- Modelo entrenado (no incluido en git, pesa 6 MB)
  mobile/
    app/
      index.tsx            <- Pantalla principal de la app (camara + deteccion)
    app.json               <- Configuracion de Expo (permisos, plugins)
    package.json           <- Dependencias de Node.js
  notebooks/
    cuaderno_entrenamiento_placas.ipynb  <- Cuaderno de entrenamiento (para Colab)
  .gitignore
  README.md
```

## Tecnologias Utilizadas

| Componente | Tecnologia | Version |
|------------|-----------|---------|
| Modelo de deteccion | YOLOv8n (Ultralytics) | v8 nano |
| OCR | EasyOCR | espanol + ingles |
| Backend API | FastAPI + Uvicorn | Python 3.12 |
| Servidor en la nube | AWS EC2 (Ubuntu 24.04) | c7i.large.flex |
| App movil | React Native + Expo | SDK 54 |
| Entrenamiento | Google Colab | GPU T4 |
| Dataset | Roboflow - placas colombianas | 674 imagenes |

---

## Como Probar la Aplicacion

### Requisitos

- Un celular Android con [Expo Go](https://play.google.com/store/apps/details?id=host.exp.exponent) instalado
- Node.js 18+ en el computador ([descargar](https://nodejs.org))
- Conexion a internet

### Paso 1: Verificar que el backend esta corriendo

Abrir en el navegador:

```
http://3.148.246.40:8080/
```

Debe mostrar:

```json
{
  "status": "ok",
  "modelo": "best.pt",
  "descripcion": "ALPR Placas Colombianas - YOLOv8 + EasyOCR"
}
```

Tambien puede probar la API interactiva (Swagger) en:

```
http://3.148.246.40:8080/docs
```

### Paso 2: Probar el backend con una imagen (opcional)

Desde la terminal, enviar una imagen de un vehiculo con placa visible:

```bash
curl -X POST -F "file=@foto_carro.jpg" http://3.148.246.40:8080/predict/
```

Respuesta esperada:

```json
{
  "placas": ["ABC-123"],
  "cantidad": 1,
  "image": "<imagen procesada en base64>"
}
```

### Paso 3: Ejecutar la app movil

```bash
git clone https://github.com/leydyyohanamacareofuentes/lector-placas-parqueadero.git
cd lector-placas-parqueadero/mobile
npm install
npx expo start -c
```

Aparecera un codigo QR en la terminal.

### Paso 4: Abrir en el celular

1. Abrir **Expo Go** en el celular Android
2. Escanear el **codigo QR** que aparece en la terminal
3. El celular y el computador deben estar en la **misma red WiFi**

### Paso 5: Usar la app

1. En el campo **"IP del servidor"** escribir: `3.148.246.40`
2. En el campo **"Puerto"** dejar: `8080`
3. Apuntar la camara a un vehiculo con placa colombiana visible
4. Tocar el boton **"Detectar Placa"**
5. La app mostrara:
   - La imagen original con la placa marcada (bounding box verde)
   - El texto de la placa detectada (ej: JNU-540)
   - Leera la placa en voz alta

---

## Entrenamiento del Modelo

El modelo fue entrenado en Google Colab usando el cuaderno `notebooks/cuaderno_entrenamiento_placas.ipynb`.

### Dataset

**Fuente:** [placa-de-carro-sxy3a v4](https://universe.roboflow.com/cicatriz/placa-de-carro-sxy3a/dataset/4) en Roboflow

| Split | Imagenes |
|-------|----------|
| Entrenamiento | 674 |
| Validacion | 110 |
| Prueba | 52 |
| **Total** | **836** |

- **Clases:** 1 (placa)
- **Formato:** YOLOv8
- **Contenido:** Fotos de vehiculos colombianos con placas anotadas

### Parametros de Entrenamiento

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| Modelo base | `yolov8n.pt` | Transfer learning desde COCO |
| Epocas | 50 | Pasadas completas sobre el dataset |
| Tamano imagen | 640x640 | Resolucion de entrada |
| Batch size | 16 | Imagenes por lote |
| Early stopping | 10 epocas | Para si no mejora |
| GPU | T4 (Colab) | Aceleracion de entrenamiento |

### Como Reproducir el Entrenamiento

1. Abrir `notebooks/cuaderno_entrenamiento_placas.ipynb` en [Google Colab](https://colab.research.google.com)
2. Activar GPU: **Runtime > Change runtime type > T4 GPU**
3. Ejecutar todas las celdas en orden (Shift+Enter)
4. Al finalizar, el modelo `best.pt` se descarga automaticamente y se guarda en Google Drive

---

## Despliegue del Backend en AWS EC2

### 1. Crear instancia EC2

- **AMI:** Ubuntu Server 24.04 LTS
- **Tipo:** c7i.large.flex (o t2.large)
- **Almacenamiento:** 32 GB
- **Security Group:** abrir puertos 22 (SSH) y 8080 (API)
- **Key Pair:** crear y descargar archivo `.pem`

### 2. Conectarse al servidor

```bash
ssh -i "tu-clave.pem" ubuntu@<IP-PUBLICA>
```

### 3. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libglib2.0-0
```

### 4. Subir archivos al servidor

Desde el computador local:

```bash
scp -i "tu-clave.pem" backend/app.py backend/requirements.txt best.pt ubuntu@<IP-PUBLICA>:/home/ubuntu/proyecto/
```

### 5. Crear entorno virtual e instalar dependencias

```bash
cd /home/ubuntu/proyecto
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 6. Ejecutar el servidor

```bash
# En primer plano (para ver logs en tiempo real):
python app.py

# En background (para que no se apague al cerrar la terminal):
nohup python app.py > server.log 2>&1 &
```

La API queda disponible en `http://<IP-PUBLICA>:8080`

### Comandos Utiles

```bash
# Ver logs del servidor
cat /home/ubuntu/proyecto/server.log

# Detener el servidor
pkill -f app.py

# Reiniciar el servidor
pkill -f app.py && source venv/bin/activate && nohup python app.py > server.log 2>&1 &
```

---

## API - Documentacion de Endpoints

### GET /

Verificacion de estado del servidor.

**Respuesta:**
```json
{
  "status": "ok",
  "modelo": "best.pt",
  "descripcion": "ALPR Placas Colombianas - YOLOv8 + EasyOCR"
}
```

### POST /predict/

Detecta y lee placas vehiculares en una imagen.

**Entrada (dos opciones):**
- `file`: imagen como archivo multipart (desde Postman o curl)
- `image_base64`: imagen codificada en base64 (desde la app movil)

**Respuesta:**
```json
{
  "placas": ["JNU-540"],
  "cantidad": 1,
  "image": "<imagen anotada con bounding box en base64>"
}
```

**Documentacion interactiva (Swagger):** `http://<IP>:8080/docs`

---

## Notas Importantes

- El archivo `best.pt` (modelo entrenado) **no se incluye en el repositorio** por su peso. Se debe generar ejecutando el cuaderno de entrenamiento o solicitarlo a la estudiante.
- La app movil fue probada en Android con Expo Go SDK 54.

## Referencias

- [Repo template del profesor Alfredo Diaz (Deployment-Mobile-Yolo)](https://github.com/adiacla/Deployment-Mobile-Yolo)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Expo](https://expo.dev/)
- [Dataset en Roboflow](https://universe.roboflow.com/cicatriz/placa-de-carro-sxy3a/dataset/4)

## Licencia

Proyecto academico - UNAB 2026
