# Clasificador de Comentarios - API con FastAPI y HuggingFace 🤖🚀

Este proyecto es una API creada con FastAPI que clasifica comentarios de texto utilizando un modelo entrenado propio y alojado en la plataforam de Hugging Face.

En la carpeta de tu proyecto tu entorno virtual con python 3.10

#### - py -3.10 -m venv venv310
#### - venv310\Scripts\activate
Usar python 3.10

## Instalar librerías:
#### python.exe -m pip install --upgrade pip   
#### pip install python-dotenv
#### pip install fastapi transformers torch uvicorn
#### pip install httpx pytest

## Ejecutar 
python app.py

pytest test.py

## Documentación API
http://0.0.0.0:7860/docs  

## 🚀 Tecnologías Usadas
📦 Requisitos
- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Torch (PyTorch)](https://pytorch.org/)
- [Hugging Face Hub](https://huggingface.co/)
- [Uvicorn](https://www.uvicorn.org/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)
- [HTTPX](https://www.python-httpx.org/)
- [Pytest](https://docs.pytest.org/)

## 📁 Estructura del Proyecto
├── app.py # Archivo principal con la API

├── .env # Variables de entorno 
 
├── requirements.txt # Dependencias del proyecto
 
├── test.py # Archivo de pruebas (si tienes uno)
 
└── README.md # Este archivo
 
## 🧠 Modelo usado
Modelo: `pedrojm/modelv2_clasificacioncomentario`  
Pipeline: `text-classification` usando `transformers`

## 🚀 Cómo usar

### 1. Clonar el repositorio

### 2. Crear entorno virtual e instalar dependencias

py -3.10 -m venv venv310

venv310\Scripts\activate

pip install -r requirements.txt

##  3. Configurar variable de entorno
Crea un archivo .env con tu token de Hugging Face:

HF_TOKEN=tu_token_aqui

### 4. Ejecutar la API
python app.py

### 5. Probar la API

### GET /
Devuelve un mensaje de bienvenida para comprobar que la API está funcionando.

http://0.0.0.0:7860/ 

Respuesta:
{
  "message": "API funcionando."
}

### POST /predict
Recibe un comentario en formato JSON y devuelve su clasificación.

http://0.0.0.0:7860/predict 

Ejemplo:

POST /predict

Body: {
  "text": "Me encantó el producto"
}

Respuesta: {
  "etiqueta": "POSITIVO",
  "confianza": 0.9876
}

### 🧪 Tests

Puedes correr pruebas con:

pytest test.py

👨‍💻 Autor
Desarrollado por Pedro Jurado


