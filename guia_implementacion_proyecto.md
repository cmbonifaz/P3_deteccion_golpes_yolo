# Guía de Implementación: Sistema Inteligente de Detección de Daños en Vehículos con YOLOv8 + Groq AI + Supabase

Guía paso a paso para construir el proyecto desde cero: dataset → entrenamiento → validación → inferencia con YOLO → diagnóstico con Groq AI → generación de reportes en PDF → almacenamiento híbrido local/nube (Supabase).

---

## 📁 Estructura General del Proyecto

```
p3Aplicaciones/
├── data/
│   ├── dataset_final/          # Dataset fusionado, listo para entrenar
│   └── reportes/               # Historial local de PDFs y registros JSON
├── notebooks/
│   ├── entrenamiento.ipynb     # Notebook de entrenamiento (Google Colab)
│   └── preparar_entrenamiento_kaggle.ipynb
├── models/
│   └── best.pt                 # Modelo especializado entrenado
├── app/
│   ├── main.py                 # Interfaz de usuario Streamlit
│   ├── inference.py            # Lógica de inferencia YOLOv8 y validación COCO
│   ├── llm_report.py           # Conexión Groq AI y prompt de diagnóstico
│   ├── pdf_generator.py        # Generador de reportes PDF profesional
│   ├── historial.py            # Lógica de persistencia híbrida en Supabase y Local
│   └── utils.py                # Utilidades de traducción y formateo
├── scripts/
│   └── merge_datasets.py       # Script de fusión de datasets
├── .env                    # Llaves de API de Groq y Supabase (No subir a Git)
├── .env.example
├── supabase_setup.md       # Guía de configuración para Supabase SQL y Storage
├── requirements.txt
└── README.md
```

---

## Fase 0: Preparación del Entorno

1. **Crea un entorno virtual de Python** (recomendado 3.10 u 11):
   ```bash
   python -m venv venv
   # En Windows
   .\venv\Scripts\activate
   # En Linux/Mac
   source venv/bin/activate
   ```
2. **Instala las dependencias del proyecto** (`requirements.txt`):
   ```
   streamlit>=1.35.0
   ultralytics>=8.1.0
   opencv-python-headless
   pandas
   python-dotenv
   groq>=0.5.0
   supabase>=2.4.0
   reportlab>=4.1.0
   pillow
   numpy
   ```
   Instálalas usando:
   ```bash
   pip install -r requirements.txt
   ```

---

## Fase 1: Preparación del Dataset

1. **Fusión de Datasets**:
   Utiliza el script `scripts/merge_datasets.py` para fusionar y mapear clases de diferentes datasets de Roboflow. El mapeo unifica las clases a las **14 categorías principales** de daños de la carrocería:
   - `scratch` (Rayón), `dent` (Abolladura), `crack` (Grieta), `glass_crack` (Parabrisas roto), `headlight_damage` (Faro roto), etc.
2. **Data Augmentation**: Aplica rotación, variación de brillo/contraste y recortes (especialmente en clases minoritarias como óxido o faros delanteros) para balancear la muestra.
3. Genera un archivo `data.yaml` con las rutas estructuradas de `train`, `val` y la descripción indexada de las clases.

---

## Fase 2: Entrenamiento del Modelo YOLO

Utiliza Google Colab con entorno GPU para entrenar tu detector:
```python
from ultralytics import YOLO

# 1. Cargar el modelo base preentrenado (YOLOv8s es idóneo por su relación precisión/velocidad)
model = YOLO("yolov8s.pt")

# 2. Iniciar el entrenamiento
model.train(
    data="data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    device=0
)

# 3. Evaluar el rendimiento en el conjunto de validación
metrics = model.val()
print("mAP50-95:", metrics.box.map)
```
Una vez finalizado, descarga el archivo `best.pt` generado en `runs/detect/train/weights/` y colócalo en el directorio local: `models/best.pt`.

---

## Fase 3: Pipeline de Doble Inferencia (`app/inference.py`)

Para evitar diagnósticos falsos (fotos de objetos ajenos a inspecciones como mascotas, recibos o paisajes), el sistema utiliza una arquitectura de **doble inferencia**:

1. **Filtro de Validación COCO**: Carga un modelo generalista (`yolov8n.pt`) para verificar si la imagen contiene al menos un vehículo (`car`, `bus` o `truck`) con un umbral bajo de confianza (`0.15` para tolerar encuadres cerrados).
2. **Modelo de Daños Especializado**: Si pasa la validación anterior, se ejecuta el modelo `models/best.pt` para detectar las 14 clases de daños del vehículo.

```python
# app/inference.py (Código simplificado)
from ultralytics import YOLO
import cv2

MODELO_DAÑOS = YOLO("models/best.pt")
MODELO_COCO = YOLO("yolov8n.pt") # Descarga automática de Ultralytics

def validar_vehiculo(ruta_imagen: str) -> bool:
    results = MODELO_COCO.predict(source=ruta_imagen, conf=0.15, verbose=False)
    for box in results[0].boxes:
        if int(box.cls[0]) in {2, 5, 7}: # car, bus, truck
            return True
    return False

def detectar_danos(ruta_imagen: str, conf_umbral: float = 0.4):
    if not validar_vehiculo(ruta_imagen):
        raise ValueError("No se detectó ningún vehículo en la imagen.")
    
    results = MODELO_DAÑOS.predict(ruta_imagen, conf=conf_umbral)
    imagen_anotada = results[0].plot()
    
    detecciones = []
    for box in results[0].boxes:
        detecciones.append({
            "clase": int(box.cls[0]),
            "confianza": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })
    return imagen_anotada, detecciones
```

---

## Fase 4: Capa de IA Explicativa (`app/llm_report.py`)

El sistema recopila todas las detecciones de daños de las fotos de inspección y le envía un JSON simplificado a **Llama 3.3 70B** a través de **Groq AI** para obtener un reporte técnico profesional e imparcial.

- **response_format={"type": "json_object"}**: Obliga al LLM a devolver un JSON con llaves exactas (`estado` y `justificacion`), asegurando un procesamiento robusto en la UI y la base de datos.
- **Categorías de Estado**:
  - `Sin daños`: Ningún daño detectado en ninguna de las imágenes.
  - `Leve`: Solo abolladuras o rayones menores aislados.
  - `Moderado`: Daños múltiples medianos en carrocería sin comprometer conducción.
  - `Grave`: Daños concurrentes en elementos críticos (luces + vidrios) o 5 o más daños totales.

```python
# app/llm_report.py (Código simplificado)
from groq import Groq
import json

client = Groq(api_key="TU_GROQ_API_KEY")

def generar_reporte(detecciones: list) -> dict:
    prompt = f"""
    Eres un perito experto. Analiza el siguiente listado de daños en formato JSON.
    Genera una respuesta en formato JSON con dos llaves:
    1. "estado": clasifica estrictamente en "Sin daños", "Leve", "Moderado" o "Grave".
    2. "justificacion": texto técnico detallado de 100-250 palabras en español enumerando los daños y explicando la gravedad.
    
    Daños detectados:
    {json.dumps(detecciones, ensure_ascii=False)}
    """
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Genera reportes de inspección de vehículos en JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)
```

---

## Fase 5: Persistencia Híbrida local y nube (`app/historial.py`)

Los reportes aprobados y sus archivos PDF oficiales correspondientes se almacenan en un entorno de persistencia híbrida:
- **Nube (Supabase)**: Inserta los datos estructurados y el estado del diagnóstico en la tabla PostgreSQL `reportes`, y carga los archivos PDF resultantes en el Storage Bucket `reportes-inspeccion`.
- **Local (Fallback)**: Si no hay conexión o no se configuran las credenciales en el archivo `.env`, se guardan localmente en la carpeta `data/reportes/` y se añade la información al archivo general `data/reportes/historial.json`.

*Consulta la estructura SQL y configuraciones completas del bucket en el archivo [supabase_setup.md](file:///c:/Users/Administrador/Desktop/p3Aplicaciones/supabase_setup.md).*

---

## Fase 6: Generador de Reportes PDF (`app/pdf_generator.py`)

Implementa la librería `reportlab` para ensamblar un informe PDF imprimible y exportable:
- **Estilo Institucional**: Encabezados sombreados, paleta de colores corporativos e información del inspector.
- **Tabla de Hallazgos**: Muestra de forma tabular todos los daños identificados con su respectiva fotografía de origen y nivel de confianza.
- **Evidencia Visual**: Mosaico comparativo que posiciona lado a lado la fotografía original e imagen anotada por YOLO de cada captura.
- **Prevención de Desbordamientos (Overflow-Proof)**: Envoltura de texto en celdas de tablas mediante `Paragraph` para prevenir truncamientos en descripciones y justificaciones extensas del LLM.

---

## Fase 7: Interfaz de Usuario y Experiencia en Streamlit (`app/main.py`)

La interfaz centraliza las fases en una experiencia rica para el usuario:
- **Directrices de Encuadre**: Recomendación en cabecera de porcentajes mínimos por vista (frontal $\ge$ 70%, lateral $\ge$ 60%, trasera $\ge$ 70%).
- **Entrada Multimedia Múltiple**: Carga de imágenes por lotes, transmisión de video en frames procesados a 1 FPS, y uso nativo de cámara del dispositivo.
- **Visualización Compacta y Centrada**: Layout de mosaico con columnas `[1.5, 2, 0.4, 2, 1.5]` para centrar y reducir el tamaño de las imágenes, con sus títulos de fotos (`📸 Foto X: <nombre>`) alineados a la izquierda.
- **Sesión Limpia al Navegar**: Al detectar el cambio de pestaña hacia la de "Historial de Reportes Guardados", la aplicación ejecuta una acción automatizada en segundo plano que limpia el estado temporal de inspección, evitando duplicados en la visualización.
- **Persistencia en la Descarga**: Los reportes y botones PDF generados se mantienen estables en caché a través de variables de sesión (`st.session_state`), previniendo recargas del LLM que consuman cuotas de la API.
- **Filtros Avanzados en Historial**: Búsqueda por placa, inspector, estado de gravedad, y rango de fechas de inspección.
