# Sistema Inteligente de Detección de Daños en Vehículos con YOLO

Sistema de visión por computadora que detecta daños en vehículos mediante **YOLOv8** y genera reportes explicativos usando **Groq (Llama 3)**.

---

## Arquitectura del Sistema

```
Imagen → Preprocesamiento (OpenCV) → Inferencia YOLOv8 → JSON de detecciones
       → Groq Llama 3 (LLM) → Reporte explicado → Interfaz Streamlit
```

## Estructura del Proyecto

```
p3Aplicaciones/
├── data/
│   ├── raw/                    # Datasets descargados (vacío en Git)
│   │   ├── dataset1/           # Automobile Damage Detection
│   │   ├── dataset2/           # Car Damage Detection CAPSTONE
│   │   └── dataset3/           # Car Damage Scratches (Oleksy1121)
│   └── dataset_final/          # Dataset fusionado y listo para entrenar
├── notebooks/
│   └── entrenamiento.ipynb     # Notebook de Google Colab
├── models/
│   └── best.pt                 # Modelo entrenado (se agrega después)
├── app/
│   ├── main.py                 # Interfaz Streamlit
│   ├── inference.py            # Lógica de inferencia YOLO
│   ├── llm_report.py           # Capa de IA explicativa (Groq)
│   └── utils.py                # Funciones auxiliares
├── scripts/
│   └── merge_datasets.py       # Script de fusión de datasets
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Clases de Daño Detectadas (14 clases)

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `scratch` | Rayón en la pintura |
| 1 | `dent` | Abolladura |
| 2 | `crack` | Grieta/crack de pintura |
| 3 | `glass_crack` | Parabrisas agrietado |
| 4 | `headlight_damage` | Faro delantero dañado |
| 5 | `taillight_damage` | Luz trasera dañada |
| 6 | `mirror_damage` | Espejo lateral roto |
| 7 | `bumper_damage` | Parachoques dañado |
| 8 | `door_damage` | Daño en puerta |
| 9 | `fender_damage` | Guardafango dañado |
| 10 | `hood_damage` | Cofre dañado |
| 11 | `trunk_damage` | Cajuela dañada |
| 12 | `roof_damage` | Techo dañado |
| 13 | `rust` | Óxido/corrosión |

---

## Instalación

### 1. Prerequisitos
- Python 3.10 o 3.11
- GPU recomendada para entrenamiento (el entrenamiento se hace en Google Colab)

### 2. Clonar el repositorio y crear entorno virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
# Copia la plantilla
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# Edita .env y agrega tu API key de Groq
# Obtén la tuya en: https://console.groq.com/
```

---

## Descarga de Datasets

> ⚠️ **Esta sección es acción manual requerida.**

Necesitas descargar **3 datasets** de Roboflow Universe. Para cada uno sigue estos pasos:

### Paso a paso para descargar en Roboflow:
1. Ve a la URL del dataset (listadas abajo)
2. Crea una cuenta gratuita en Roboflow si no tienes una
3. Haz clic en el botón **"Download Dataset"** (o "Export Dataset")
4. Selecciona formato: **YOLOv8** (importante: exactamente esta versión)
5. Elige **"download zip to computer"**
6. Descarga el archivo ZIP

### Dataset 1 — Automobile Damage Detection (Base Principal) ✅
- **URL:** https://universe.roboflow.com/automobile-damage-detection/automobile-damage-detection
- **Tamaño:** ~6,900 imágenes — 14 clases (faros, luces, espejos, parachoques, puertas, cofre, etc.)
- **Guardar en:** `data/raw/dataset1/`

### Dataset 2 — Car Damage Detection por Root (Complementario — OPCIONAL)
- **URL:** https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r
- **Tamaño:** ~3,008 imágenes — mismas clases detalladas que Dataset 1 (distinta fuente de imágenes)
- **Guardar en:** `data/raw/dataset2/`
- > ⚠️ Este dataset es **opcional**. Si no está disponible, el script de fusión lo omite automáticamente.

### Dataset 3 — Car Damage Assessment (Rayones/Roturas) ✅
- **URL:** https://universe.roboflow.com/project-joggx/car-damage-assessment-8mb45
- **Tamaño:** ~3,700 imágenes — Scratch, Dent, Broken
- **Guardar en:** `data/raw/dataset3/`

### Estructura esperada tras la descarga:
```
data/raw/
├── dataset1/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── dataset2/    (misma estructura)
└── dataset3/    (misma estructura)
```

### Fusionar los datasets:
```bash
python scripts/merge_datasets.py
```
Esto genera `data/dataset_final/` con todas las imágenes unificadas y el `data.yaml` final.

---

## Entrenamiento en Google Colab

1. Sube `notebooks/entrenamiento.ipynb` a Google Drive
2. Sube la carpeta `data/dataset_final/` a Google Drive
3. Abre el notebook en Colab (**Runtime → Change runtime type → GPU T4**)
4. Ejecuta todas las celdas en orden
5. Al finalizar, descarga `best.pt` desde la celda de descarga
6. Copia `best.pt` a `models/best.pt` en tu proyecto local

---

## Ejecutar la Aplicación

```bash
# Asegúrate de tener models/best.pt y .env configurado
streamlit run app/main.py
```

La app abrirá en tu navegador en `http://localhost:8501`

---

## Métricas del Modelo (completar tras entrenamiento)

| Métrica | Valor |
|---------|-------|
| mAP50 | — |
| Precisión | — |
| Recall | — |
| Epochs | 100 |
| Modelo base | YOLOv8s |
| Imágenes de entrenamiento | ~10,000+ |

---

## Dependencias Principales

| Librería | Versión | Uso |
|----------|---------|-----|
| ultralytics | ≥8.3.0 | Modelo YOLOv8 |
| opencv-python | ≥4.9.0 | Procesamiento de imágenes |
| streamlit | ≥1.35.0 | Interfaz de usuario |
| groq | ≥0.9.0 | API Groq LLM (Llama 3) |
| python-dotenv | ≥1.0.0 | Variables de entorno |

---

## Limitaciones Conocidas

- El modelo requiere fotografías con buena iluminación y ángulo razonable
- Daños muy pequeños o en ángulos extremos pueden no ser detectados
- El LLM solo razona sobre los daños que YOLO detecta (no "inventa" daños)
- Tiempo de inferencia en CPU: ~2-5 segundos por imagen

---

## Créditos y Datasets

- [Automobile Damage Detection Dataset](https://universe.roboflow.com/automobile-damage-detection/automobile-damage-detection) — Roboflow Universe
- [Car Damage Detection CAPSTONE](https://universe.roboflow.com/capstone-j0d6y/car-damage-detection) — Roboflow Universe  
- [Car Damage Assessment Dataset](https://universe.roboflow.com/project-joggx/car-damage-assessment-8mb45) — Roboflow Universe
- Modelo LLM: Groq (Llama 3)
