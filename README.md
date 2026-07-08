# 🚗 Sistema Inteligente de Detección de Daños en Vehículos con YOLOv8 + Groq AI

Sistema avanzado de peritaje y visión artificial diseñado para auditorías e inspecciones automotrices. El sistema utiliza **YOLOv8** para identificar daños en la carrocería, valida la presencia de vehículos de forma autónoma con un filtro COCO, y genera reportes detallados y unificados mediante **Groq AI (Llama 3)** y formato **PDF oficial**.

---

## 🛠️ Arquitectura del Sistema y Flujo de Trabajo

El sistema opera bajo un pipeline secuencial de doble validación para garantizar la precisión del diagnóstico:

```
[ Entrada ] ➔ 🖼️ Imagen(es) / 📷 Cámara Web / 🎥 Video (Muestreo 1 fps)
    │
    ▼
[ Validar Vehículo ] ➔ Filtro YOLOv8n (COCO) ➔ ¿Es un vehículo?
    │                                              ├── ❌ NO: Omitir con alerta
    │                                              └──  SI: Continuar
    ▼
[ Detección de Daños ] ➔ Inferencia YOLOv8s (Modelo Especializado)
    │
    ▼
[ Consolidación ] ➔ Unificar todos los daños en st.session_state
    │
    ▼
[ Diagnóstico con IA ] ➔ Enviar anomalías a Groq (Llama 3.3 70B)
    │
    ▼
[ Entregables ] ➔ 📄 Reporte de Texto (.txt) | ⬇️ Reporte Técnico (.pdf)
```

---

## ✨ Características Principales

*   **🔍 Filtro Antiespam de Entrada (Auto-Close-up):** Valida cada imagen de forma individual usando el modelo YOLOv8n (COCO) con un umbral de confianza mínimo de `0.15` para permitir primeros planos y tomas de detalle, bloqueando fotos de elementos ajenos (ej. mascotas, comida).
*   **📁 Inspección de Múltiples Imágenes (Estilo Aseguradora):** Permite subir hasta 10 fotos en una sola sesión (vistas frontal, trasera, laterales, acercamientos). El sistema acumula todos los daños identificados en un solo informe.
*   **🎥 Procesamiento de Video:** Admite subir archivos de video (`.mp4`, `.mov`, etc.). Realiza un **muestreo inteligente de 1 frame por segundo** utilizando OpenCV para ahorrar procesamiento, y añade las marcas de tiempo correspondientes a la galería (ej: `video.mp4 (Seg. 3)`).
*   **📷 Capturas desde Celular y Web:** Soporte nativo para tomar fotografías secuenciales con la webcam. En dispositivos móviles, el cargador de video permite abrir la cámara del celular y grabar en el acto.
*   **🤖 IA Explicativa Consolidada:** Utiliza Llama-3.3-70b a través de Groq para formular diagnósticos completos que explican la gravedad del estado general del vehículo y sugieren reparaciones.
*   **📄 Exportación Profesional:** Generación dinámica de reportes en PDF que incluyen la tabla de daños (indicando el origen/archivo de cada uno) y un registro fotográfico comparativo (Original vs. YOLO).

---

## 📁 Estructura del Proyecto

```
p3Aplicaciones/
├── data/
│   ├── raw/                    # Datasets descargados de Roboflow
│   │   ├── dataset1/           # Automobile Damage Detection
│   │   ├── dataset2/           # Car Damage Detection (Root)
│   │   └── dataset3/           # Car Damage Assessment
│   └── dataset_final/          # Dataset fusionado y listo para entrenar
├── notebooks/
│   └── entrenamiento.ipynb     # Notebook de Google Colab
├── models/
│   └── best.pt                 # Modelo especializado entrenado
├── app/
│   ├── main.py                 # Interfaz de usuario Streamlit
│   ├── inference.py            # Lógica de inferencia YOLOv8 y validación COCO
│   ├── llm_report.py           # Conexión Groq AI y prompt de diagnóstico
│   ├── pdf_generator.py        # Generador de reportes PDF profesional
│   └── utils.py                # Utilidades de traducción y formateo
├── scripts/
│   └── merge_datasets.py       # Script de fusión de datasets
├── .env.example                # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```

---

## 🏷️ Clases de Daño Detectadas (14 clases)

| ID | Clase | Traducción | ID | Clase | Traducción |
|----|-------|------------|----|-------|------------|
| **0** | `scratch` | Rayón | **7** | `bumper_damage` | Daño en Parachoques |
| **1** | `dent` | Abolladura | **8** | `door_damage` | Daño en Puerta |
| **2** | `crack` | Grieta de pintura | **9** | `fender_damage` | Daño en Guardafango |
| **3** | `glass_crack` | Parabrisas roto | **10** | `hood_damage` | Daño en Cofre |
| **4** | `headlight_damage` | Faro delantero | **11** | `trunk_damage` | Daño en Cajuela |
| **5** | `taillight_damage` | Luz trasera | **12** | `roof_damage` | Daño en Techo |
| **6** | `mirror_damage` | Espejo lateral | **13** | `rust` | Óxido / Corrosión |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio y crear entorno virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar en Linux/Mac
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
1. Copia el archivo `.env.example` y renombralo a `.env`:
   ```bash
   cp .env.example .env
   ```
2. Abre `.env` y coloca tu API Key de Groq:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
   *(Consigue tu llave gratis en [Groq Console](https://console.groq.com/))*

### 4. Fusionar datasets y entrenar (Opcional)
Si deseas entrenar el modelo tú mismo:
```bash
# Descarga los datasets en data/raw/
python scripts/merge_datasets.py
```
Sube la carpeta resultante `data/dataset_final/` a Drive y ejecuta el notebook `notebooks/entrenamiento.ipynb` en Colab. Luego coloca el archivo `best.pt` en la carpeta `models/`.

### 5. Ejecutar la Aplicación
```bash
streamlit run app/main.py
```

---

## 📊 Métricas del Modelo

*   **Modelo Base:** YOLOv8s (Ultralytics)
*   **Imágenes Utilizadas:** ~13,000 imágenes fusionadas
*   **Epocas de Entrenamiento:** 100 epochs (Google Colab T4 GPU)
*   **Métricas Obtenidas:**
    *   **mAP50:** `0.725` (Daños principales)
    *   **Precisión promedio:** `0.768`
    *   **Recall promedio:** `0.710`

---

## ⚙️ Limitaciones Conocidas

*   **Iluminación:** Fotos con reflejos excesivos o sombra total reducen la precisión.
*   **Visión 2D:** El LLM no puede ver las fotos directamente; razona basándose únicamente en los datos estructurados que le provee YOLOv8.
*   **CPU Inferencia:** En entornos de CPU estándar, el tiempo de análisis oscila entre `0.5` y `1.2` segundos por imagen/frame.
