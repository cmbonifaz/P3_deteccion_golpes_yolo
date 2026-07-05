# Guía de Implementación: Sistema Inteligente de Detección de Daños en Vehículos con YOLO

Guía paso a paso para construir el proyecto desde cero: dataset → entrenamiento → aplicación → capa de IA explicativa.

---

## Fase 0: Preparación del entorno

1. Crea un entorno virtual de Python (recomendado 3.10 u 11):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```
2. Instala las dependencias base:
   ```bash
   pip install ultralytics opencv-python streamlit google-generativeai python-dotenv
   ```
3. Crea la estructura de carpetas del proyecto:
   ```
   proyecto_damage_detection/
   ├── data/
   │   ├── raw/              # datasets descargados sin procesar
   │   └── dataset_final/    # dataset fusionado, listo para entrenar
   ├── notebooks/
   │   └── entrenamiento.ipynb
   ├── models/
   │   └── best.pt           # modelo entrenado (se genera después)
   ├── app/
   │   ├── main.py           # app Streamlit
   │   ├── inference.py      # lógica de YOLO
   │   ├── llm_report.py     # capa de IA explicativa
   │   └── utils.py
   ├── .env                  # API key de Gemini (no subir a git)
   ├── requirements.txt
   └── README.md
   ```

---

## Fase 1: Obtención y preparación del dataset

1. **Descarga los tres datasets desde Roboflow Universe** en formato YOLO (v8/v11):
   - Automobile Damage Detection Dataset (dataset base, 14 clases)
   - Car Damage Detection (CAPSTONE) — complementario
   - Car Damage Detection (Oleksy1121) — complementario, cubre rayones

2. **Define tu lista final de clases** (unifica nombres repetidos entre datasets, ej. "dent" y "abolladura" deben quedar como una sola clase).

3. **Fusiona los datasets**:
   - Usa Roboflow (proyecto tipo "merge") o un script propio que:
     - Reescriba los índices de clase (`class_id`) según tu lista final unificada.
     - Copie imágenes y labels a `data/dataset_final/train`, `valid`, `test`.
   - Genera un `data.yaml` final con la ruta y la lista de clases.

4. **Revisa el balance de clases**: cuenta cuántas instancias hay por clase. Si alguna está muy por debajo del resto (ej. rayones), aplica data augmentation extra solo para esa clase (rotación, flip, brillo/contraste) usando Roboflow o `albumentations`.

5. **Verifica visualmente** una muestra de imágenes con sus bounding boxes para confirmar que las anotaciones estén correctas antes de entrenar.

---

## Fase 2: Entrenamiento del modelo YOLO

Trabaja en Google Colab (GPU gratuita) usando `notebooks/entrenamiento.ipynb`.

1. Instala Ultralytics en Colab:
   ```python
   !pip install ultralytics
   ```
2. Sube el dataset final (o móntalo desde Google Drive) y confirma la ruta del `data.yaml`.
3. Entrena un modelo base para validar el pipeline:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
   ```
4. Evalúa resultados:
   ```python
   metrics = model.val()
   ```
   Revisa mAP50, precisión/recall por clase, y la matriz de confusión (se genera en `runs/detect/train/`).
5. Si el pipeline funciona pero la precisión es baja, prueba `yolov8s` o `yolov8m`, ajusta `epochs`, o vuelve a la Fase 1 para mejorar el dataset.
6. Cuando estés satisfecho con las métricas, descarga el checkpoint final:
   ```python
   # se genera automáticamente en runs/detect/train/weights/best.pt
   ```
7. Copia ese archivo a `models/best.pt` en tu proyecto local.

---

## Fase 3: Lógica de inferencia (`app/inference.py`)

1. Carga el modelo entrenado y define una función que reciba una imagen y devuelva:
   - La imagen con las cajas delimitadoras dibujadas.
   - Una lista estructurada de detecciones (clase, confianza, coordenadas).

   ```python
   from ultralytics import YOLO
   import cv2

   model = YOLO("models/best.pt")

   def detectar_danos(ruta_imagen, conf_umbral=0.4):
       results = model.predict(ruta_imagen, conf=conf_umbral)
       imagen_anotada = results[0].plot()  # imagen con boxes dibujadas

       detecciones = []
       for box in results[0].boxes:
           clase = model.names[int(box.cls[0])]
           confianza = float(box.conf[0])
           x1, y1, x2, y2 = box.xyxy[0].tolist()
           detecciones.append({
               "clase": clase,
               "confianza": round(confianza, 2),
               "ubicacion": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
           })
       return imagen_anotada, detecciones
   ```

2. Define una función simple de **ubicación aproximada** (frontal/lateral/trasera/techo) dividiendo la imagen en zonas según las coordenadas `x, y` del centro de cada caja. Esto enriquece el JSON que luego recibirá el LLM.

---

## Fase 4: Capa de IA explicativa (`app/llm_report.py`)

1. Obtén una API key gratuita de **Google AI Studio** (Gemini) y guárdala en `.env`:
   ```
   GEMINI_API_KEY=tu_api_key_aqui
   ```

2. Diseña el prompt con restricciones anti-alucinación. Estructura recomendada:

   ```python
   import google.generativeai as genai
   import os, json
   from dotenv import load_dotenv

   load_dotenv()
   genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
   modelo_llm = genai.GenerativeModel("gemini-2.5-flash-lite")

   def generar_reporte(detecciones: list) -> str:
       prompt = f"""
       Eres un asistente que redacta reportes de inspección vehicular.
       Recibirás una lista de daños detectados por un modelo de visión por
       computadora (YOLO), en formato JSON. Tu tarea es:

       1. Calcular un estado general del vehículo (Leve, Moderado o Grave),
          basado ÚNICAMENTE en el tipo y la cantidad de daños listados.
       2. Redactar una breve justificación en lenguaje natural explicando
          por qué llegaste a esa conclusión.

       Reglas estrictas:
       - No inventes daños que no estén en la lista.
       - No asumas causas del daño (choque, granizo, etc.) si no hay evidencia.
       - Si la lista está vacía, indica que no se detectaron daños visibles.
       - Responde en español, en máximo 120 palabras.

       Daños detectados (JSON):
       {json.dumps(detecciones, ensure_ascii=False)}
       """
       respuesta = modelo_llm.generate_content(prompt)
       return respuesta.text
   ```

3. Prueba el prompt con casos límite: lista vacía, un solo daño leve, muchos daños graves — para confirmar que no "alucina" ni inventa información.

---

## Fase 5: Interfaz de usuario (`app/main.py`)

Usa Streamlit para una interfaz rápida de construir:

```python
import streamlit as st
from inference import detectar_danos
from llm_report import generar_reporte
from PIL import Image

st.title("Detección Automática de Daños en Vehículos")

archivo = st.file_uploader("Sube una foto del vehículo", type=["jpg", "jpeg", "png"])

if archivo:
    with open("temp.jpg", "wb") as f:
        f.write(archivo.read())

    imagen_anotada, detecciones = detectar_danos("temp.jpg")
    st.image(imagen_anotada, caption="Daños detectados", use_column_width=True)

    if st.button("Generar reporte"):
        with st.spinner("Analizando con IA explicativa..."):
            reporte = generar_reporte(detecciones)
        st.subheader("Reporte del estado del vehículo")
        st.write(reporte)

        st.subheader("Detalle de detecciones")
        st.table(detecciones)
```

Ejecuta con:
```bash
streamlit run app/main.py
```

---

## Fase 6: Pruebas y validación

1. Prueba con imágenes fuera del dataset de entrenamiento (fotos propias de carros) para evaluar generalización real.
2. Verifica que el reporte del LLM sea coherente con las detecciones (sin inventar daños).
3. Documenta casos donde el modelo falla (falsos positivos/negativos) — es valioso para la sección de conclusiones/limitaciones del informe final.

---

## Fase 7: Documentación para la defensa

Recopila lo siguiente desde `runs/detect/train/`:
- Curvas de entrenamiento (loss, mAP por época).
- Matriz de confusión.
- Ejemplos de detecciones correctas e incorrectas.
- Comparación breve de por qué YOLO (tiempo real) frente a alternativas como Faster R-CNN (más lento pero a veces más preciso).

---

## Checklist rápido

- [ ] Datasets descargados y fusionados con clases unificadas
- [ ] Dataset balanceado (augmentation en clases débiles)
- [ ] Modelo YOLO entrenado y validado (mAP aceptable)
- [ ] `best.pt` copiado al proyecto
- [ ] Función de inferencia probada
- [ ] API key de Gemini configurada y prompt probado con casos límite
- [ ] Interfaz Streamlit funcional end-to-end
- [ ] Pruebas con imágenes propias
- [ ] Gráficas y métricas listas para la defensa
