# -*- coding: utf-8 -*-
"""
preparar_entrenamiento_kaggle.py
---------------------------------
Copia el contenido de este archivo en celdas de tu Notebook de Kaggle o Google Colab.
Este script automatiza:
1. La conversión de anotaciones COCO JSON a formato YOLO TXT.
2. La creación del archivo data.yaml.
3. El inicio del entrenamiento/fine-tuning.
"""

import json
import os
import shutil
from pathlib import Path

# =====================================================================
# 1. FUNCIÓN DE CONVERSIÓN COCO JSON A YOLO TXT
# =====================================================================
def convertir_coco_a_yolo(json_path, images_dir, output_dir, class_mapping):
    """
    Convierte anotaciones COCO JSON a archivos de etiquetas individuales YOLO TXT.
    
    Args:
        json_path (str): Ruta al archivo .json de anotaciones COCO.
        images_dir (str): Directorio con las imágenes originales.
        output_dir (str): Directorio destino para 'images' y 'labels'.
        class_mapping (dict): Diccionario de mapeo {id_clase_coco: id_clase_yolo}.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"⚠️ Archivo JSON no encontrado: {json_path}")
        return
        
    print(f"🔄 Cargando anotaciones desde {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)
        
    print(f"✏️ Procesando {len(images)} imágenes...")
    convertidas = 0
    
    for img_id, anns in annotations_by_image.items():
        if img_id not in images:
            continue
        img_info = images[img_id]
        filename = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        src_image_path = os.path.join(images_dir, filename)
        dst_image_path = os.path.join(output_dir, "images", filename)
        
        # Copiar imagen si existe
        if not os.path.exists(src_image_path):
            continue
            
        shutil.copy(src_image_path, dst_image_path)
        
        # Crear archivo .txt de etiquetas
        txt_filename = Path(filename).with_suffix('.txt')
        txt_path = os.path.join(output_dir, "labels", txt_filename)
        
        with open(txt_path, 'w') as out_f:
            for ann in anns:
                coco_cat_id = ann['category_id']
                if coco_cat_id not in class_mapping:
                    continue
                yolo_class_id = class_mapping[coco_cat_id]
                
                bbox = ann['bbox'] # [x_min, y_min, width, height]
                x_center = (bbox[0] + bbox[2] / 2.0) / img_w
                y_center = (bbox[1] + bbox[3] / 2.0) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h
                
                out_f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        convertidas += 1
        
    print(f"✅ Conversión finalizada. {convertidas} imágenes preparadas en {output_dir}")

# =====================================================================
# 2. CONFIGURACIÓN DEL ARCHIVO DATA.YAML
# =====================================================================
def crear_data_yaml():
    yaml_content = """
path: /kaggle/working/dataset_final
train: images/train
val: images/val

names:
  0: scratch
  1: dent
  2: crack
  3: glass_crack
  4: headlight_damage
  5: taillight_damage
  6: mirror_damage
  7: bumper_damage
  8: door_damage
  9: fender_damage
  10: hood_damage
  11: trunk_damage
  12: roof_damage
  13: rust
"""
    with open('/kaggle/working/data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    print("✅ Archivo /kaggle/working/data.yaml generado.")

# =====================================================================
# 3. ENTRENAMIENTO DE YOLOv8 (FINE-TUNING)
# =====================================================================
# Ejecutar en tu celda de Notebook:
# 
# !pip install ultralytics -q
# 
# from ultralytics import YOLO
# 
# # Cargar tus pesos previos (sube 'best.pt' como dataset de entrada a Kaggle)
# model = YOLO('/kaggle/input/tus-pesos-previos/best.pt')
# 
# # Iniciar entrenamiento
# model.train(
#     data='/kaggle/working/data.yaml',
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     device=0,
#     project='inspeccion',
#     name='yolov8_danos_mejorado'
# )
