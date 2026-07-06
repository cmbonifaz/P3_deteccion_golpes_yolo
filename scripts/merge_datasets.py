"""
merge_datasets.py
-----------------
Script para fusionar los datasets de Roboflow en un dataset unificado
con clases normalizadas para YOLOv8.

Uso:
    python scripts/merge_datasets.py

Datasets esperados (dataset2 es OPCIONAL si no se pudo descargar):
    data/raw/dataset1/  → Automobile Damage Detection (14 clases con guiones)
                          Clases: Front-windscreen-damage, Headlight-damage, etc.
    data/raw/dataset2/  → Car Damage Detection por Root (opcional, 3008 imágenes)
                          Clases: Bodypanel-Dent, bonnet-dent, boot-dent, doorouter-dent,
                                  fender-dent, front-bumper-dent, Front-Windscreen-Damage,
                                  Headlight-Damage, pillar-dent, quaterpanel-dent,
                                  rear-bumper-dent, Rear-windscreen-Damage, roof-dent,
                                  RunningBoard-Dent, Sidemirror-Damage, Signlight-Damage,
                                  Taillight-Damage
                          URL: https://universe.roboflow.com/root-6ntpq/car-damage-detection-u0q4r
    data/raw/dataset3/  → Car Damage Assessment (3 clases simples)
                          Clases: Broken, Dent, Scratch

    Si dataset2/ está vacío o no existe, el script lo omite y avisa.
"""

import os
import shutil
import yaml
import re
from pathlib import Path

# ============================================================
# CONFIGURACIÓN: Lista unificada de 14 clases
# ============================================================
CLASES_FINALES = [
    "scratch",           # 0 — Rayón
    "dent",              # 1 — Abolladura
    "crack",             # 2 — Grieta/crack de pintura
    "glass_crack",       # 3 — Parabrisas agrietado
    "headlight_damage",  # 4 — Faro delantero dañado
    "taillight_damage",  # 5 — Luz trasera dañada
    "mirror_damage",     # 6 — Espejo lateral roto
    "bumper_damage",     # 7 — Parachoques dañado
    "door_damage",       # 8 — Daño en puerta
    "fender_damage",     # 9 — Guardafango dañado
    "hood_damage",       # 10 — Cofre dañado
    "trunk_damage",      # 11 — Cajuela dañada
    "roof_damage",       # 12 — Techo dañado
    "rust",              # 13 — Óxido/corrosión
]

# ============================================================
# MAPEO de nombres alternativos → nombre canónico
# Basado en los data.yaml REALES de los datasets descargados:
#
# Dataset 1 (Automobile Damage Detection v4):
#   'Front-windscreen-damage', 'Headlight-damage', 'Rear-windscreen-Damage',
#   'Runningboard-Damage', 'Sidemirror-Damage', 'Taillight-Damage',
#   'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
#   'front-bumper-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
#
# Dataset 2 (VEDD - si se descarga):
#   'Bumper', 'Door', 'Fender', 'Bonnet', 'Dickey', 'Light', 'Windshield'
#
# Dataset 3 (Car Damage Assessment v10):
#   'Broken', 'Dent', 'Scratch'
# ============================================================
MAPEO_NOMBRES = {
    # ── Dataset 3 ── (nombres exactos en minúsculas)
    "scratch": "scratch",
    "broken": "glass_crack",   # 'Broken' en DS3 = vidrio/parte rota → glass_crack
    "dent": "dent",

    # ── Dataset 1 ── (nombres exactos en minúsculas)
    "front-windscreen-damage": "glass_crack",
    "rear-windscreen-damage": "glass_crack",
    "headlight-damage": "headlight_damage",
    "taillight-damage": "taillight_damage",
    "sidemirror-damage": "mirror_damage",
    "runningboard-damage": "door_damage",   # running board → agrupa con puerta
    "bonnet-dent": "hood_damage",
    "boot-dent": "trunk_damage",
    "doorouter-dent": "door_damage",
    "fender-dent": "fender_damage",
    "front-bumper-dent": "bumper_damage",
    "rear-bumper-dent": "bumper_damage",
    "quaterpanel-dent": "fender_damage",    # cuarto de panel = guardafango extendido
    "roof-dent": "roof_damage",

    # ── Dataset 2 (Root - si se descarga) ── mismo esquema que DS1 pero con variaciones
    "bodypanel-dent": "dent",        # panel de carrocería genérico → abolladura
    "pillar-dent": "door_damage",    # pilar (entre puertas) → agrupa con puerta
    "runningboard-dent": "door_damage",  # estribera → zona lateral = puerta
    "signlight-damage": "taillight_damage",  # luz de dirección → luz trasera
    # Los demás nombres del dataset Root son idénticos al DS1 (ya cubiertós arriba)

    # ── Dataset 2 (VEDD - alternativa anterior, por si acaso) ──

    # ── Variantes genéricas (fallback) ──
    "scratches": "scratch",
    "paint_scratch": "scratch",
    "dents": "dent",
    "abolladura": "dent",
    "crack": "crack",
    "glass_crack": "glass_crack",
    "windscreen": "glass_crack",
    "headlight": "headlight_damage",
    "headlamp": "headlight_damage",
    "taillight": "taillight_damage",
    "mirror": "mirror_damage",
    "sidemirror": "mirror_damage",
    "front-bumper": "bumper_damage",
    "rear-bumper": "bumper_damage",
    "hood": "hood_damage",
    "trunk": "trunk_damage",
    "boot": "trunk_damage",
    "roof": "roof_damage",
    "rust": "rust",
    "corrosion": "rust",
}

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_FINAL = BASE_DIR / "data" / "dataset_final"
SPLITS = ["train", "valid", "test"]
DATASETS = ["dataset1", "dataset2", "dataset3"]

# Índice de clase final
CLASE_IDX = {nombre: idx for idx, nombre in enumerate(CLASES_FINALES)}


def leer_clases_dataset(dataset_path: Path) -> dict:
    """Lee el data.yaml de un dataset y devuelve {id_original: nombre_clase}."""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        # Buscar en subdirectorios
        yamls = list(dataset_path.rglob("data.yaml"))
        if not yamls:
            raise FileNotFoundError(f"No se encontró data.yaml en {dataset_path}")
        yaml_path = yamls[0]

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nombres = data.get("names", [])
    if isinstance(nombres, dict):
        return nombres  # {0: 'clase', 1: 'clase', ...}
    elif isinstance(nombres, list):
        return {i: n for i, n in enumerate(nombres)}
    else:
        raise ValueError(f"Formato de 'names' desconocido en {yaml_path}")


def mapear_clase(nombre_original: str) -> str | None:
    """
    Intenta mapear un nombre de clase original al nombre canónico.
    Devuelve None si no hay mapeo y la clase debe ser omitida.
    """
    nombre_lower = nombre_original.lower().strip()
    # Buscar directo
    if nombre_lower in MAPEO_NOMBRES:
        return MAPEO_NOMBRES[nombre_lower]
    # Buscar por contenido parcial
    for patron, canónico in MAPEO_NOMBRES.items():
        if patron in nombre_lower or nombre_lower in patron:
            return canónico
    # Sin mapeo encontrado
    return None


def procesar_label(
    label_path: Path,
    map_clase_id: dict,  # {id_original_int: id_final_int_o_None}
    dest_path: Path,
) -> int:
    """
    Reescribe un archivo de label YOLO con los nuevos class_id.
    Omite líneas cuya clase no tenga mapeo.
    Retorna el número de anotaciones escritas.
    """
    if not label_path.exists():
        return 0

    nuevas_lineas = []
    with open(label_path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.split()
            if len(partes) < 5:
                continue
            id_original = int(partes[0])
            id_final = map_clase_id.get(id_original)
            if id_final is None:
                continue  # clase no mapeada, se omite
            nuevas_lineas.append(f"{id_final} " + " ".join(partes[1:]))

    if nuevas_lineas:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(nuevas_lineas) + "\n")

    return len(nuevas_lineas)


def dataset_tiene_imagenes(ds_path: Path) -> bool:
    """Verifica que el dataset tenga al menos una imagen en train/."""
    train_images = ds_path / "train" / "images"
    if not train_images.exists():
        return False
    imgs = [f for f in train_images.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    return len(imgs) > 0


def merge():
    print("=" * 60)
    print("   FUSIÓN DE DATASETS — Sistema Detección de Daños")
    print("=" * 60)

    # Verificar datasets: dataset1 y dataset3 son obligatorios, dataset2 es opcional
    DATASETS_OBLIGATORIOS = ["dataset1", "dataset3"]
    for ds in DATASETS_OBLIGATORIOS:
        ds_path = DATA_RAW / ds
        if not ds_path.exists() or not dataset_tiene_imagenes(ds_path):
            print(f"\n❌ ERROR: No se encontró {ds_path} o está vacío.")
            print(f"   Descarga el dataset y extráelo en data/raw/{ds}/")
            return

    # dataset2 es opcional — informar si no está
    ds2_path = DATA_RAW / "dataset2"
    if not ds2_path.exists() or not dataset_tiene_imagenes(ds2_path):
        print("\n⚠️  dataset2/ está vacío o no existe — se omitirá.")
        print("   (Dataset VEDD opcional: https://universe.roboflow.com/car-damage-detection-t0g92-hjzfi/car-damage-detection)")
        datasets_a_procesar = ["dataset1", "dataset3"]
    else:
        datasets_a_procesar = DATASETS
        print(f"\n✅ dataset2/ encontrado — se incluirá en la fusión.")

    # Crear carpetas de destino
    for split in SPLITS:
        (DATA_FINAL / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_FINAL / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {split: {"imagenes": 0, "anotaciones": 0} for split in SPLITS}
    clases_no_mapeadas = set()

    # Procesar cada dataset
    for ds_nombre in datasets_a_procesar:
        ds_path = DATA_RAW / ds_nombre
        print(f"\n📂 Procesando {ds_nombre}...")

        # Leer clases del dataset
        try:
            clases_orig = leer_clases_dataset(ds_path)
        except FileNotFoundError as e:
            print(f"   ⚠️  {e} — saltando dataset")
            continue

        print(f"   Clases originales ({len(clases_orig)}): {list(clases_orig.values())}")

        # Construir mapa: id_original → id_final (o None si no se mapea)
        map_clase_id = {}
        for id_orig, nombre_orig in clases_orig.items():
            nombre_canonico = mapear_clase(nombre_orig)
            if nombre_canonico and nombre_canonico in CLASE_IDX:
                map_clase_id[id_orig] = CLASE_IDX[nombre_canonico]
                print(f"   ✅ {nombre_orig!r} → {nombre_canonico!r} (id {CLASE_IDX[nombre_canonico]})")
            else:
                map_clase_id[id_orig] = None
                clases_no_mapeadas.add(nombre_orig)
                print(f"   ⚠️  {nombre_orig!r} → sin mapeo (se omitirá)")

        # Mapa de prefijo corto para evitar MAX_PATH de Windows (260 chars)
        PREFIJO = {"dataset1": "d1", "dataset2": "d2", "dataset3": "d3"}
        prefijo = PREFIJO.get(ds_nombre, ds_nombre[:2])

        # Procesar cada split
        for split in SPLITS:
            images_dir = ds_path / split / "images"
            labels_dir = ds_path / split / "labels"

            if not images_dir.exists():
                continue

            # Crear directorios destino explícitamente
            dest_images_dir = DATA_FINAL / split / "images"
            dest_labels_dir = DATA_FINAL / split / "labels"
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            dest_labels_dir.mkdir(parents=True, exist_ok=True)

            imagenes_fallidas = 0
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                # Nombre corto: prefijo 2 chars + hash 8 chars + extensión
                # Esto garantiza que nunca se supere MAX_PATH en Windows
                ext = img_path.suffix.lower()
                nombre_base = img_path.stem[:40]  # truncar a 40 chars máx
                nuevo_nombre = f"{prefijo}_{nombre_base}{ext}"
                dest_img = dest_images_dir / nuevo_nombre

                try:
                    shutil.copy2(str(img_path), str(dest_img))
                    stats[split]["imagenes"] += 1
                except Exception as e:
                    imagenes_fallidas += 1
                    if imagenes_fallidas <= 3:
                        print(f"   ⚠️  No se pudo copiar: {img_path.name} → {e}")
                    continue

                # Procesar label correspondiente
                label_path = labels_dir / (img_path.stem + ".txt")
                dest_label = dest_labels_dir / f"{prefijo}_{nombre_base}.txt"
                n = procesar_label(label_path, map_clase_id, dest_label)
                stats[split]["anotaciones"] += n

            if imagenes_fallidas > 0:
                print(f"   ⚠️  {split}: {imagenes_fallidas} imágenes fallaron al copiarse")

    # Generar data.yaml final
    data_yaml = {
        "path": str(DATA_FINAL.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASES_FINALES),
        "names": CLASES_FINALES,
    }
    yaml_path = DATA_FINAL / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    # Resumen
    print("\n" + "=" * 60)
    print("   RESUMEN FINAL")
    print("=" * 60)
    total_imgs = sum(s["imagenes"] for s in stats.values())
    total_anots = sum(s["anotaciones"] for s in stats.values())
    for split, s in stats.items():
        print(f"   {split:6s}: {s['imagenes']:5d} imágenes, {s['anotaciones']:6d} anotaciones")
    print(f"   TOTAL : {total_imgs:5d} imágenes, {total_anots:6d} anotaciones")
    print(f"\n   Clases finales: {len(CLASES_FINALES)}")
    print(f"   data.yaml generado en: {yaml_path}")

    if clases_no_mapeadas:
        print(f"\n⚠️  Clases sin mapeo (omitidas): {clases_no_mapeadas}")
        print("   Si quieres incluirlas, agrega el mapeo en MAPEO_NOMBRES.")

    print("\n✅ Fusión completada. Revisa data/dataset_final/ antes de entrenar.")


if __name__ == "__main__":
    merge()
