"""
test_llm.py
-----------
Script para probar la capa de IA explicativa (Groq) sin necesitar el modelo YOLO.
Prueba múltiples casos borde para verificar que el prompt funciona correctamente.

Uso:
    cd c:/Users/Administrador/Desktop/p3Aplicaciones
    python scripts/test_llm.py
"""

import sys
import json
import time
from pathlib import Path

# Agregar app/ al path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from llm_report import generar_reporte

# ============================================================
# Casos de prueba
# ============================================================
CASOS = [
    {
        "nombre": "Sin daños (lista vacía)",
        "detecciones": [],
        "estado_esperado": "Sin daños",
    },
    {
        "nombre": "Daño leve (1 rayón)",
        "detecciones": [
            {"clase": "scratch", "clase_legible": "Rayón", "confianza": 0.72, "zona": "Lateral Izquierdo"},
        ],
        "estado_esperado": "Leve",
    },
    {
        "nombre": "Daño moderado (múltiples componentes)",
        "detecciones": [
            {"clase": "bumper_damage",    "clase_legible": "Parachoques",    "confianza": 0.88, "zona": "Inferior Central"},
            {"clase": "headlight_damage", "clase_legible": "Faro delantero", "confianza": 0.81, "zona": "Inferior Central"},
            {"clase": "door_damage",      "clase_legible": "Puerta",         "confianza": 0.65, "zona": "Lateral Derecho"},
            {"clase": "dent",             "clase_legible": "Abolladura",     "confianza": 0.77, "zona": "Lateral Derecho"},
        ],
        "estado_esperado": "Moderado",
    },
    {
        "nombre": "Daño grave (parabrisas + múltiples críticos)",
        "detecciones": [
            {"clase": "glass_crack",      "clase_legible": "Parabrisas dañado",  "confianza": 0.93, "zona": "Zona Central"},
            {"clase": "bumper_damage",    "clase_legible": "Parachoques",        "confianza": 0.89, "zona": "Inferior Central"},
            {"clase": "hood_damage",      "clase_legible": "Cofre",              "confianza": 0.85, "zona": "Zona Central"},
            {"clase": "headlight_damage", "clase_legible": "Faro delantero",     "confianza": 0.91, "zona": "Inferior Central"},
            {"clase": "mirror_damage",    "clase_legible": "Espejo lateral",     "confianza": 0.78, "zona": "Lateral Izquierdo"},
            {"clase": "dent",             "clase_legible": "Abolladura",         "confianza": 0.82, "zona": "Lateral Izquierdo"},
        ],
        "estado_esperado": "Grave",
    },
]


def separador(titulo: str):
    print("\n" + "=" * 55)
    print(f"  {titulo}")
    print("=" * 55)


def main():
    separador("TEST DE CAPA LLM (Groq)")
    print("Verificando conexión con Groq API...")
    print()

    resultados = []

    for i, caso in enumerate(CASOS, 1):
        nombre = caso["nombre"]
        detecciones = caso["detecciones"]
        esperado = caso["estado_esperado"]

        print(f"[{i}/{len(CASOS)}] {nombre}")
        print(f"       Detecciones: {len(detecciones)}")

        # Esperar entre peticiones (Groq es rápido, solo 1s de cortesía)
        if i > 1:
            print("       [INFO] Esperando 1s...")
            time.sleep(1)

        try:
            reporte = generar_reporte(detecciones)
            estado = reporte.get("estado", "?")
            justificacion = reporte.get("justificacion", "")

            # Verificar estado
            ok = esperado.lower() in estado.lower() or estado.lower() in esperado.lower()
            simbolo = "[OK]" if ok else "[WARNING]"

            print(f"       Reporte obtenido: {reporte}")
            print()

            resultados.append({
                "caso": nombre,
                "esperado": esperado,
                "obtenido": estado,
                "ok": ok,
                "justificacion": justificacion,
            })

        except RuntimeError as e:
            print(f"       [ERROR] ERROR DE CONFIGURACION: {e}")
            print()
            print("       Verifica que tu .env tenga GROQ_API_KEY configurada.")
            return
        except Exception as e:
            print(f"       [ERROR] ERROR INESPERADO: {e}")
            print()
            resultados.append({
                "caso": nombre,
                "esperado": esperado,
                "obtenido": "ERROR",
                "ok": False,
                "justificacion": str(e),
            })

    # Resumen
    separador("RESUMEN")
    aprobados = sum(1 for r in resultados if r["ok"])
    total = len(resultados)
    print(f"  Pruebas aprobadas: {aprobados}/{total}")
    print()

    for r in resultados:
        simbolo = "[OK]" if r["ok"] else "[ERROR]"
        print(f"  {simbolo}  {r['caso']}")
        if not r["ok"]:
            print(f"       Esperado: {r['esperado']} | Obtenido: {r['obtenido']}")

    if aprobados == total:
        print("\n[OK] La capa Groq funciona correctamente.")
        print("   El LLM responde coherentemente a todos los casos de prueba.")
    else:
        print(f"\n[WARNING] {total - aprobados} caso(s) con resultado inesperado.")
        print("   Revisa el prompt en app/llm_report.py si los estados no son correctos.")

    # Guardar log detallado
    log_path = Path(__file__).parent.parent / "scripts" / "test_llm_resultado.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n   Log completo guardado en: {log_path.name}")


if __name__ == "__main__":
    main()
