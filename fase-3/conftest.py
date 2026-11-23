"""Hace importable el paquete de la Fase 3 al ejecutar pytest.

Las pruebas de la API importan `main`, que vive en esta carpeta. Al correr
`python -m pytest` desde la raíz del repositorio, pytest solo agrega al
`sys.path` el directorio de cada archivo de prueba, así que ese import falla y
la recolección se interrumpe para todas las suites. Añadir esta carpeta aquí
funciona igual desde la raíz y desde `fase-3/`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
