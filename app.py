# -*- coding: utf-8 -*-
"""
App web para predecir si un telefono usado se revendera a precio ALTO o BAJO.
Carga el modelo entrenado en el paso 2 (modelo_precio.joblib) y lo usa desde un formulario.
"""
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- Cargar el modelo una sola vez, cuando arranca la app ---
RUTA_MODELO = os.path.join(os.path.dirname(__file__), 'modelo_precio.joblib')
artefacto = joblib.load(RUTA_MODELO)
modelo = artefacto['modelo']
COLUMNAS = artefacto['columnas']          # orden exacto de columnas que espera el modelo
CLASES = artefacto['clases']              # {0: 'Precio bajo', 1: 'Precio alto'}
MEDIANA = artefacto['mediana_precio']     # el corte alto/bajo
NOMBRE_MODELO = artefacto['nombre']       # 'AdaBoost' o 'RandomForest'

# Estas son las columnas categoricas (las mismas que use con get_dummies en el paso 2)
CATEGORICAS = ['brand', 'os_type', 'condition', 'city_tier', 'seller_type']

# --- Descripcion de cada campo del formulario ---
# Cada campo numerico: (nombre, etiqueta, valor_por_defecto, minimo, maximo, paso)
CAMPOS_NUMERICOS = [
    ('release_year', 'Anio de lanzamiento', 2023, 2019, 2025, 1),
    ('ram_gb', 'RAM (GB)', 8, 4, 16, 1),
    ('storage_gb', 'Almacenamiento (GB)', 256, 64, 1024, 1),
    ('screen_size_inches', 'Tamanio de pantalla (pulgadas)', 6.5, 5.5, 7.0, 0.1),
    ('battery_capacity', 'Capacidad de bateria (mAh)', 4500, 3000, 6500, 50),
    ('processor_score', 'Puntaje del procesador (40-100)', 75, 40, 100, 1),
    ('camera_score', 'Puntaje de camara (40-100)', 80, 40, 100, 1),
    ('has_5g', 'Tiene 5G (1 = si, 0 = no)', 1, 0, 1, 1),
    ('original_price', 'Precio original', 60000, 10000, 150000, 100),
    ('purchase_year', 'Anio de compra', 2023, 2019, 2025, 1),
    ('age_months', 'Edad del telefono (meses)', 24, 1, 71, 1),
    ('usage_hours_per_day', 'Horas de uso por dia', 4.0, 1.0, 12.0, 0.5),
    ('battery_health', 'Salud de la bateria (%)', 85, 55, 100, 1),
    ('screen_cracked', 'Pantalla rota (1 = si, 0 = no)', 0, 0, 1, 1),
    ('body_damage', 'Danio en el cuerpo (1 = si, 0 = no)', 0, 0, 1, 1),
    ('repair_history', 'Fue reparado antes (1 = si, 0 = no)', 0, 0, 1, 1),
    ('water_damage', 'Danio por agua (1 = si, 0 = no)', 0, 0, 1, 1),
    ('warranty_remaining_months', 'Garantia restante (meses)', 6, 0, 24, 1),
    ('box_available', 'Tiene caja (1 = si, 0 = no)', 1, 0, 1, 1),
    ('charger_available', 'Tiene cargador (1 = si, 0 = no)', 1, 0, 1, 1),
    ('market_demand_score', 'Demanda de mercado (40-100)', 70, 40, 100, 1),
]

# Cada campo categorico: nombre -> (etiqueta, [opciones])
CAMPOS_CATEGORICOS = {
    'brand':       ('Marca', ['Apple', 'Google', 'OnePlus', 'Realme', 'Samsung', 'Vivo', 'Xiaomi']),
    'os_type':     ('Sistema operativo', ['Android', 'iOS']),
    'condition':   ('Condicion', ['Excellent', 'Good', 'Fair', 'Poor']),
    'city_tier':   ('Nivel de ciudad', ['Tier1', 'Tier2', 'Tier3']),
    'seller_type': ('Tipo de vendedor', ['Individual', 'Store']),
}


def construir_fila(form):
    """Convierte los datos del formulario en una fila con las columnas que el modelo espera."""
    datos = {}

    # 1) campos numericos: los leo y los paso a numero
    for nombre, _etq, _def, _mn, _mx, _paso in CAMPOS_NUMERICOS:
        datos[nombre] = float(form.get(nombre))

    # 2) campos categoricos: los leo como texto
    for nombre in CAMPOS_CATEGORICOS:
        datos[nombre] = form.get(nombre)

    # 3) armo un DataFrame de una sola fila
    fila = pd.DataFrame([datos])

    # 4) aplico get_dummies igual que en el entrenamiento
    fila = pd.get_dummies(fila, columns=CATEGORICAS)

    # 5) reordeno para que tenga EXACTAMENTE las columnas del modelo.
    #    Las que falten se rellenan con 0 (esto reproduce el drop_first del paso 2).
    fila = fila.reindex(columns=COLUMNAS, fill_value=0)
    return fila


@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None

    if request.method == 'POST':
        fila = construir_fila(request.form)
        proba_alto = modelo.predict_proba(fila)[0, 1]   # probabilidad de precio alto
        clase = 1 if proba_alto >= 0.5 else 0
        resultado = {
            'etiqueta': CLASES[clase],
            'proba_alto': round(proba_alto * 100, 1),
            'es_alto': clase == 1,
        }

    return render_template(
        'index.html',
        campos_num=CAMPOS_NUMERICOS,
        campos_cat=CAMPOS_CATEGORICOS,
        resultado=resultado,
        mediana=round(MEDIANA, 2),
        nombre_modelo=NOMBRE_MODELO,
        valores=request.form,  # para no perder lo que el usuario escribio
    )


if __name__ == '__main__':
    # debug=True para desarrollo local (se reinicia solo al guardar cambios)
    app.run(debug=True)
