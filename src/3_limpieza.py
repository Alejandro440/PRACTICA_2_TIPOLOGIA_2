import pandas as pd
import numpy as np
import re

# =========================================================
# CONFIGURACIÓN
# =========================================================
INPUT_CSV = "ev_selected.csv"
OUTPUT_CSV = "ev_clean.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("LIMPIEZA DE LOS DATOS")
print("=" * 80)

# =========================================================
# CARGA DEL DATASET
# =========================================================
df = pd.read_csv(INPUT_CSV)

print("\nDimensiones iniciales del dataset:")
print(df.shape)

# =========================================================
# 3.1 IDENTIFICACIÓN DE VALORES FALTANTES
# =========================================================
print("\nValores nulos iniciales por columna:")
print(df.isna().sum())

# =========================================================
# FUNCIÓN DE CONVERSIÓN DE SUPERFICIES A m²
# =========================================================
def convertir_superficie(valor):
    """
    Convierte superficies expresadas en metros cuadrados (㎡, m²)
    o hectáreas (㏊) a metros cuadrados.
    """
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()

    if "㏊" in texto:
        try:
            num = texto.replace("㏊", "").replace(",", ".").strip()
            return float(num) * 10000
        except:
            return np.nan

    if "㎡" in texto or "m²" in texto:
        try:
            num = re.sub(r"[^\d,\.]", "", texto)
            num = num.replace(".", "").replace(",", ".")
            return float(num)
        except:
            return np.nan

    return np.nan

# =========================================================
# CONVERSIÓN DE VARIABLES DE SUPERFICIE
# =========================================================
df["superficie_construida_m2"] = df["superficie_construida"].apply(convertir_superficie)
df["superficie_parcela_m2"] = df["superficie_parcela"].apply(convertir_superficie)

df = df.drop(columns=["superficie_construida", "superficie_parcela"])

# =========================================================
# LIMPIEZA DE LA VARIABLE PRECIO
# =========================================================
def convertir_precio(valor):
    if pd.isna(valor):
        return np.nan

    texto = str(valor)
    texto = texto.replace("EUR", "").replace("€", "").strip()
    texto = texto.replace(".", "").replace(",", ".")
    try:
        return float(texto)
    except:
        return np.nan

df["precio"] = df["precio"].apply(convertir_precio)

# =========================================================
# CONVERSIÓN DE TIPOS
# =========================================================
df["habitaciones"] = df["habitaciones"].astype("Int64")
df["banos"] = df["banos"].astype("Int64")

df["zona"] = df["zona"].astype("category")
df["localizacion"] = df["localizacion"].astype("category")

# =========================================================
# IMPUTACIÓN CORRECTA: VALORES NO APLICABLES → 0
# =========================================================
cols_no_aplica = [
    "superficie_construida_m2",
    "superficie_parcela_m2",
    "habitaciones",
    "banos"
]

for col in cols_no_aplica:
    df[col] = df[col].fillna(0)

# =========================================================
# LIMITACIÓN DE PRECISIÓN NUMÉRICA
# =========================================================
float_cols = df.select_dtypes(include=["float64", "Float64"]).columns

for col in float_cols:
    df[col] = df[col].round(2)

# =========================================================
# RESUMEN FINAL
# =========================================================
print("\nResumen estadístico tras la limpieza:")
print(df.describe())

print("\nValores nulos finales por columna:")
print(df.isna().sum())

print("\nTipos de datos finales:")
print(df.dtypes)

# =========================================================
# GUARDADO DEL DATASET LIMPIO
# =========================================================
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDataset limpio guardado en: {OUTPUT_CSV}")
print("=" * 80)
