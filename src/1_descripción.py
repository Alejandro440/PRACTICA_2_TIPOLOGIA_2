import pandas as pd
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================
CSV_PATH = "ev_mallorca_scrape.csv"
N_PREVIEW = 5

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("DIAGNÓSTICO INICIAL DEL DATASET")
print("=" * 80)

# =========================================================
# CARGA DEL DATASET
# =========================================================
df = pd.read_csv(CSV_PATH)

print("\nDataset cargado")
print(f"Número de filas: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")

# =========================================================
# PREVISUALIZACIÓN
# =========================================================
print("\nPrimeras filas del dataset:")
print(df.head(N_PREVIEW))

# =========================================================
# TIPOS DE DATOS
# =========================================================
print("\nTipos de datos detectados:")
print(df.dtypes)

# =========================================================
# VALORES NULOS EXPLÍCITOS
# =========================================================
print("\nValores nulos explícitos por columna:")
nulls = df.isna().sum().sort_values(ascending=False)
nulls = nulls[nulls > 0]

if nulls.empty:
    print("No se han detectado valores nulos explícitos")
else:
    print(nulls)

# =========================================================
# POSIBLES NULOS IMPLÍCITOS
# =========================================================
PLACEHOLDERS = ["", " ", "NA", "N/A", "null", "None", "-", "--"]

print("\nPosibles valores que indican datos faltantes (placeholders):")

found = False
for col in df.columns:
    if df[col].dtype == "object":
        count = df[col].isin(PLACEHOLDERS).sum()
        if count > 0:
            print(f"{col}: {count}")
            found = True

if not found:
    print("No se han detectado placeholders comunes")

# =========================================================
# CEROS EN VARIABLES NUMÉRICAS
# =========================================================
print("\nCeros en variables numéricas:")

found = False
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            print(f"{col}: {zeros}")
            found = True

if not found:
    print("No se han detectado ceros en variables numéricas")

# =========================================================
# ANÁLISIS PRELIMINAR DE PRECIO
# =========================================================
print("\nAnálisis preliminar de la variable precio:")

precio_cols = [c for c in df.columns if "precio" in c.lower()]

if not precio_cols:
    print("No se ha detectado ninguna columna relacionada con el precio")
else:
    for col in precio_cols:
        print(f"\nColumna: {col}")
        print(f"Tipo: {df[col].dtype}")
        print("Ejemplos de valores:")
        print(df[col].dropna().head(10))

# =========================================================
# VARIABLES CATEGÓRICAS DE ALTA CARDINALIDAD
# =========================================================
print("\nVariables categóricas con alta cardinalidad:")

found = False
for col in df.columns:
    if df[col].dtype == "object":
        nunique = df[col].nunique(dropna=True)
        if nunique > 50:
            print(f"{col}: {nunique} valores distintos")
            found = True

if not found:
    print("No se han detectado variables categóricas de alta cardinalidad")

# =========================================================
# RESUMEN
# =========================================================
print("\nResumen del diagnóstico:")
print("- Dataset obtenido mediante scraping web")
print("- Tipos de datos no normalizados")
print("- Presencia potencial de valores faltantes")
print("- Variable precio pendiente de transformación")
print("=" * 80)
