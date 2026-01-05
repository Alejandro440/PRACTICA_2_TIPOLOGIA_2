import pandas as pd

# =========================================================
# CONFIGURACIÓN
# =========================================================
INPUT_CSV = "ev_mallorca_scrape.csv"
OUTPUT_CSV = "ev_selected.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("INTEGRACIÓN Y SELECCIÓN DE LOS DATOS")
print("=" * 80)

# =========================================================
# CARGA DEL DATASET
# =========================================================
df = pd.read_csv(INPUT_CSV)

print("\nDataset cargado")
print(f"Dimensiones originales: {df.shape}")

# =========================================================
# RENOMBRADO DE COLUMNAS (.svg → nombres semánticos)
# =========================================================
rename_map = {}

for col in df.columns:
    if col.endswith(".svg"):
        if "livingSpace" in col:
            rename_map[col] = "superficie_construida"
        elif "propertyArea" in col:
            rename_map[col] = "superficie_parcela"
        elif "bedroom" in col:
            rename_map[col] = "habitaciones"
        elif "bathroom" in col:
            rename_map[col] = "banos"
        else:
            rename_map[col] = col.replace(".svg", "")

df = df.rename(columns=rename_map)

print("\nColumnas renombradas:")
for k, v in rename_map.items():
    print(f"{k} -> {v}")

# =========================================================
# SELECCIÓN DE VARIABLES RELEVANTES
# =========================================================
# Variables seleccionadas en función del objetivo analítico:
# estimación del precio del inmueble
selected_columns = [
    "superficie_construida",
    "superficie_parcela",
    "habitaciones",
    "banos",
    "zona",
    "localizacion",
    "precio"
]

df_selected = df[selected_columns]

print("\nVariables seleccionadas para el análisis:")
print(df_selected.columns.tolist())

print("\nDimensiones tras selección:")
print(df_selected.shape)

# =========================================================
# RESUMEN INICIAL DE LOS DATOS
# =========================================================
print("\nResumen inicial del dataset seleccionado:")
print(df_selected.info())

print("\nEjemplo de registros:")
print(df_selected.head())

# =========================================================
# GUARDADO
# =========================================================
df_selected.to_csv(OUTPUT_CSV, index=False)

print(f"\nDataset seleccionado guardado en: {OUTPUT_CSV}")
print("=" * 80)
