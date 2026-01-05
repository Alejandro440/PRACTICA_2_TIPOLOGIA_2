import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================
INPUT_CSV = "ev_clean.csv"
OUTPUT_DIR = "figuras"
LOW_Q = 0.01
HIGH_Q = 0.99

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# =========================================================
# CARGA
# =========================================================
df = pd.read_csv(INPUT_CSV)

# =========================================================
# FUNCIÓN AUXILIAR: LÍMITES POR PERCENTILES (SOLO VISUAL)
# =========================================================
def limits_by_quantile(series, q_low=LOW_Q, q_high=HIGH_Q):
    return series.quantile(q_low), series.quantile(q_high)

# =========================================================
# 5.1 DISTRIBUCIÓN DEL PRECIO (RECORTADA)
# =========================================================
low, high = limits_by_quantile(df["precio"].dropna())

plt.figure(figsize=(8, 5))
sns.histplot(df["precio"].dropna(), bins=50, kde=True)
plt.xlim(low, high)
plt.title("Distribución del precio de los inmuebles (recorte visual 1%-99%)")
plt.xlabel("Precio (€)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribucion_precio.png")
plt.close()

# =========================================================
# 5.2 DISTRIBUCIÓN DE SUPERFICIE CONSTRUIDA (RECORTADA)
# =========================================================
low, high = limits_by_quantile(df["superficie_construida_m2"])

plt.figure(figsize=(8, 5))
sns.histplot(df["superficie_construida_m2"], bins=50, kde=True)
plt.xlim(low, high)
plt.title("Distribución de la superficie construida (recorte visual 1%-99%)")
plt.xlabel("Superficie construida (m²)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribucion_superficie_construida.png")
plt.close()

# =========================================================
# 5.3 PRECIO VS SUPERFICIE (RECORTE EN AMBOS EJES)
# =========================================================
x_low, x_high = limits_by_quantile(df["superficie_construida_m2"])
y_low, y_high = limits_by_quantile(df["precio"])

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="superficie_construida_m2",
    y="precio",
    alpha=0.5
)
plt.xlim(x_low, x_high)
plt.ylim(y_low, y_high)
plt.title("Precio vs superficie construida (recorte visual 1%-99%)")
plt.xlabel("Superficie construida (m²)")
plt.ylabel("Precio (€)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precio_vs_superficie.png")
plt.close()

# =========================================================
# 5.4 PRECIO POR HABITACIONES (SIN RECORTE NECESARIO)
# =========================================================
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="habitaciones",
    y="precio"
)
plt.ylim(y_low, y_high)
plt.title("Precio según número de habitaciones (recorte visual 1%-99%)")
plt.xlabel("Número de habitaciones")
plt.ylabel("Precio (€)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precio_por_habitaciones.png")
plt.close()

# =========================================================
# 5.5 PRECIO POR ZONA (RECORTADO)
# =========================================================
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="zona",
    y="precio"
)
plt.ylim(y_low, y_high)
plt.xticks(rotation=45)
plt.title("Distribución del precio por zona (recorte visual 1%-99%)")
plt.xlabel("Zona")
plt.ylabel("Precio (€)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precio_por_zona.png")
plt.close()

# =========================================================
# 5.6 CLUSTERING POR SUPERFICIES (RECORTADO)
# =========================================================
if "cluster" in df.columns:
    x_low, x_high = limits_by_quantile(df["superficie_construida_m2"])
    y_low, y_high = limits_by_quantile(df["superficie_parcela_m2"])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="superficie_construida_m2",
        y="superficie_parcela_m2",
        hue="cluster",
        palette="tab10",
        alpha=0.7
    )
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.title("Clustering de inmuebles por superficies (recorte visual 1%-99%)")
    plt.xlabel("Superficie construida (m²)")
    plt.ylabel("Superficie parcela (m²)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_superficies.png")
    plt.close()

print("Gráficos generados con recorte visual por percentiles.")
