import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu


# =========================================================
# CONFIGURACIÓN
# =========================================================
INPUT_CSV = "ev_clean.csv"
RANDOM_STATE = 42

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("APARTADO 4 - ANÁLISIS DE LOS DATOS")
print("=" * 80)

# =========================================================
# CARGA
# =========================================================
df = pd.read_csv(INPUT_CSV)

print("\nDimensiones dataset limpio:")
print(df.shape)

print("\nNulos por columna:")
print(df.isna().sum())


# =========================================================
# 4.1 MODELO SUPERVISADO: REGRESIÓN PARA ESTIMAR EL PRECIO
# =========================================================
print("\n" + "-" * 80)
print("4.1 MODELO SUPERVISADO (REGRESIÓN)")
print("-" * 80)

# Para supervisado necesitamos precio no nulo
df_sup = df.dropna(subset=["precio"]).copy()

print("\nDimensiones para supervisado (sin nulos en precio):")
print(df_sup.shape)

target = "precio"
features = [
    "superficie_construida_m2",
    "superficie_parcela_m2",
    "habitaciones",
    "banos",
    "zona",
    "localizacion",
]

X = df_sup[features]
y = df_sup[target]

num_features = ["superficie_construida_m2", "superficie_parcela_m2", "habitaciones", "banos"]
cat_features = ["zona", "localizacion"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append((name, mae, rmse, r2))

res_df = pd.DataFrame(results, columns=["modelo", "MAE", "RMSE", "R2"]).sort_values("RMSE")
print("\nRendimiento (test 20%):")
print(res_df.to_string(index=False))


# =========================================================
# 4.1 MODELO NO SUPERVISADO: CLUSTERING (KMEANS)
# =========================================================
print("\n" + "-" * 80)
print("4.1 MODELO NO SUPERVISADO (CLUSTERING)")
print("-" * 80)

# Para clustering usamos variables de características (sin precio)
# y solo numéricas para evitar alta dimensionalidad por one-hot.
cluster_features = ["superficie_construida_m2", "superficie_parcela_m2", "habitaciones", "banos"]
df_clu = df[cluster_features].copy()

# Imputación simple por mediana (aquí no es "no aplica", porque ya limpiamos a 0 en el apartado 3)
# pero dejamos robustez por si hubiera algún NaN residual.
df_clu = df_clu.fillna(df_clu.median(numeric_only=True))

scaler = StandardScaler()
Xc = scaler.fit_transform(df_clu)

# Elegimos k probando varios valores y reportando silhouette
k_values = list(range(2, 9))
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xc)
    score = silhouette_score(Xc, labels)
    sil_scores.append((k, score))

sil_df = pd.DataFrame(sil_scores, columns=["k", "silhouette"]).sort_values("silhouette", ascending=False)
print("\nSilhouette por k (mayor es mejor):")
print(sil_df.to_string(index=False))

best_k = int(sil_df.iloc[0]["k"])
print(f"\nValor k seleccionado: {best_k}")

km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = km_best.fit_predict(Xc)

print("\nTamaño de cada cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nResumen por cluster (medianas):")
print(df.groupby("cluster")[cluster_features].median(numeric_only=True))


# =========================================================
# 4.2 PRUEBA POR CONTRASTE DE HIPÓTESIS
# =========================================================
print("\n" + "-" * 80)
print("4.2 CONTRASTE DE HIPÓTESIS")
print("-" * 80)

# Ejemplo: comparar precio entre las dos zonas más frecuentes
# (se hace así para garantizar tamaños muestrales razonables)
zone_counts = df_sup["zona"].value_counts()
top2 = zone_counts.index[:2].tolist()

if len(top2) < 2:
    print("No hay suficientes zonas para hacer contraste entre dos grupos.")
else:
    z1, z2 = top2[0], top2[1]
    p1 = df_sup.loc[df_sup["zona"] == z1, "precio"].dropna()
    p2 = df_sup.loc[df_sup["zona"] == z2, "precio"].dropna()

    print(f"\nZonas comparadas: '{z1}' (n={len(p1)}) vs '{z2}' (n={len(p2)})")

    # Normalidad (Shapiro). Si n es grande, Shapiro puede ser muy sensible;
    # usamos submuestra si excede 5000 para evitar problemas.
    def shapiro_safe(x):
        x = x.sample(n=min(len(x), 5000), random_state=RANDOM_STATE)
        stat, p = shapiro(x)
        return stat, p

    sh1_stat, sh1_p = shapiro_safe(p1)
    sh2_stat, sh2_p = shapiro_safe(p2)

    print("\nNormalidad (Shapiro-Wilk):")
    print(f"{z1}: p-value = {sh1_p:.6f}")
    print(f"{z2}: p-value = {sh2_p:.6f}")

    # Homocedasticidad (Levene)
    lev_stat, lev_p = levene(p1, p2)
    print("\nHomocedasticidad (Levene):")
    print(f"p-value = {lev_p:.6f}")

    normal = (sh1_p > 0.05) and (sh2_p > 0.05)
    equal_var = (lev_p > 0.05)

    if normal:
        # t-test (Welch si varianzas desiguales)
        t_stat, t_p = ttest_ind(p1, p2, equal_var=equal_var)
        test_name = "t-test (Student/Welch)"
        p_value = t_p
    else:
        # Mann-Whitney (no paramétrica)
        u_stat, u_p = mannwhitneyu(p1, p2, alternative="two-sided")
        test_name = "Mann-Whitney U"
        p_value = u_p

    print("\nPrueba seleccionada:")
    print(test_name)
    print(f"p-value = {p_value:.6f}")

    # Diferencia de medianas / medias como apoyo interpretativo
    print("\nResumen comparativo de precios:")
    print(f"{z1}: media={p1.mean():.2f}, mediana={p1.median():.2f}")
    print(f"{z2}: media={p2.mean():.2f}, mediana={p2.median():.2f}")

