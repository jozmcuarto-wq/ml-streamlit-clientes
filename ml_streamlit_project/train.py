from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "datos.csv"
MODEL_PATH = BASE / "model.joblib"

N_CLUSTERS = 3
NUMERIC = ["Fresh", "Milk", "Grocery", "Frozen",
           "Detergents_Paper", "Delicassen"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def name_clusters(df: pd.DataFrame, labels) -> dict:
    """Asigna descripciones segun el perfil de gasto, no segun el numero."""
    tmp = df[NUMERIC].copy()
    tmp["cluster"] = labels
    ratio = tmp.groupby("cluster")[NUMERIC].mean() / df[NUMERIC].mean()

    desc = {}
    big = ratio.mean(axis=1).idxmax()
    desc[int(big)] = "Grandes compradores: gasto alto en todas las categorias."
    rest = ratio.drop(index=big)
    grocery = (rest["Grocery"] + rest["Detergents_Paper"]).idxmax()
    desc[int(grocery)] = ("Abarrotes y limpieza: gasto alto en abarrotes y "
                          "detergentes/papel, bajo en frescos y congelados.")
    for c in rest.index:
        if int(c) not in desc:
            desc[int(c)] = ("Frescos y congelados: gasto medio en frescos y "
                            "congelados, bajo en lacteos, abarrotes y limpieza.")
    return desc


def train_model(df: pd.DataFrame):
    X = df[NUMERIC].copy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=N_CLUSTERS, random_state=42,
                           n_init="auto")),
    ])
    labels = model.fit_predict(X)

    score = silhouette_score(model[:-1].transform(X), labels)
    print("Silhouette:", round(score, 3))
    print("Tamanos:", pd.Series(labels).value_counts().sort_index().to_dict())

    artifact = {
        "model": model,
        "features": NUMERIC,
        "numeric_features": NUMERIC,
        "categorical_features": [],
        "task": "clustering",
        "cluster_descriptions": name_clusters(df, labels),
        "defaults": X.median().to_dict(),
    }
    return artifact


def main() -> None:
    df = load_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    for k, v in artifact["cluster_descriptions"].items():
        print(k, "->", v)


if __name__ == "__main__":
    main()
