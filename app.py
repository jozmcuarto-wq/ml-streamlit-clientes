from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛒")


@st.cache_resource
def load_artifact():
    return joblib.load(Path(__file__).parent / "model.joblib")


artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
numeric_features = artifact.get("numeric_features", [])
categorical_features = artifact.get("categorical_features", [])
category_options = artifact.get("category_options", {})
defaults = artifact.get("defaults", {})
task = artifact["task"]

LABELS = {
    "Fresh": "Productos frescos",
    "Milk": "Lácteos",
    "Grocery": "Abarrotes",
    "Frozen": "Congelados",
    "Detergents_Paper": "Detergentes y papel",
    "Delicassen": "Delicatessen",
}

st.title("Segmentación de clientes mayoristas")
st.write(
    "Ingrese el **gasto anual** del cliente en cada categoría "
    "(unidades monetarias) y el modelo le asignará un perfil de compra."
)

values = {}
for feature in numeric_features:
    values[feature] = st.number_input(
        LABELS.get(feature, feature),
        min_value=0.0,
        value=float(defaults.get(feature, 0.0)),
        step=100.0,
    )

for feature in categorical_features:
    if feature in category_options:
        values[feature] = st.selectbox(feature, category_options[feature])
    else:
        values[feature] = st.text_input(feature)

if st.button("Ejecutar modelo"):
    X_new = pd.DataFrame([values], columns=features)
    result = model.predict(X_new)[0]

    if task == "classification":
        st.success(f"Clase predicha: {result}")
        if hasattr(model, "predict_proba"):
            st.write("Probabilidades:", model.predict_proba(X_new)[0])
    elif task == "clustering":
        descriptions = artifact["cluster_descriptions"]
        st.success(f"Cluster asignado: {result}")
        st.info(descriptions.get(int(result), "Sin descripción"))
