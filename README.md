# Segmentación de clientes mayoristas (K-Means + Streamlit + Docker + GCP)

## Objetivo
Agrupar clientes de un distribuidor mayorista según su gasto anual por categoría de producto y ofrecer una app web que, dado un nuevo cliente, le asigne un perfil de compra.

## Dataset
- **Wholesale customers data** (UCI Machine Learning Repository).
- Cada fila = **un cliente mayorista**; 440 filas.
- Columnas: `Channel` (1=Horeca, 2=Retail), `Region` (1=Lisboa, 2=Oporto, 3=Otra), y gasto anual en `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`.
- Sin datos faltantes. Archivo en `data/datos.csv`.
- Fuente: https://archive.ics.uci.edu/dataset/292/wholesale+customers

## Ruta elegida
**No supervisada** (clustering con K-Means).

## Variables utilizadas
Solo las 6 variables de gasto. `Channel` y `Region` se excluyen del entrenamiento y se usan únicamente para interpretar los clusters.

## Preparación de datos (dentro del Pipeline)
1. `SimpleImputer(median)`
2. `log1p`: el gasto es muy asimétrico, con valores extremos
3. `StandardScaler`
4. `KMeans(n_clusters=3, random_state=42)`

## Justificación del modelo y evaluación
K-Means permite asignar nuevos casos con `predict()`. Se probó k = 2..8:

| k | Inercia | Silhouette |
|---|---------|-----------|
| 2 | 1844.1 | 0.290 |
| 3 | 1553.8 | 0.261 |
| 4 | 1387.7 | 0.185 |
| 5 | 1284.8 | 0.192 |

Se eligió **k = 3** por ser un buen compromiso entre silhouette (0.261) y perfiles interpretables. k = 2 tiene el mayor silhouette pero separa casi solo por tamaño de gasto. Un silhouette cercano a 0.26 indica estructura moderada, con clusters que se solapan.

## Interpretación de la salida
| Perfil | Tamaño | Descripción |
|--------|--------|-------------|
| Frescos y congelados | 216 | Gasto en frescos y congelados cercano al promedio; bajo en lácteos, abarrotes y limpieza. Casi todos Horeca (213 de 216). |
| Abarrotes y limpieza | 81 | Gasto alto en abarrotes y detergentes/papel; muy bajo en frescos y congelados. Mayoría Retail. |
| Grandes compradores | 143 | Gasto alto en todas las categorías. |

Las descripciones se asignan según el perfil de gasto, no según el número de cluster, así que son estables entre re-entrenamientos.

## Ejecutar entrenamiento
```bash
pip install -r requirements.txt
python train.py
```
Genera `model.joblib`.

## Ejecutar Streamlit
```bash
streamlit run app.py
```
Abrir http://localhost:8501

## Docker
Construir la imagen (el entrenamiento se ejecuta durante el build):
```bash
docker build -t ml-streamlit-app .
```
Ejecutar el contenedor:
```bash
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
docker ps
docker logs ml-streamlit
```

## Despliegue en una VM de Google Cloud (Ubuntu 24.04)
```bash
sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable --now docker
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
sudo docker build -t ml-streamlit-app .
sudo docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```
Abrir el puerto 8501: poner a la VM la etiqueta de red `streamlit-server` y crear la regla:
```bash
gcloud compute firewall-rules create allow-streamlit-8501 \
  --allow=tcp:8501 --direction=INGRESS --target-tags=streamlit-server
```
Abrir `http://IP_EXTERNA_DE_LA_VM:8501`.

Para actualizar tras cambios: `git pull`, `docker stop ml-streamlit`, `docker rm ml-streamlit`, volver a hacer `docker build` y `docker run`.

## Ejemplo de entrada y salida
Entrada: Fresh=12000, Milk=3000, Grocery=4000, Frozen=2000, Detergents_Paper=500, Delicassen=1000

Salida: `Cluster asignado: 1` → *Frescos y congelados: gasto medio en frescos y congelados, bajo en lácteos, abarrotes y limpieza.*

(El número del cluster puede variar entre entrenamientos; la descripción es la referencia.)

## Autores
- _Jorge_Zainea_
- _Daniel_Gamba_
