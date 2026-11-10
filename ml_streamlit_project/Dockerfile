FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Se re-entrena dentro de la imagen para que model.joblib sea
# compatible con las versiones de scikit-learn instaladas aqui.
RUN python train.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
