import streamlit as st
import numpy as np
import pickle  # Asegúrate de tener tu modelo guardado en un archivo pickle

# Cargar el modelo entrenado
MODEL_PATH = "model_trained_regressor.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

# Definir las 13 características del conjunto de datos Boston Housing
features = [
    "CRIM - Tasa de criminalidad por zona",
    "ZN - Proporción de terrenos residenciales grandes",
    "INDUS - Proporción de acres comerciales por ciudad",
    "CHAS - Variable ficticia del río Charles (1 si está cerca, 0 si no)",
    "NOX - Concentración de óxidos de nitrógeno",
    "RM - Número medio de habitaciones por vivienda",
    "AGE - Proporción de unidades construidas antes de 1940",
    "DIS - Distancia ponderada a cinco centros de empleo de Boston",
    "RAD - Índice de accesibilidad a carreteras radiales",
    "TAX - Tasa de impuesto a la propiedad",
    "PTRATIO - Relación alumno-profesor por ciudad",
    "B - Proporción de personas de raza negra por ciudad",
    "LSTAT - Porcentaje de población con estatus socioeconómico bajo",
]

st.title("Predicción del Precio de una Casa en Boston")
st.write("Ingrese los valores para cada característica:")

# Crear entradas para las 13 características
input_data = []
for feature in features:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

# Botón para predecir
if st.button("Predecir Precio"):
    # Convertir la entrada en un array numpy y hacer la predicción
    input_array = np.array(input_data).reshape(1, -1)
    predicted_price = model.predict(input_array)[0]
    
    st.success(f"El precio estimado de la casa es: ${predicted_price * 1000:,.2f}")

# Instrucciones para despliegue en GitHub Pages o Streamlit Cloud
st.write("### Instrucciones para ejecución en GitHub")
st.markdown("1. Sube este archivo a un repositorio de GitHub.\n" 
            "2. Asegúrate de incluir el archivo `model_trained_regressor.pkl`.\n"
            "3. Despliega en [Streamlit Community Cloud](https://streamlit.io/cloud) enlazando tu repo.")
