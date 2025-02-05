import streamlit as st
import pandas as pd
import numpy as np
pip install matplotlib
import os

st.title("Visualizador de Datos del Notebook")

# Listar archivos en el directorio de entrada
st.header("Archivos disponibles")
input_dir = "./input"  # Ajusta la ruta según sea necesario
if os.path.exists(input_dir):
    files = os.listdir(input_dir)
    if files:
        st.write(files)
    else:
        st.write("No hay archivos en el directorio de entrada.")
else:
    st.write("El directorio de entrada no existe.")

# Cargar y mostrar un archivo CSV si está disponible
st.header("Carga y vista previa de datos")
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    # Mostrar estadísticas básicas
    st.subheader("Estadísticas básicas")
    st.write(df.describe())
    
    # Generar gráfico si hay columnas numéricas
    st.subheader("Gráficos de datos")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        selected_column = st.selectbox("Selecciona una columna para graficar", numeric_columns)
        fig, ax = plt.subplots()
        df[selected_column].hist(ax=ax, bins=20)
        st.pyplot(fig)
    else:
        st.write("No hay columnas numéricas para graficar.")

    # Análisis adicional basado en el código del notebook
    st.subheader("Análisis Adicional")
    st.write("Cantidad de valores nulos en el dataset:")
    st.write(df.isnull().sum())
    
    # Generar una matriz de correlación si hay más de una columna numérica
    if len(numeric_columns) > 1:
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots()
        cax = ax.matshow(df[numeric_columns].corr(), cmap='coolwarm')
        plt.colorbar(cax)
        st.pyplot(fig)
    
    # Gráfico de dispersión entre dos columnas numéricas
    if len(numeric_columns) > 1:
        st.subheader("Gráfico de dispersión")
        col1, col2 = st.selectbox("Selecciona la primera columna", numeric_columns), st.selectbox("Selecciona la segunda columna", numeric_columns)
        if col1 and col2 and col1 != col2:
            fig, ax = plt.subplots()
            ax.scatter(df[col1], df[col2], alpha=0.5)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            st.pyplot(fig)
        else:
            st.write("Selecciona dos columnas diferentes para el gráfico de dispersión.")
