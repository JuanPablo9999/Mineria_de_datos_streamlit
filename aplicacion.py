import streamlit as st
from PTL import Image

def main(): 
  st.title("Clasificación de la base de datos mnist")
  st.markdown("Sube una imagen para clasificar")

 uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG:)", type = ["jpg", "png", "jpeg"])

if uploaded file is not None:
  image = Image.open(uploaded_file)
  st.image(image.caption = "imagen subida")


if __name__=="__main__":
  main()
