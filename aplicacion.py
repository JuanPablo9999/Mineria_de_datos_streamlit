import streamlit as st
from PTL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image)
  image = image.convert("L") #Este comando lo convierte en escala de grises
  image = image.resize(28,28)
  image_array = imag.to_array(image)/255.0
  image_array = np.expand_dims(image_array, axis=0)
  return image_array

def main(): 
  st.title("Clasificación de la base de datos mnist")
  st.markdown("Sube una imagen para clasificar")

 uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG:)", type = ["jpg", "png", "jpeg"])

if uploaded file is not None:
  image = Image.open(uploaded_file)
  st.image(image,caption = "imagen subida")

  preprocessed_image = preprocess_image(image)
  st.image(preprocessed_image, caption = "imagen procesada")
  

if __name__=="__main__":
  main()
