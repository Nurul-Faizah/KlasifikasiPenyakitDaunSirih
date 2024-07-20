import streamlit as st # type: ignore
from PIL import Image # type: ignore
import tensorflow as tf # type: ignore
# from tensorflow import keras # type: ignore
import keras
import numpy as np # type: ignore
import math
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout# type: ignore
from keras.utils import to_categorical # type: ignore
from keras.models import load_model# type: ignore
from io import BytesIO
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from efficientnet import EfficientNetWithTripletAttention
from efficientnet import get_efficient_net_triplet_attention
from efficientnet import build_mbconv_block_with_triplet_attention
from efficientnet import efficient_net_b7_with_triplet_attention
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tqdm import tqdm

st.set_page_config(
    page_title="Project"
)
page_bg_img = """
<style>
[data-testid="stApp"]{
    background-image: url("https://cdn.pixabay.com/photo/2017/03/14/08/10/wood-2142241_1280.jpg");
    background-size: cover;
}
[data-testid = "stHeader"]{
    background-color:rgba(0,0,0,0);
}
[data-testid = "stAToolbar"]{
    right: 2rem;
}
<style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)
st.title('Classification Of Betel Leaf')

tab1, tab2 = st.tabs(["Information", "Test Image"])  

with tab1:
    st.subheader("Pada klasifikasi ini terbagi menjadi 4 kelas yaitu :")
    st.write("""
    <ol>
        <li>Daun Sirih Hijau Sehat : Daun sirih berwarna hijau segar tanpa tanda-tanda penyakit.</li>
        <li>Daun Sirih Hijau_Antraknos : Daun sirih hijau dengan bercak coklat atau hitam yang menunjukkan adanya infeksi jamur antraknos.</li>
        <li>Daun Sirih Hijau_Karat : Daun sirih hijau dengan bercak oranye atau coklat yang menandakan adanya penyakit karat.</li>
        <li>Daun Sirih Merah_Sehat : Daun sirih berwarna merah segar tanpa tanda-tanda penyakit.</li>
    </ol>""",unsafe_allow_html=True)
with tab2:        
        # Menu pilihan

        uploaded_file = st.file_uploader("Select photo", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            model = load_model("fold_5_model_B7_Training_EfficientNet_TripletAttention_epoch_50.h5",  custom_objects={"EfficientNetWithTripletAttention": EfficientNetWithTripletAttention})
            LABELS = ["Hijau_Sehat", "Hijau_Antraknos", "Hijau_Karat", "Merah_Sehat"]
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Photo', use_column_width=True)
            # Mengubah gambar menjadi bentuk yang sesuai untuk prediksi
            input_image = image.resize((224, 224))
            input_image = np.array(input_image)
            input_image = np.expand_dims(input_image, axis=0)
            # st.write(input_image)

            # Melakukan prediksi menggunakan model atau tindakan lain
            prediction = model.predict(input_image, batch_size=16)
            class_index = np.argmax(prediction, axis=1)
            st.write(prediction)
            st.write(class_index)
            class_name = LABELS[class_index[0]]

            # Menampilkan hasil prediksi
            st.success(f"Hasil Prediksi: {class_name}")





                