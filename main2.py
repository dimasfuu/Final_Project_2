import streamlit as st
import pandas as pd
import joblib
import numpy as np



# Dictionary untuk pemilihan model
muat_model_lr20 = joblib.load("model_rasio_testing_20linearregression.joblib")
muat_model_rr20 = joblib.load("model_rasio_testing_20ridgeregression.joblib")
muat_model_ls20 = joblib.load("model_rasio_testing_20lassoregression.joblib")
muat_model_dt20 = joblib.load("model_rasio_testing_20decisiontreeregression.joblib")
muat_model_gb20 = joblib.load("model_rasio_testing_20gradientboostingregressor.joblib")
muat_model_xg20 = joblib.load("model_rasio_testing_20extremegradientboostingregressor.joblib")

# model = [muat_model_lr,muat_model_rr,muat_model_ls,muat_model_dt,muat_model_dt,muat_model_gb,muat_model_xg]
# Memilih model berdasarkan pilihan pengguna

df = pd.read_csv("df_deploy.csv")

st.header("Final Project 2 Delta Indie Course !")
st.subheader("Analisis data tentang Kepatuhan perawatan pasien")

st.title("Dikumpulkan sebagai Protofolio Final Project 2 untuk Kursus Delta Indie Course")
st.write("Nama :  Dimas Furqon Prawimastoro")


st.dataframe(df)

# streamlit UI
st.title("Persentase Kepatuhan Perawatan Pasien")
st.write("Masukkan data pasien untuk mengetahui level kepatuhan pasien.")

#pilih model
model_option = st.selectbox("Pilih Model",("Model Linear Regression 20:80","Model Ridge Regression 20:80","Model Lasso Regression 20:80","Model Decision Tree Regression 20:80","Model Gradient Boosting 20:80","Model Extreme Gradient Boosting 20:80")) # pilihan model

import streamlit as st

# # Generate data dummy

# X = df.drop(['Target'],axis=1)  # Fitur (independen)
# y = df['Target']  # Target (dependen)

# # Membagi data menjadi training dan testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Membuat dan melatih model
# model = model
# model.fit(X_train, y_train)

# # Evaluasi model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)

# st.write(f"Mean Squared Error: {mse:.2f}")

# Input dari pengguna
Age = st.number_input("Usia", min_value=0, step=1) 
physical = st.slider("Masukan skor kondisi fisik (1-10) : ",min_value=1, max_value= 10, step=1) 
sleep = st.slider("Masukan skor kualitas tidur (1-10) : ",min_value=1, max_value= 10, step=1) 
duration = st.number_input("Masukan durasi perawatan dalam sepekan (dalam jam):", min_value=0, step=1) 
stress = st.slider("Masukan skor tingkat stress (1-10) : ",min_value=1, max_value= 10, step=1)  
# Target = st.number_input("Tingkat kepatuhan pasien adalah:", min_value=0, step=1) 

# if st.button("Prediksi"):  # Jika tombol ditekan
#     model = muat_model_xg

if st.button("Prediksi"):  # Jika tombol ditekan
    if model_option == "Model Linear Regression 20:80":
        model = muat_model_lr20
    elif model_option == "Model Ridge Regression 20:80":
        model = muat_model_rr20
    elif model_option == "Model Lasso Regression 20:80":
        model = muat_model_ls20
    elif model_option == "Model Decision Tree Regression 20:80":
        model = muat_model_dt20
    elif model_option == "Model Decision Tree Regression 20:80":
        model = muat_model_gb20
    else:
        model = muat_model_xg20
    # Prediksi berdasarkan input pengguna

    input_data = np.array([[Age, physical, sleep,duration, stress]])#membentuk array input

    y_output = model.predict(input_data)
    # if st.write(f"Prediksi nilai kepatuhan perawatan pasien dari model {model_option} adalah: {y_output[0]:.2f} %"):
    if y_output[0] <= 30 :
        st.error(f"Prediksi nilai kepatuhan perawatan pasien dari model {model_option} adalah: {y_output[0]:.2f} % ,termasuk pada Kepatuhan kurang.")
    elif y_output[0] >=80:
        st.success(f"Prediksi nilai kepatuhan perawatan pasien dari model {model_option} adalah: {y_output[0]:.2f} % , termasuk pada Kepatuhan baik")
    else : 
        st.warning(f"Prediksi nilai kepatuhan perawatan pasien dari model {model_option} adalah: {y_output[0]:.2f} %, termasuk pada kepatuhan sedang")

    #     prediction = model.predict(input_data) #melakukan prediksi dengan model

    #     st.write(prediction)

