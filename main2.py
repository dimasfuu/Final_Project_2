import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Dictionary untuk pemilihan model
muat_model_xg = joblib.load("model_rasio_testing_20extremegradientboostingregressor.joblib")


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
model_option = st.selectbox("Pilih Model",("model_linear_regression.joblib","model_ridge_regression.joblib","model_lasso_regression.joblib","model_decision_tree.joblib","model_gradient_boosting.joblib","model_rasio_testing_20extremegradientboostingregressor.joblib")) # pilihan model

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

if st.button("Prediksi"):  # Jika tombol ditekan
    model = muat_model_xg

# if st.button("Prediksi"):  # Jika tombol ditekan
#     if model_option == "model_linear_regression.joblib":
#         model = muat_model_lr
#     elif model_option == "model_ridge_regression.joblib":
#         model = muat_model_rr
#     elif model_option == "model_lasso_regression.joblib":
#         model = muat_model_ls
#     elif model_option == "model_decision_tree.joblib":
#         model = muat_model_dt
#     elif model_option == "model_gradient_boosting.joblib":
#         model = muat_model_gb
#     else:
#         model = muat_model_xg

 # Prediksi berdasarkan input pengguna
input_data = np.array([[Age, physical, sleep,duration, stress]])#membentuk array input
y_output = model.predict(input_data)

st.sidebar.write(f"Prediksi nilai Y: {y_output[0]:.2f}")


#     prediction = model.predict(input_data) #melakukan prediksi dengan model

#     st.write(prediction)

#     if prediction[0] <= 30 :
#         st.error("Kepatuhan kurang.")
#     elif prediction[0] >=80:
#         st.success("Kepatuhan baik")
#     else : 
#         st.warning("Kepatuhan sedang")