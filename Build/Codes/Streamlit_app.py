import streamlit as st
import os
import time
import tensorflow as tf
from Build.Codes.Transformer import *
from Build.Codes.Pdf_Extract import *

st.title("Chatbot")

choice = st.selectbox("User Type", ["Admin", "User"])

if choice == "Admin":
    st.title("Admin Section")
    password = st.text_input("Password", type="password")
    if(st.button('Submit')):
        if password == "12345":
            st.subheader("Upload PDF to Train Model")
            uploaded_files =  os.listdir("/workspaces/transformer/Dataset/PDF_Dataset" )
            choice = st.selectbox('Choose Context' , uploaded_files)
            print(choice)
            st.success("PDF files uploaded successfully!")
            with st.spinner("Training in progress..."):
                loaded_model = model_fit(f"Dataset/PDF_Dataset/{choice}")
                st.success("Model training completed!")
        else:
            st.error("Incorrect password")

elif choice == "User":
    st.title("User Section")
    options =  os.listdir("/workspaces/transformer/Dataset/PDF_Dataset" )
    choice = st.selectbox('Choose Context' , options)
    loaded_model = load_model(choice )
    user_input = st.text_input("Ask a question")
    
    if st.button("Submit"):
        response = predict(user_input , loaded_model)  # Replace with your chatbot interaction function
        st.write("Chatbot response:", response)
    else:
        st.warning("Model weights not found. Please train the model first.")