import streamlit as st
import pandas as pd
import pickle

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Classification")
st.write("Upload a CSV file containing the flower measurements and get the predictions.")

#Upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    #Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df)

    try:
        predictions = model.predict(df)
        df["Predicted Species"] = predictions

        st.write("Predictions:")
        st.dataframe(df)

        #Downloadable CSV file with predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predicted CSV",
            data=csv,
            file_name="iris_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file to get the prdeictions")
