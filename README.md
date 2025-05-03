# 🌸 Iris Flower Classification App

A simple and interactive machine learning web app that predicts the species of Iris flower based on their features. It allows users to **upload their own CSV file** and get predicted downloadable result file.

---

## Features

- Upload a .csv file with Iris flower data
- Predicts the species using a trained ML model
- View predictions directly in the app
- Download the result as a `.csv` file
- Clean UI built with Streamlit

---

## Project Structure

```plaintext
iris-classification/
├── app.py               # Streamlit App
├── iris_model.py        # ML training logic
├── iris_model.pkl       # Trained ML model
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

---

## Sample CSV Format
Upload a CSV with the following column format (no headers required if handled in your code):
```
sepal length,sepal width,petal length,petal width
5.1,3.5,1.4,0.2
6.7,3.0,5.2,2.3
```

---

## How to run the app locally

1. **Clone this repository:**

```bash 
git clone https://github.com/sakshamnagar-7227/iris-flower-classification.git
cd iris-flower-classification
```
2. Install Dependencies

```bash 
pip install -r requirements.txt
```

3. Run the app:

streamlit run app.py

---

## Model Used
```
Logistic Regression
Trained on the classic Iris Dataset
Achievd 100% accuracy on test due to clear decision boundaries and low noise
```

---

## Requirements

See the requirements.txt file or install manually:
```bash 
pip install pandas matplotlib.pyplot seaborn scikit-learn streamlit pickle
```

## Connect
Feel free to open an issue or suggestion to collabrate or improve the project!
 