import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

#Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"]=iris.target
df["species"]=df["target"].apply(lambda x: iris.target_names[x])


# Visualize the data
sns.pairplot(df, hue="species")
plt.suptitle("Iris Flower Relationships", y=1.02)
plt.show()

#Features and labels
X = df.iloc[:, :4]
y = df["target"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

#Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()


with open("iris_model.pkl", "wb") as f:
    pkl.dump(model, f)

print("Model saved as iris_model.pkl")