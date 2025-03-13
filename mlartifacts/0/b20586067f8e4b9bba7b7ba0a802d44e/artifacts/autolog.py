import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# print("Setting MLflow tracking URI")
mlflow.set_tracking_uri('http://localhost:5000')

# Load data
print("Loading data")
data = load_wine()

# Split data
print("Splitting data")
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

max_depth = 15
n_estimators = 10

print("Starting MLflow autolog")
mlflow.autolog() # autologging
with mlflow.start_run():
    # print("Training model")
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, Y_train)

    # print("Predicting")
    Y_pred = rf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    # Log confusion matrix
    # print("Logging confusion matrix")
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    # plt.show()

    # print("Logging artifacts")
    # mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # log tags
    # print("Setting tags")
    mlflow.set_tag("author", "Yash Jain")
    mlflow.set_tag("model", "RandomForestClassifier")

    # log model
    # print("Logging model")
    # mlflow.sklearn.log_model(rf, "random_forest_model")

    print("Accuracy:", accuracy)

