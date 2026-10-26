from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

#load the iris dataset
iris = load_iris()
x=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#Initialize the ransom forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
#Train the model
clf.fit(X_train, y_train)
#Make predictions
y_pred = clf.predict(X_test)
#Calculate the accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of the model is: {accuracy*100:.2f}%")
#Save the trained model
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "iris_random_forest_model.joblib"
joblib.dump(clf, model_path)
print(f"Trained model saved at: {model_path}")