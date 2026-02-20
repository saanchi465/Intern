import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("IRIS.csv")

X = df.drop("species", axis=1)
y = df["species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nBest Model:")
best_model = max(
    [("Logistic Regression", accuracy_score(y_test, lr_pred)),
     ("SVM", accuracy_score(y_test, svm_pred)),
     ("Random Forest", accuracy_score(y_test, rf_pred))],
    key=lambda x: x[1]
)
print(best_model)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))