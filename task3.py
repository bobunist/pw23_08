import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("insurance.csv")

features = data[["age", "bmi", "smoker"]]

target = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train["smoker"] = X_train["smoker"].apply(lambda x: 1 if x == "yes" else 0)
X_test["smoker"] = X_test["smoker"].apply(lambda x: 1 if x == "yes" else 0)


model_age = LinearRegression()
model_bmi = LinearRegression()
model_smoker = LinearRegression()

model_age.fit(X_train[["age"]], y_train)
model_bmi.fit(X_train[["bmi"]], y_train)
model_smoker.fit(X_train[["smoker"]], y_train)

y_pred_age = model_age.predict(X_test[["age"]])
y_pred_bmi = model_bmi.predict(X_test[["bmi"]])
y_pred_smoker = model_smoker.predict(X_test[["smoker"]])

r2_age_train = r2_score(y_train, model_age.predict(X_train[["age"]]))
r2_bmi_train = r2_score(y_train, model_bmi.predict(X_train[["bmi"]]))
r2_smoker_train = r2_score(y_train, model_smoker.predict(X_train[["smoker"]]))

r2_age_test = r2_score(y_test, y_pred_age)
r2_bmi_test = r2_score(y_test, y_pred_bmi)
r2_smoker_test = r2_score(y_test, y_pred_smoker)

results = pd.DataFrame({
    "Model": ["age", "bmi", "smoker"],
    "R2 (Training)": [r2_age_train, r2_bmi_train, r2_smoker_train],
    "R2 (Testing)": [r2_age_test, r2_bmi_test, r2_smoker_test]
})

print(results)

best_r2_test = max(r2_age_test, r2_bmi_test, r2_smoker_test)
best_model = "age" if best_r2_test == r2_age_test else ("bmi" if best_r2_test == r2_bmi_test else "smoker")
print(f"Лучший R2 на тестовом наборе для {best_model}")