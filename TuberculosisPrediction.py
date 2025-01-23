import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import os

# Step 1: Load the data
data_file = "Tuberculosis_Dataset.csv"
if not os.path.exists(data_file):
    print(f"The file '{data_file}' does not exist. Please check the file path and try again.")
    exit()

data = pd.read_csv(data_file)

# Step 2: Process the data
required_columns = [
    "Country", "Year", "TB_Incidence_Rate", "Population", "HIV_Prevalence", "Treatment_Success_Rate", "Mortality_Rate", "Urban_Population_Percentage", "Smoking_Rate", "Alcohol_Consumption", "risk"
]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"The following required columns are missing in the dataset: {missing_columns}")
    exit()


label_encoders = {}
categorical_columns = ["Country", "Year", "TB_Incidence_Rate", "Population", "HIV_Prevalence", "Treatment_Success_Rate", "Mortality_Rate", "Urban_Population_Percentage", "Smoking_Rate", "Alcohol_Consumption", "risk"]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


X = data.drop("risk", axis=1)
y = data["risk"]


le_target = LabelEncoder()
y = le_target.fit_transform(y)


numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))



