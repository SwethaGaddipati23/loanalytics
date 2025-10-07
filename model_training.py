import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- 1. Load Data ---
df = pd.read_csv('loan_data.csv')

# --- 2. Preprocessing ---
df = df.drop('Loan_ID', axis=1)

# Handle categorical variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
df['Education'] = df['Education'].map({'Graduate': 0, 'Not Graduate': 1})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Fill missing values with the median
df.fillna(df.median(), inplace=True)

# --- IMPROVEMENT: Advanced Feature Engineering ---
# 1. Create Total_Income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# 2. Create Loan_to_Income_Ratio
epsilon = 1e-6
df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + epsilon)

# 3. Create Installment amount - this is a much better feature
df['Installment'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + epsilon)

# Apply log transformation to skewed numerical features
df['LoanAmount'] = np.log1p(df['LoanAmount'])
df['Total_Income'] = np.log1p(df['Total_Income'])

# Drop redundant columns
df = df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

# Define features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

model_columns = X.columns

# --- 3. Scale the Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
y_pred = model.predict(X_test)
print(f"Model Accuracy with new features: {accuracy_score(y_test, y_pred):.2f}")

# --- 6. Save Artifacts ---
with open('loan_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('model_columns.pkl', 'wb') as columns_file:
    pickle.dump(model_columns, columns_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\nModel, columns, and scaler have been successfully retrained and saved with new features!")

