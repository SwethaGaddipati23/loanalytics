import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sqlite3

app = Flask(__name__)

# --- Load the trained model, columns, and scaler ---
try:
    with open('loan_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('model_columns.pkl', 'rb') as model_columns_file:
        model_columns = pickle.load(model_columns_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model, columns, and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Please run model_training.py first.")
    model = scaler = model_columns = None

# --- Helper Function for Database Connection ---
def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- Helper function to preprocess input data ---
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Convert columns to numeric
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Feature Engineering (must match training script) ---
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    epsilon = 1e-6
    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + epsilon)
    df['Installment'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + epsilon)

    # Apply log transformation
    df['LoanAmount'] = np.log1p(df['LoanAmount'])
    df['Total_Income'] = np.log1p(df['Total_Income'])

    # Drop original income columns
    df = df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    # Map categorical features
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    df['Education'] = df['Education'].map({'Graduate': 0, 'Not Graduate': 1})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semi-urban': 1, 'Rural': 0})
    
    # Ensure all columns are numeric and in the correct order
    for col in model_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[model_columns]


# --- Database Setup ---
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS loan_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Gender TEXT, Married TEXT, Dependents TEXT, Education TEXT,
            Self_Employed TEXT, ApplicantIncome INTEGER, CoapplicantIncome INTEGER,
            LoanAmount REAL, Loan_Amount_Term INTEGER, Credit_History REAL,
            Property_Area TEXT, Loan_Status TEXT, Prediction_Probability REAL,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Main Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# --- Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or model_columns is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        original_data = data.copy() 
        
        query_df = preprocess_input(data)
        query_scaled = scaler.transform(query_df)

        prediction = model.predict(query_scaled)
        probability = model.predict_proba(query_scaled)
        
        output = 'Y' if prediction[0] == 1 else 'N'
        confidence = probability[0][prediction[0]]
        
        coefficients = model.coef_[0]
        feature_impacts = pd.Series(coefficients * query_scaled[0], index=model_columns)
        
        top_feature_name = feature_impacts.abs().nlargest(1).index[0]
        top_feature_impact = feature_impacts[top_feature_name]
        feature_name_clean = top_feature_name.replace('_', ' ')

        description = f"The model's decision was primarily influenced by the applicant's {feature_name_clean}. "

        if top_feature_impact > 0:
             description += f"This factor had a strong positive impact, increasing the chances of approval."
        else:
             description += f"This factor had a strong negative impact, increasing the chances of rejection, though it may have been outweighed by other positive factors."

        
        top_factors = feature_impacts.abs().nlargest(3)
        feature_importance_data = [{'feature': feature, 'impact': feature_impacts[feature]} for feature in top_factors.index]

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO loan_records (
                Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, 
                LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status, Prediction_Probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', list(original_data.values()) + [output, confidence])
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': output,
            'confidence': f'{confidence:.2f}',
            'description': description,
            'feature_importance': feature_importance_data
        })

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 400

# --- Dashboard Data API Endpoint ---
@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    conn = get_db_connection()
    records_df = pd.read_sql_query("SELECT * FROM loan_records", conn)
    conn.close()
    
    status_counts = records_df['Loan_Status'].value_counts()
    pie_data = {
        'labels': ['Approved', 'Rejected'],
        'values': [int(status_counts.get('Y', 0)), int(status_counts.get('N', 0))]
    }

    area_approval = records_df[records_df['Loan_Status'] == 'Y']['Property_Area'].value_counts()
    bar_data = {
        'labels': area_approval.index.tolist(),
        'values': [int(v) for v in area_approval.values]
    }

    hist_data = {
        'incomes': [int(v) for v in records_df['ApplicantIncome'].dropna()]
    }
    
    scatter_df = records_df[['ApplicantIncome', 'LoanAmount']].dropna()
    scatter_data = [
        {'ApplicantIncome': int(row.ApplicantIncome), 'LoanAmount': float(row.LoanAmount)}
        for row in scatter_df.itertuples()
    ]

    numeric_df = records_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].copy().dropna()
    
    if len(numeric_df) < 2 or not all(numeric_df.nunique() > 1):
        heatmap_data = {'labels': [], 'data': []}
    else:
        corr_matrix = numeric_df.corr()
        corr_matrix = corr_matrix.fillna(0)
        heatmap_data = {
            'labels': corr_matrix.columns.tolist(),
            'data': [[float(v) for v in row] for row in corr_matrix.values]
        }

    records_df['Timestamp'] = pd.to_datetime(records_df['Timestamp'])
    records_df['Date'] = records_df['Timestamp'].dt.date
    line_data = records_df.groupby('Date').size()
    linegraph_data = {
        'labels': [str(d) for d in line_data.index],
        'values': [int(v) for v in line_data.values]
    }

    return jsonify({
        'pie_chart': pie_data,
        'bar_chart': bar_data,
        'histogram_data': hist_data,
        'scatter_data': scatter_data,
        'heatmap_data': heatmap_data,
        'line_graph': linegraph_data
    })


if __name__ == '__main__':
    app.run(debug=True)

