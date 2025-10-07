# loanalytics
**Loan Prediction System with ML and Flask**


This project is a complete, end-to-end web application designed to predict loan eligibility based on applicant details. It serves as a practical demonstration of deploying a machine learning model in a real-world context. The system analyzes an applicant's financial and demographic information to provide an instant, data-driven decision. It also features a detailed analysis of each prediction and an interactive dashboard for visualizing historical trends, making it a valuable tool for both educational purposes and as a prototype for financial decision-making systems.

**Core Features**


ML-Powered Predictions: At its core, the application utilizes a LogisticRegression model, a powerful and highly interpretable algorithm. This model was chosen specifically for its ability to provide clear insights while maintaining high predictive accuracy, making it ideal for financial applications where explainability is crucial.

Detailed & Interpretable Analysis: Going beyond a simple "approved" or "rejected" status, the system provides a detailed breakdown for each prediction. It displays a confidence score and visually identifies the key positive (approving) and negative (rejecting) factors that most heavily influenced the model's decision, offering transparent and understandable results.

Interactive Dashboard: A comprehensive dashboard provides a high-level overview of all historical prediction data stored in the database. It features a variety of interactive charts—including pie charts for loan status distribution, bar charts for approval rates by property area, and scatter plots for income versus loan amount—allowing users to uncover trends and gain deeper insights from the data.

Web Interface: The application is built with a clean, modern, and user-friendly web interface. Developed using Flask and styled with Tailwind CSS, the UI is fully responsive, ensuring a seamless experience across desktops, tablets, and mobile devices.

Database Integration: Every prediction made through the application is automatically saved to a lightweight SQLite database. This persistence allows for powerful historical analysis and populates the dashboard, turning the tool into a dynamic system that learns and displays insights over time.  


**Technologies Used**


Backend: Python and the Flask micro-framework handle the server-side logic, API endpoints, and database communication.

Machine Learning: The core predictive power comes from Scikit-learn, with Pandas and NumPy used for efficient data manipulation and numerical operations.

Frontend: The user interface is built with standard HTML and styled using Tailwind CSS. Chart.js is used to render the interactive and visually appealing data visualizations on the dashboard.

Database: SQLite is used for its simplicity and file-based nature, providing a lightweight yet powerful solution for storing all prediction records.



**How to Run This Project**


Clone the repository:
Download the project files to your local machine using git clone.

git clone <your-repo-url>
cd loan_prediction_project

Create and activate a virtual environment:
This creates an isolated environment for the project's dependencies, preventing conflicts with other Python projects.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required libraries:
This command reads the requirements.txt file (if you have one) or installs the necessary packages for the project to run.

pip install Flask pandas scikit-learn numpy

Train the machine learning model:
Before running the app, you must first train the model. This script processes the loan_data.csv file and creates the loan_model.pkl file that the application uses for predictions.

(Make sure you have the loan_data.csv file from Kaggle in the root directory).

python model_training.py

Run the Flask application:
This command starts the local development server.

python app.py 
