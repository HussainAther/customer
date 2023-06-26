from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.graph_objects as go

app = Flask(__name__)

# Load the trained model
model = joblib.load("your_model.pkl")

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input values from the form
        customer_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            # Add other input fields here...
        }

        # Preprocess the input data
        df = pd.DataFrame([customer_data])
        # Apply the same preprocessing steps used during training (e.g., one-hot encoding)

        # Make predictions using the trained model
        prediction = model.predict(df)

        # Visualize the prediction
        fig = go.Figure(data=[go.Pie(labels=['Not Churned', 'Churned'], values=prediction)])
        graph = fig.to_html(full_html=False)

        return render_template('index.html', prediction=graph)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

