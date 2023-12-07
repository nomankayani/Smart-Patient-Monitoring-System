from flask import Flask, request, jsonify
import pandas as pd
from joblib import load 
from flask_cors import CORS   # Assuming you used joblib to save your models

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Use double backslashes
# Use relative paths
logistic_model_path = 'models/logistic_regression_model.joblib'
rf_model_path = 'models/random_forest_model.joblib'

# Load the Logistic Regression model
logistic_model = load(logistic_model_path)
rf_model =load(rf_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Prepare the data for prediction (assuming it's a JSON object)
        input_data = pd.DataFrame(data, index=[0])

        # Make predictions
        logistic_prediction = logistic_model.predict(input_data)
        rf_prediction = rf_model.predict(input_data)

        # Return the predictions as JSON
        result = {
            'logistic_prediction': int(logistic_prediction[0]),
            'rf_prediction': int(rf_prediction[0])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
