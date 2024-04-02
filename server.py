from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# Load your model and label encoder
label_encoder = joblib.load('../model/label_encoder.pkl')
model = joblib.load('../model/gradient_boosting.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from form
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Perform label encoding
        sex_encoded = label_encoder.fit_transform([sex])[0]
        smoker_encoded = label_encoder.fit_transform([smoker])[0]
        region_encoded = label_encoder.fit_transform([region])[0]
        
        # Create feature vector
        features = [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
