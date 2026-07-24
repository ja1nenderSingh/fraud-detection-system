from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/isolation_forest.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    amount   = float(data['amount'])
    time_val = float(data['time'])

    features = np.zeros(30)
    features[0] = (time_val - 86106) / 49978
    features[1] = (amount - 50) / 100

    np.random.seed(int(amount) % 9999)
    if amount > 800:
        features[2:] = np.random.normal(2, 3, 28)
    else:
        features[2:] = np.random.normal(0, 1, 28)

    prediction = model.predict([features])
    score      = model.decision_function([features])[0]
    is_fraud   = bool(prediction[0] == -1)
    risk       = float(abs(score) * 100)

    return jsonify({
        'is_fraud': is_fraud,
        'risk':     risk,
        'result':   'Fraud' if is_fraud else 'Normal'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8501))
    app.run(debug=False, host='0.0.0.0', port=port)