from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Only load model if it exists
    if not os.path.exists(model_path):
        return "Error: model.pkl not found. Run train_model.py first!"
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        text = request.form['message']
        prediction = model.predict([text])[0]
        result = "🔴 Bullying Detected" if prediction == 1 else "🟢 Safe Content"
        return render_template('index.html', prediction=result, original_text=text)

if __name__ == '__main__':
    app.run(debug=True)