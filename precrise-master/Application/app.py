from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('../model8Age.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les valeurs du formulaire
    features = ['Sex', 'Age', 'SleepHours', 'DifficultyErrands', 'SmokerStatus', 'PhysicalActivities', 'AlcoholDrinkers', 'CovidPos']
    input_features = []

    for feature in features:
        value = request.form.get(feature)
        input_features.append(float(value if value else 0))

    # Effectuer la prédiction
    prediction = model.predict_proba([input_features])[0]
    
    # Normaliser la probabilité prédite entre 0 et 100
    normalized_prediction = prediction[0] * 100
    result = 100 - ((normalized_prediction - 50) * 2)
    # Préparer le texte de prédiction
    prediction_text = f'Probabilité de crise cardiaque : {round(result, 2)}%'
    
    return render_template('index.html', prediction_text=prediction_text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
