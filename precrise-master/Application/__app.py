from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open('model.pkl', 'rb'))

# Lire le DataFrame
df = pd.read_csv('/Users/hamzadriss/Downloads/PreCrise/heart_2022_with_nans.csv')

# Extraire les âges
df['Age'] = df['AgeCategory'].str.extract('(\d+)').astype(float).fillna(df['AgeCategory'].str.extract('(\d+)').astype(float).mean()).astype(int)
unique_ages = sorted(df['Age'].unique())

# Mapper les valeurs de SmokerStatus
smoker_status_mapping = {
    'Never smoked': 3,
    'Current smoker - now smokes some days': 1,
    'Former smoker': 2,
    'Current smoker - now smokes every day': 0
}

df['SmokerStatus'] = df['SmokerStatus'].map(smoker_status_mapping)

# Extraire les valeurs uniques de SmokerStatus
unique_smoker_status = sorted(smoker_status_mapping.items(), key=lambda item: item[1])

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
    result = (normalized_prediction - 50) * 2

    # Préparer le texte de prédiction
    prediction_text = f'Probabilité de crise cardiaque : {round(result, 2)}%'
    
    return render_template('index.html', prediction_text=prediction_text, ages=unique_ages, smoker_status=unique_smoker_status)

if __name__ == '__main__':
    app.run(debug=True)
