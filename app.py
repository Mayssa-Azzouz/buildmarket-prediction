from flask import Flask, request, jsonify
import os
import pickle

app = Flask(__name__)

# Charger modèle (adapte si besoin)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/health')
def health():
    return {"status": "ok"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # sécuriser les valeurs null
    features = [
        data.get('meeting_attended', False),
        data.get('meeting_diff_minutes') or 0,
        data.get('interet_client') or 0,
        data.get('probabilite_ressentie') or 0,
        data.get('budget_confirme', False),
        data.get('decideur_present', False),
        data.get('nb_relances') or 0
    ]

    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0][1]

    return jsonify({
        "prediction": str(prediction),
        "score_percent": round(proba * 100, 2)
    })

# IMPORTANT pour gunicorn
if __name__ != "__main__":
    app = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)