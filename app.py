from flask import Flask, request, jsonify
import os
import pickle

app = Flask(__name__)

# 🔥 Chargement du modèle
model = pickle.load(open("model.pkl", "rb"))

# (optionnel si tu as encoders.pkl)
try:
    encoders = pickle.load(open("encoders.pkl", "rb"))
except:
    encoders = None

# ✅ Health check
@app.route('/health')
def health():
    return {"status": "ok"}

# ✅ Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 DATA RECEIVED:", data)

        # 🔒 Sécurisation + conversion
        meeting_attended = int(data.get('meeting_attended', False))
        meeting_diff_minutes = int(data.get('meeting_diff_minutes') or 0)
        interet_client = int(data.get('interet_client') or 0)
        probabilite_ressentie = int(data.get('probabilite_ressentie') or 0)
        budget_confirme = int(data.get('budget_confirme', False))
        decideur_present = int(data.get('decideur_present', False))
        nb_relances = int(data.get('nb_relances') or 0)

        # 🔥 Gestion prochaine_etape (si encoder existe)
        if encoders and 'prochaine_etape' in encoders:
            prochaine_etape = data.get('prochaine_etape', 'Inconnu')
            prochaine_etape_encoded = encoders['prochaine_etape'].transform(
                [prochaine_etape]
            )[0]
        else:
            prochaine_etape_encoded = 0  # fallback

        # 🎯 FEATURES (ADAPTE SI TON MODÈLE CHANGE)
        features = [[
            meeting_attended,
            meeting_diff_minutes,
            interet_client,
            probabilite_ressentie,
            budget_confirme,
            decideur_present,
            nb_relances,
            prochaine_etape_encoded
        ]]

        print("📊 FEATURES:", features)

        # 🔮 Prediction
        prediction = model.predict(features)[0]

        # ⚠️ certains modèles n'ont pas predict_proba
        try:
            proba = model.predict_proba(features)[0][1]
        except:
            proba = 0.5

        result = {
            "prediction": str(prediction),
            "score_percent": round(float(proba) * 100, 2)
        }

        print("🎯 RESULT:", result)

        return jsonify(result)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


# 🔥 IMPORTANT pour gunicorn
if __name__ != "__main__":
    app = app

# 🔥 Run local
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)