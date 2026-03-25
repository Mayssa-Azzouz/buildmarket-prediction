"""
BuildMarket — Prediction API (Flask) v2.0
==========================================
9 features post-réunion uniquement

Installer  : pip install flask scikit-learn pandas numpy
Lancer     : python app.py
Endpoint   : POST /predict
"""

from flask import Flask, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

with open('model.pkl',          'rb') as f: model    = pickle.load(f)
with open('encoders.pkl',       'rb') as f: encoders = pickle.load(f)
with open('model_metadata.json','r')  as f: metadata = json.load(f)

FEATURE_COLS = metadata['feature_cols']
CAT_COLS     = metadata['cat_cols']

# Valeurs par défaut si champ manquant
DEFAULTS = {
    'meeting_attended':       0,
    'meeting_diff_minutes':   999,
    'interet_client':         3,
    'probabilite_ressentie':  50,
    'budget_confirme':        0,
    'decideur_present':       0,
    'prochaine_etape':        'Rappel',
    'nb_relances':            1,
}


def encode_input(data: dict) -> list:
    row = []
    for col in FEATURE_COLS:
        val = data.get(col, DEFAULTS.get(col))

        if col in CAT_COLS:
            le = encoders[col]
            if str(val) not in le.classes_:
                val = DEFAULTS[col]
            val = le.transform([str(val)])[0]
        else:
            val = float(val) if val is not None else DEFAULTS[col]

        row.append(val)
    return row


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Body JSON manquant'}), 400

        # Normalisation booléens
        for field in ['meeting_attended', 'budget_confirme', 'decideur_present']:
            if field in data:
                data[field] = 1 if data[field] in [True, 1, 'true', 'True'] else 0

        features   = np.array([encode_input(data)])
        proba      = model.predict_proba(features)[0][1]
        prediction = 'CONVERTI' if proba >= 0.5 else 'NON_CONVERTI'

        if proba >= 0.75 or proba <= 0.25:
            confidence = 'HIGH'
        elif proba >= 0.60 or proba <= 0.40:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return jsonify({
            'conversion_probability': round(float(proba), 4),
            'prediction':             prediction,
            'confidence':             confidence,
            'score_percent':          round(float(proba) * 100, 1)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'model':    metadata.get('model_name'),
        'version':  metadata.get('version'),
        'features': len(FEATURE_COLS)
    }), 200


@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    return jsonify(metadata.get('feature_importance', {})), 200


if __name__ == '__main__':
    print("🚀 BuildMarket Prediction API v2.0")
    print("   POST /predict            → prédire la conversion")
    print("   GET  /health             → status")
    print("   GET  /feature-importance → importance des features")
    app.run(host='0.0.0.0', port=5000, debug=False)
