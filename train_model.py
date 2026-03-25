"""
BuildMarket — Lead Conversion Prediction Model
================================================
9 features POST-RÉUNION uniquement (complémentaire au modèle de lead scoring)

Exécuter : python train_model.py
Résultat : model.pkl + encoders.pkl + model_metadata.json + synthetic_dataset.csv
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 1000

# ══════════════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
#    9 features post-réunion uniquement
# ══════════════════════════════════════════════════════════════════════

def generate_dataset(n=1000):

    data = []

    for _ in range(n):

        # ── Réunion honorée ou non ────────────────────────────────────
        meeting_attended = np.random.choice([True, False], p=[0.65, 0.35])

        # ── Ponctualité ───────────────────────────────────────────────
        if meeting_attended:
            meeting_diff_minutes = int(np.clip(np.random.normal(5, 20), -15, 60))
        else:
            meeting_diff_minutes = 999  # sentinelle = no show

        # ── Features post-réunion ─────────────────────────────────────
        if meeting_attended:
            interet_client        = np.random.choice([1,2,3,4,5], p=[0.05,0.10,0.25,0.35,0.25])
            probabilite_ressentie = int(np.clip(np.random.normal(58, 20), 0, 100))
            budget_confirme       = np.random.choice([True, False], p=[0.55, 0.45])
            decideur_present      = np.random.choice([True, False], p=[0.60, 0.40])
            prochaine_etape       = np.random.choice(
                ['Devis', 'Rappel', 'Demo', 'Perdu'],
                p=[0.35, 0.25, 0.20, 0.20]
            )
            nb_relances           = int(np.clip(np.random.poisson(1.5), 0, 5))
        else:
            interet_client        = np.random.choice([1,2,3], p=[0.50,0.30,0.20])
            probabilite_ressentie = int(np.clip(np.random.normal(22, 12), 0, 50))
            budget_confirme       = False
            decideur_present      = False
            prochaine_etape       = np.random.choice(['Rappel', 'Perdu'], p=[0.35, 0.65])
            nb_relances           = int(np.clip(np.random.poisson(3), 0, 8))

        # ── Calcul du label (logique métier réaliste) ─────────────────
        score = 0.0

        # Probabilité ressentie — poids le plus fort
        score += probabilite_ressentie / 100 * 0.30

        # Intérêt client
        score += (interet_client - 1) / 4 * 0.25

        # Prochaine étape
        etape_score = {'Devis': 0.15, 'Demo': 0.08, 'Rappel': 0.02, 'Perdu': -0.20}
        score += etape_score.get(prochaine_etape, 0)

        # Budget confirmé
        score += 0.12 if budget_confirme else 0.0

        # Décideur présent
        score += 0.10 if decideur_present else 0.0

        # Réunion honorée
        score += 0.08 if meeting_attended else -0.10

        # Ponctualité
        if meeting_attended and meeting_diff_minutes != 999:
            if meeting_diff_minutes <= 5:
                score += 0.05
            elif meeting_diff_minutes >= 45:
                score -= 0.05

        # Nb relances (trop = signal négatif)
        score -= nb_relances * 0.025

        # Bruit
        score += np.random.normal(0, 0.05)

        # Label binaire via sigmoid
        prob_convert = 1 / (1 + np.exp(-8 * (score - 0.42)))
        is_converted = int(np.random.random() < prob_convert)

        data.append({
            'meeting_attended':       int(meeting_attended),
            'meeting_diff_minutes':   meeting_diff_minutes,
            'interet_client':         interet_client,
            'probabilite_ressentie':  probabilite_ressentie,
            'budget_confirme':        int(budget_confirme),
            'decideur_present':       int(decideur_present),
            'prochaine_etape':        prochaine_etape,
            'nb_relances':            nb_relances,
            'is_converted':           is_converted
        })

    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess(df):
    cat_cols = ['prochaine_etape']
    encoders = {}
    df_enc   = df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    return df_enc, encoders, cat_cols


# ══════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'meeting_attended',
    'meeting_diff_minutes',
    'interet_client',
    'probabilite_ressentie',
    'budget_confirme',
    'decideur_present',
    'prochaine_etape',
    'nb_relances',
]

def train_and_evaluate(df_enc):

    X = df_enc[FEATURE_COLS]
    y = df_enc['is_converted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n📊 Comparaison des modèles (Cross-Validation 5 folds)\n")
    print(f"{'Modèle':<25} {'AUC':>10} {'Accuracy':>12}")
    print("─" * 50)

    for name, model in models.items():
        if name == 'LogisticRegression':
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        else:
            pipeline = Pipeline([('model', model)])

        auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

        results[name] = {
            'pipeline': pipeline,
            'auc_mean': auc_scores.mean(),
            'acc_mean': acc_scores.mean()
        }

        print(f"{name:<25} {auc_scores.mean():.4f} ± {auc_scores.std():.4f}   {acc_scores.mean():.4f}")

    best_name     = max(results, key=lambda k: results[k]['auc_mean'])
    best_pipeline = results[best_name]['pipeline']

    print(f"\n✅ Meilleur modèle : {best_name}")
    print(f"   AUC = {results[best_name]['auc_mean']:.4f}")

    best_pipeline.fit(X_train, y_train)

    y_pred       = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print(f"\n📈 Résultats sur le test set :")
    print(f"   Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   AUC-ROC  : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non converti','Converti'])}")

    return best_pipeline, best_name, X_test, y_test, y_pred_proba


# ══════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

def print_feature_importance(pipeline, model_name):
    model = pipeline.named_steps['model']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1]

        print(f"\n🎯 Importance des features ({model_name}) :\n")
        print(f"{'Feature':<30} {'Importance':>12}")
        print("─" * 45)

        importance_data = {}
        for i in indices:
            bar = '█' * int(importances[i] * 50)
            print(f"{FEATURE_COLS[i]:<30} {importances[i]:.4f}  {bar}")
            importance_data[FEATURE_COLS[i]] = round(float(importances[i]), 4)

        return importance_data

    return {}


# ══════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════

def save_artifacts(pipeline, encoders, model_name, importance_data, cat_cols):

    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n✅ model.pkl sauvegardé")

    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("✅ encoders.pkl sauvegardé")

    metadata = {
        'model_name':         model_name,
        'feature_cols':       FEATURE_COLS,
        'cat_cols':           cat_cols,
        'feature_importance': importance_data,
        'version':            '2.0',
        'trained_on':         'synthetic_buildmarket_post_meeting_1000',
        'description':        '9 features post-reunion uniquement — complementaire au lead scoring'
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ model_metadata.json sauvegardé")


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 60)
    print("  BuildMarket — Modèle Conversion Post-Réunion v2.0")
    print("  9 features | Complémentaire au Lead Scoring")
    print("=" * 60)

    print(f"\n📦 Génération de {N} leads synthétiques...")
    df = generate_dataset(N)

    taux = df['is_converted'].mean()
    print(f"   Taux de conversion simulé : {taux:.1%}")
    print(f"   Convertis     : {df['is_converted'].sum()}")
    print(f"   Non convertis : {(1 - df['is_converted']).sum()}")

    df.to_csv('synthetic_dataset.csv', index=False)
    print(f"   synthetic_dataset.csv sauvegardé ({len(df)} lignes)")

    print("\n⚙️  Preprocessing...")
    df_enc, encoders, cat_cols = preprocess(df)

    pipeline, best_name, X_test, y_test, y_pred_proba = train_and_evaluate(df_enc)

    importance_data = print_feature_importance(pipeline, best_name)

    save_artifacts(pipeline, encoders, best_name, importance_data, cat_cols)

    print("\n" + "=" * 60)
    print("  ✅ Terminé ! Fichiers générés :")
    print("    → model.pkl")
    print("    → encoders.pkl")
    print("    → model_metadata.json")
    print("    → synthetic_dataset.csv")
    print("=" * 60)
