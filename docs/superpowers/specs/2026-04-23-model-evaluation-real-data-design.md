# Design Spec — Évaluation du modèle sur données réelles Kaggle

**Date :** 2026-04-23  
**Projet :** Anomaly — Fraud Detection MLOps  
**Fichier cible :** `tests/evaluate_model.py`

---

## Objectif

Valider la qualité du modèle Random Forest entraîné sur le dataset Kaggle Credit Card Fraud Detection en testant sur la totalité des 284 807 transactions, avec deux modes de sortie : rapport ML complet (terminal + fichiers) et test du pipeline API pour alimenter le monitoring Prometheus/Grafana.

---

## Architecture

Un seul script `tests/evaluate_model.py` en deux phases séquentielles.

### Phase 1 — Évaluation ML directe

- Charge `data/creditcard.csv` (284 807 lignes)
- Applique le `RobustScaler` depuis `app/robust_scaler.joblib` sur `Amount` et `Time`
- Construit le vecteur de features dans l'ordre exact : `[scaled_amount, scaled_time, V1…V28]`
- Lance `model.predict()` et `model.predict_proba()` sur tout le dataset
- Compare avec la colonne `Class` (ground truth)
- Affiche le rapport dans le terminal
- Sauvegarde les fichiers dans `evaluation_output/`

### Phase 2 — Test API avec échantillon représentatif

- Extrait toutes les 492 fraudes du CSV
- Tire aléatoirement 500 légitimes (seed=42 pour reproductibilité)
- Envoie chaque transaction à `POST http://127.0.0.1:8001/predict` (valeurs originales non scalées, l'API gère le scaling)
- Compare les réponses API avec les vraies étiquettes
- Affiche un résumé de précision et latence moyenne

---

## Métriques produites

### Terminal — Phase 1
- Nombre de transactions, répartition fraudes/légitimes
- Matrice de confusion (4 valeurs : TP, TN, FP, FN)
- Classification report : precision, recall, F1 par classe
- ROC-AUC score
- Temps d'évaluation total

### Terminal — Phase 2
- Nombre de fraudes correctement détectées (recall API)
- Nombre de légitimes correctement classées (specificity API)
- Latence moyenne par requête

### Fichiers sauvegardés — `evaluation_output/`
| Fichier | Contenu |
|---|---|
| `report.csv` | Colonnes : `id, amount, true_label, predicted_label, fraud_probability` |
| `confusion_matrix.png` | Heatmap seaborn annotée |
| `roc_curve.png` | Courbe ROC avec AUC affiché en légende |

---

## Preprocessing

Le script réutilise exactement le même pipeline que `train.py` :

1. Chargement du CSV complet sans filtrage
2. Chargement du scaler depuis `app/robust_scaler.joblib`
3. Application du scaler sur `Amount` et `Time` séparément
4. Ordre des features : `[scaled_amount, scaled_time, V1, V2, …, V28]`
5. Ground truth : colonne `Class` (0 = légitime, 1 = fraude)

Pour la Phase 2, les transactions utilisent les valeurs brutes du CSV (Amount, Time, V1-V28) car l'API applique elle-même le scaling dans `app/model.py`.

---

## Dépendances

- `app/random_forest_fraud_model.joblib` — modèle entraîné
- `app/robust_scaler.joblib` — scaler entraîné
- `data/creditcard.csv` — dataset Kaggle complet
- API démarrée sur `http://127.0.0.1:8001` (requise pour Phase 2 uniquement)

---

## Fichiers à créer

| Fichier | Action |
|---|---|
| `tests/evaluate_model.py` | Créer |
| `evaluation_output/` | Créé automatiquement par le script |

Aucun fichier existant n'est modifié.

---

## Contraintes et décisions

- La Phase 1 ne nécessite pas que l'API soit démarrée.
- La Phase 2 s'arrête proprement avec un message d'erreur si l'API n'est pas accessible, sans faire échouer la Phase 1.
- Le CSV source n'est jamais modifié.
- Seed fixe (`random_state=42`) pour l'échantillon légitimes en Phase 2, garantissant la reproductibilité.
