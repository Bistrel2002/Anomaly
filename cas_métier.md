# 📊 ÉTAPE 2 — Compréhension du Cas Métier : Fraude à la Carte Bancaire

Ce document définit précisément le cas d'usage métier pour la pipeline MLOps de détection d'anomalies, en se concentrant sur le dataset sélectionné : **Credit Card Fraud Detection** (`data/creditcard.csv`).

---

### 1. Qu'est-ce qu'une anomalie dans ce contexte ?

Dans le cadre de ce projet, une anomalie représente une **transaction frauduleuse** à la carte bancaire.
Il s'agit d'une transaction qui rompt avec les schémas de consommation habituels ou physiques normaux, signifiant que la carte est utilisée par une personne non autorisée.

*   **Exemples d'anomalies typiques capturées par les modèles :**
    *   **"Le voyage impossible" :** Une utilisation physique à Paris, suivie d'une tentative de retrait à New York une heure plus tard.
    *   **Card Testing :** Une série de micro-transactions (ex: 0,05 €) pour vérifier si la carte est active avant un gros achat.
    *   **Incohérence de montant/fréquence :** Des achats de montants exceptionnellement élevés ou une fréquence de transactions inhabituellement dense pour ce profil client.

Dans le dataset `creditcard.csv`, ces anomalies sont représentées par la classe d'intérêt (la cible à prédire). Les variables explicatives (features V1 à V28) sont déjà anonymisées via PCA pour des raisons de confidentialité, mais elles capturent mathématiquement ces déviations comportementales.

---

### 2. Quel est l'impact business ?

L'impact de la fraude (et de sa détection) est double et asymétrique :

*   **Impact d'un Faux Négatif (Fraude non détectée) :**
    *   Perte financière directe pour la banque (remboursement du client).
    *   Coûts opérationnels liés à l'investigation et à la gestion du litige.
    *   Atteinte à la confiance et à la réputation de l'institution.

*   **Impact d'un Faux Positif (Transaction légitime bloquée) :**
    *   **Friction client majeure :** Bloquer la carte d'un client légitime en plein voyage génère une immense frustration et un risque élevé de churn (départ vers la concurrence).
    *   Perte de revenus directs sur la transaction bloquée (frais d'interchange perdus).

**Objectif métier pour le modèle ML :**
L'enjeu n'est pas seulement de maximiser la détection globale (Accuracy), mais de trouver l'équilibre parfait entre le **Rappel (Recall)** — pour attraper un maximum de fraudes — et la **Précision (Precision)** — pour minimiser les faux positifs. Le modèle sera jugé sur des métriques comme l'AUPRC (Area Under the Precision-Recall Curve), particulièrement adaptée à ce problème fortement déséquilibré.

---

### 3. Quelle est la fréquence des données ?

Le système de détection est un système en **Temps Réel (Real-Time)** pur, de type "Pre-auth" (avant autorisation bancaire).

*   **Latence critique :** La décision algorithmique (Bloquer, Autoriser, ou Demande d'authentification forte via 3D Secure) doit être prise et retournée au terminal de paiement en **moins de 100 à 300 millisecondes maximum**.
*   Ce flux asynchrone et rapide impose une architecture de *Serving* (Étape 4 de l'architecture) ultra-performante et optimisée, où le modèle ML réside déjà en mémoire pour l'inférence.

---

### 4. Quel volume par minute ?

L'analyse de volume est basée sur notre dataset de référence `creditcard.csv`.

*   **Caractéristiques du Dataset :** Le fichier représente une période de **48 heures**.
*   **Nombre total de transactions :** 284 807
*   **Transactions frauduleuses (Anomalies) :** 492
*   **Déséquilibre de classe :** 0,172% (un problème *extrêmement* déséquilibré, typique du monde réel).

**Calcul du flux théorique à gérer :**
*   Volume total sur 48h = 284 807 transactions.
*   Volume par heure = 284 807 / 48 ≈ **5 933 transactions / heure**.
*   **Volume par minute = ~99 transactions / minute en moyenne**.

*(Note : Dans un environnement bancaire global réel au niveau de la passerelle, ce volume serait considérablement plus élevé, pouvant atteindre des milliers de transactions par seconde lors des pics. L'architecture technique MLOps conçue doit donc être capable de **scaler** dynamiquement pour absorber ces variations avec Kafka et un cluster Kubernetes)*.
