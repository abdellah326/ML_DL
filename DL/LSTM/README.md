# 📈 Prédiction des Prix d'Actions en Bourse avec LSTM

## 📝 Description du Projet
Ce projet a été réalisé dans le cadre d'un travail pratique (TP) en Deep Learning à l'**École Supérieure de Technologie (EST) de Fquih Ben Salah** (Filière : IDS). 
L'objectif principal est de prédire le prix de clôture (`Close`) d'une action boursière pour le jour suivant en utilisant une architecture de réseau de neurones récurrents **LSTM (Long Short-Term Memory)**, particulièrement adaptée à l'analyse des séries temporelles.

## 🎯 Objectifs
- Analyser et prétraiter des données financières historiques.
- Construire des séquences temporelles (fenêtre glissante de 60 jours).
- Développer, entraîner et évaluer un modèle Deep Learning (LSTM).
- Visualiser les prédictions par rapport aux valeurs réelles du marché.

## 📊 Base de Données (Dataset)
Le modèle s'entraîne sur un historique de données boursières (`dataset.csv`). Les caractéristiques disponibles sont :
- `Date` : Date de la transaction (utilisée comme index).
- `Open` : Prix d'ouverture.
- `High` : Prix le plus haut de la journée.
- `Low` : Prix le plus bas de la journée.
- `Close` : **Prix de clôture (Target)**.
- `Adj Close` : Prix de clôture ajusté.
- `Volume` : Nombre d'actions échangées.

## 🛠️ Technologies et Bibliothèques Utilisées
- **Langage** : Python 3
- **Manipulation de données** : Pandas, NumPy
- **Machine Learning / Preprocessing** : Scikit-learn (MinMaxScaler)
- **Deep Learning** : TensorFlow / Keras (Sequential, LSTM, Dense, Dropout)
- **Visualisation** : Matplotlib

## ⚙️ Architecture du Modèle
Le réseau de neurones a été conçu pour éviter le surapprentissage (overfitting) et capturer les tendances complexes :
1. **Couche d'Entrée (Input)** : Accepte des séquences de forme `(60, 1)`.
2. **Couche LSTM 1** : 50 neurones avec `return_sequences=True`.
3. **Couche Dropout 1** : 20% de désactivation.
4. **Couche LSTM 2** : 50 neurones avec `return_sequences=False`.
5. **Couche Dropout 2** : 20% de désactivation.
6. **Couches Denses** : Une couche de 25 neurones, suivie de la couche de sortie (1 neurone).
- **Optimiseur** : Adam
- **Fonction de perte** : Mean Squared Error (MSE)

## 🚀 Comment Exécuter le Projet
1. Assurez-vous d'avoir installé les bibliothèques requises :
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
