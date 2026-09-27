# 🎙️ Déverrouillage par Reconnaissance de la Voix

Système de reconnaissance du **locuteur** (pas du contenu) basé sur un réseau de neurones (RNA). Le but est d'identifier qui parle, peu importe la phrase prononcée.

## 🎯 Objectif

Reconnaître automatiquement si une voix appartient à une personne autorisée (`Moi`, `User1`, `User2`) ou à quelqu'un d'inconnu (`Autres`), pour simuler un déverrouillage vocal.

## 🧠 Fonctionnement

1. Extraction des caractéristiques **MFCC** de chaque enregistrement audio
2. Entraînement d'un réseau dense (Keras/TensorFlow) sur 4 classes : `Moi`, `User1`, `User2`, `Autres`
3. La normalisation des données est intégrée directement dans le modèle (une seule sortie `.keras`, sans fichier annexe)
4. En prédiction, le modèle indique **qui** parle ou signale une voix inconnue


## 📂 Fichiers

- `entrainement_reconnaissance_vocale.py` — extraction des MFCC, entraînement et sauvegarde du modèle
- `prediction_reconnaissance_vocale.py` — chargement du modèle et prédiction sur un fichier audio

## 🛠️ Technologies

Python · TensorFlow / Keras · Librosa (MFCC) · Pydub (conversion audio)

## ▶️ Utilisation

```bash
python3 entrainement_reconnaissance_vocale.py
python3 prediction_reconnaissance_vocale.py
```
