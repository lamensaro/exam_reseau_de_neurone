import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment


# =========================================================
# PARAMÈTRES (identiques à l'entraînement)
# =========================================================

DUREE_MAX = 3
TAUX_ECHANTILLONNAGE = 22050
N_MFCC = 40

# Doit être EXACTEMENT le même ordre que dans le script d'entraînement
CLASSES = ["Moi", "User1", "User2", "Autres"]

DOSSIER_MODELE = "/home/ubuntu/modele"

# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================
# Un seul fichier : la normalisation est intégrée au modèle,
# plus besoin de charger un scaler séparé.

modele = tf.keras.models.load_model(
    os.path.join(DOSSIER_MODELE, "modele_voix.keras")
)

# =========================================================
# EXTRACTION MFCC (même fonction que pour l'entraînement)
# =========================================================

def convertir_en_wav(chemin_fichier):
    """Convertit .aac/.m4a/.mp3... en .wav temporaire via ffmpeg."""
    if chemin_fichier.lower().endswith(".wav"):
        return chemin_fichier

    audio = AudioSegment.from_file(chemin_fichier)
    fichier_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(fichier_temp.name, format="wav")
    return fichier_temp.name


def extraire_mfcc(chemin_fichier):
    chemin_wav = convertir_en_wav(chemin_fichier)

    signal, sr = librosa.load(
        chemin_wav,
        sr=TAUX_ECHANTILLONNAGE,
        duration=DUREE_MAX
    )

    if chemin_wav != chemin_fichier:
        os.remove(chemin_wav)

    longueur_cible = TAUX_ECHANTILLONNAGE * DUREE_MAX
    if len(signal) < longueur_cible:
        signal = np.pad(signal, (0, longueur_cible - len(signal)))
    else:
        signal = signal[:longueur_cible]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    caracteristiques = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return caracteristiques

# =========================================================
# FONCTION DE PRÉDICTION
# =========================================================

def predire_voix(chemin_audio):
    caracteristiques = extraire_mfcc(chemin_audio)
    entree = np.expand_dims(caracteristiques, axis=0)   # ajoute la dimension batch

    prediction = modele.predict(entree, verbose=0)[0]   # vecteur de probabilités (softmax)
    indice = int(np.argmax(prediction))
    nom_classe = CLASSES[indice]
    confiance = float(prediction[indice])

    print("\n===== PRÉDICTION =====")
    print("Fichier :", chemin_audio)
    for i, nom in enumerate(CLASSES):
        print(f"  {nom:8s} : {prediction[i]:.4f}")

    if nom_classe == "Autres":
        print(f"\n❌ C'EST LA VOIX D'UN AUTRE (confiance : {confiance:.2%})")
        print("🔒 DÉVERROUILLAGE REFUSÉ")
    else:
        print(f"\n✅ C'EST LA VOIX DE {nom_classe.upper()} (confiance : {confiance:.2%})")
        print("🔓 DÉVERROUILLAGE")

    return nom_classe, confiance

# =========================================================
# TEST SUR UN FICHIER AUDIO EXISTANT
# =========================================================

CHEMIN_AUDIO = "/home/ubuntu/donnes/tests/User2/arctic_a0013.wav"

predire_voix(CHEMIN_AUDIO)

