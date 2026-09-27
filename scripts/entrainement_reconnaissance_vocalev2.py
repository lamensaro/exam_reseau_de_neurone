import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# =========================================================
# PARAMÈTRES
# =========================================================

DUREE_MAX = 3                 # durée standardisée de chaque enregistrement (secondes)
TAUX_ECHANTILLONNAGE = 22050  # fréquence d'échantillonnage
N_MFCC = 40                   # nombre de coefficients MFCC extraits
EPOCHS = 100
BATCH_SIZE = 16

DOSSIER_DATASET = "/home/ubuntu/donnes/entrainements"
DOSSIER_MODELE = "/home/ubuntu/modele"

# L'ordre ici DOIT être exactement le même que dans le script de prédiction
CLASSES = ["Moi", "User1", "User2", "Autres"]

# =========================================================
# EXTRACTION DES CARACTÉRISTIQUES (MFCC)
# =========================================================

def convertir_en_wav(chemin_fichier):
    """
    Convertit n'importe quel format audio (.aac, .m4a, .mp3, ...)
    en .wav temporaire, lisible ensuite par librosa/soundfile.
    Retourne directement le chemin d'origine si c'est déjà un .wav.
    """
    if chemin_fichier.lower().endswith(".wav"):
        return chemin_fichier

    audio = AudioSegment.from_file(chemin_fichier)  # utilise ffmpeg en interne
    fichier_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(fichier_temp.name, format="wav")
    return fichier_temp.name


def extraire_mfcc(chemin_fichier):
    """
    Charge un fichier audio (wav, aac, m4a, mp3...) et extrait un
    vecteur de caractéristiques (moyenne + écart-type des MFCC
    dans le temps). Ce vecteur résume le timbre et le spectre de
    la voix, indépendamment de la phrase prononcée.
    """
    chemin_wav = convertir_en_wav(chemin_fichier)

    signal, sr = librosa.load(
        chemin_wav,
        sr=TAUX_ECHANTILLONNAGE,
        duration=DUREE_MAX
    )

    # Nettoyage du fichier temporaire s'il a été créé
    if chemin_wav != chemin_fichier:
        os.remove(chemin_wav)

    # Uniformiser la longueur (padding ou coupe)
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
# CONSTRUCTION DU DATASET (X, y)
# =========================================================
# Structure attendue :
# entrainements/
# ├── Moi/
# ├── User1/
# ├── User2/
# └── Autres/

X = []
y = []

for label, nom_classe in enumerate(CLASSES):
    dossier_classe = os.path.join(DOSSIER_DATASET, nom_classe)
    fichiers = os.listdir(dossier_classe)

    print(f"\nTraitement de la classe '{nom_classe}' ({len(fichiers)} fichiers)")

    for nom_fichier in fichiers:
        chemin = os.path.join(dossier_classe, nom_fichier)
        try:
            caracteristiques = extraire_mfcc(chemin)
            X.append(caracteristiques)
            y.append(label)
        except Exception as e:
            print(f"Erreur avec {nom_fichier} : {e}")

X = np.array(X)
y = np.array(y)

print("\nForme du dataset X :", X.shape)
print("Forme du dataset y :", y.shape)
print("Classes             :", CLASSES)

# =========================================================
# SÉPARATION ENTRAÎNEMENT / TEST
# =========================================================
# (pas de normalisation manuelle ici : elle est intégrée au modèle
# via la couche Normalization ci-dessous, pour ne garder qu'un seul
# fichier de modèle à la fin)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nEntraînement :", X_train.shape[0], "exemples")
print("Test         :", X_test.shape[0], "exemples")

# =========================================================
# COUCHE DE NORMALISATION (fait partie du modèle)
# =========================================================
# Elle apprend la moyenne/variance des caractéristiques sur le
# jeu d'entraînement, puis ces valeurs sont sauvegardées AVEC
# le modèle : plus besoin de scaler séparé.

normaliseur = layers.Normalization(axis=-1)
normaliseur.adapt(X_train)

# =========================================================
# MODÈLE : RÉSEAU DE NEURONES DENSE MULTI-CLASSES
# =========================================================

modele = tf.keras.Sequential([
    layers.Input(shape=(X.shape[1],)),

    normaliseur,   # normalisation intégrée

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    # Une sortie par classe (Moi / User1 / User2 / Autres)
    layers.Dense(len(CLASSES), activation='softmax')
])

# =========================================================
# COMPILATION
# =========================================================

modele.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # y = entiers (0,1,2,3), pas one-hot
    metrics=['accuracy']
)

# =========================================================
# RÉSUMÉ
# =========================================================

print("\n===== RÉSUMÉ DU MODÈLE =====")
modele.summary()

# =========================================================
# ENTRAÎNEMENT
# =========================================================

print("\n===== ENTRAÎNEMENT =====")

historique = modele.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================================================
# ÉVALUATION
# =========================================================

print("\n===== ÉVALUATION =====")

loss, accuracy = modele.evaluate(X_test, y_test)

print(f"\nLoss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")

# =========================================================
# SAUVEGARDE DU MODÈLE (un seul fichier, tout compris)
# =========================================================

os.makedirs(DOSSIER_MODELE, exist_ok=True)

chemin_modele = os.path.join(DOSSIER_MODELE, "modele_voix.keras")
modele.save(chemin_modele)

print("\nModèle sauvegardé :", chemin_modele)

