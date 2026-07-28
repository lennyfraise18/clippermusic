"""Configuration centrale : clés API, constantes vidéo, détection de l'encodeur.

Tout ce qui se règle se règle ici. Aucun autre module ne lit les variables
d'environnement directement.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Charge le fichier .env s'il existe (en local).
# Sur Hugging Face Spaces, les secrets sont déjà dans l'environnement.
load_dotenv()

# --- Racine du projet -------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"


def memoire_disponible_mo() -> float | None:
    """Mémoire réellement allouée au conteneur, en mégaoctets.

    Un hébergeur limite la mémoire par cgroup : `/proc/meminfo` montrerait
    alors la RAM de la machine physique, pas ce à quoi le conteneur a droit.
    On lit donc d'abord la limite cgroup, qui est la vraie contrainte.

    Renvoie None si l'information n'est pas lisible (Windows, par exemple).

    Définie ici, tout en haut, parce que plusieurs réglages en dépendent :
    l'application adapte sa résolution et son modèle à ce qu'elle trouve.
    """
    chemins = (
        "/sys/fs/cgroup/memory.max",                      # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",    # cgroup v1
    )
    for chemin in chemins:
        try:
            with open(chemin) as fichier:
                brut = fichier.read().strip()
        except OSError:
            continue
        if brut == "max":
            break
        try:
            octets = int(brut)
        except ValueError:
            continue
        # Une limite absurdement grande signifie « pas de limite ».
        if octets < (1 << 50):
            return octets / (1024 * 1024)

    try:
        with open("/proc/meminfo") as fichier:
            for ligne in fichier:
                if ligne.startswith("MemTotal:"):
                    return int(ligne.split()[1]) / 1024
    except OSError:
        pass
    return None


# Dossier de travail : fichiers intermédiaires (clips téléchargés, .ass, etc.).
# Nettoyé après chaque traitement par pipeline.py.
WORK_DIR = ROOT_DIR / "work"

# Dossier des vidéos finales, servies par Gradio pour le téléchargement.
OUTPUT_DIR = ROOT_DIR / "output"


# --- Clés API ---------------------------------------------------------------

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "").strip()


# --- Paramètres vidéo -------------------------------------------------------

# Résolution de sortie. 1080x1920 est le format natif de TikTok, Reels et
# Shorts, et reste le bon choix par défaut.
#
# Elle se règle par variable d'environnement parce que c'est le premier levier
# quand l'hébergement manque de mémoire : encoder en 720x1280 divise par 2,25
# le nombre de pixels à traiter, donc à peu près d'autant la mémoire dont
# ffmpeg a besoin. Les plateformes acceptent le 720x1280 sans broncher — elles
# le ré-encodent de toute façon.
#     VIDEO_WIDTH=720  VIDEO_HEIGHT=1280
def _resolution_par_defaut() -> tuple[int, int]:
    """1080x1920 si la mémoire le permet, 720x1280 sinon.

    Encoder du 1080x1920 demande à ffmpeg 500 à 700 Mo. En dessous d'environ
    1,2 Go alloués au conteneur, il se fait tuer par le système — sans message
    d'erreur, puisqu'il ne s'arrête pas de lui-même.

    Plutôt que d'échouer, on descend en 720x1280 : 2,25 fois moins de pixels à
    traiter, donc à peu près autant de mémoire en moins. TikTok, Reels et
    Shorts acceptent ce format sans réserve — ils ré-encodent de toute façon.
    Mieux vaut un clip un peu moins défini qu'aucun clip.
    """
    largeur = os.getenv("VIDEO_WIDTH")
    hauteur = os.getenv("VIDEO_HEIGHT")
    if largeur and hauteur:
        return int(largeur), int(hauteur)

    memoire = memoire_disponible_mo()
    if memoire is not None and memoire < 1200:
        return 720, 1280
    return 1080, 1920


VIDEO_WIDTH, VIDEO_HEIGHT = _resolution_par_defaut()
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))

# Au-delà de cette durée on ne traite pas la chanson entière : on sélectionne
# automatiquement le passage le plus dense en paroles (voir transcribe.py).
#
# 15 secondes, pour deux raisons qui vont dans le même sens :
#   - c'est le format des edits qui tournent le mieux sur TikTok et Reels ;
#   - le temps de traitement en dépend directement (nombre de plans à
#     télécharger, durée à encoder). Sur un hébergement modeste, une minute et
#     demie dépassait le délai au bout duquel le navigateur coupe la connexion.
#
# Le passage retenu n'est pas le début de la chanson mais le moment fort :
# voir transcribe._select_best_window, qui privilégie le refrain.
MAX_CLIP_SECONDS = int(os.getenv("MAX_CLIP_SECONDS", "15"))

# Durée d'audio réellement envoyée à Whisper.
#
# C'est LE facteur qui détermine le temps de traitement : la transcription est
# de loin l'étape la plus lente, et son coût est proportionnel à la durée
# analysée. Sans cette limite, une chanson de quatre minutes est transcrite
# en entier pour n'en garder que quinze secondes — l'essentiel du calcul part
# donc à la poubelle.
#
# 120 secondes prises après l'intro couvrent en général couplet, refrain,
# couplet, refrain : largement de quoi trouver le moment fort.
MAX_TRANSCRIBE_SECONDS = int(os.getenv("MAX_TRANSCRIBE_SECONDS", "120"))

# On ne commence pas à zéro : les premières secondes sont souvent une intro
# instrumentale, sans parole à transcrire.
TRANSCRIBE_START_RATIO = 0.15

# Durée minimale et maximale d'un plan vidéo de fond.
#
# Rythme resserré : sur un edit de 15 secondes, des plans de 4 à 6 secondes
# donnent une impression de diaporama. Entre 1,2 et 3 secondes, l'image change
# assez souvent pour tenir l'attention, sans empêcher de lire les paroles.
MIN_SHOT_SECONDS = float(os.getenv("MIN_SHOT_SECONDS", "1.2"))
MAX_SHOT_SECONDS = float(os.getenv("MAX_SHOT_SECONDS", "3.0"))

# Mouvement de caméra lent sur chaque plan (zoom avant ou arrière alterné).
# Un plan fixe paraît figé ; un léger zoom continu donne de la vie, même sur
# une image statique. Désactivable, car c'est l'effet le plus coûteux en calcul.
ZOOM_ACTIF = os.getenv("ZOOM_ACTIF", "1").strip() not in {"0", "false", "no"}

# Amplitude du zoom : 1.12 = 12 % d'agrandissement sur la durée du plan.
# Au-delà, le mouvement devient visible au point de distraire.
ZOOM_AMPLITUDE = float(os.getenv("ZOOM_AMPLITUDE", "1.12"))

# Refus immédiat des fichiers trop longs ou trop courts (validation à l'upload).
MIN_AUDIO_SECONDS = 5
MAX_AUDIO_SECONDS = 60 * 12  # 12 minutes


# --- Transcription ----------------------------------------------------------

# « base » par défaut : sur un processeur d'hébergeur mutualisé, « small » est
# trois à quatre fois plus lent pour un gain de précision modeste sur du chant.
# Le choix reste modifiable dans l'interface et par variable d'environnement.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip() or "base"

# Nombre de cœurs utilisés par la transcription.
# 0 laisse la bibliothèque décider. Sur un hébergement mutualisé, la limiter
# évite qu'elle réserve de la mémoire pour des cœurs qu'elle n'aura jamais.
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))

# Whisper "hallucine" volontiers sur du silence ou de l'instrumental.
# Tout segment dont la confiance moyenne est sous ce seuil est jeté.
MIN_SEGMENT_CONFIDENCE = 0.35

# Nombre minimum de mots retenus, en dessous duquel on considère qu'il n'y a
# pas de paroles exploitables (fichier instrumental).
MIN_WORDS_REQUIRED = 8


# --- Police des sous-titres -------------------------------------------------

# PIÈGE CLASSIQUE : si la police nommée dans le .ass n'existe pas dans le
# conteneur, ffmpeg affiche des carrés vides — sans message d'erreur.
# Le Dockerfile installe fonts-dejavu ; "DejaVu Sans" est donc garanti sous Linux.
# Sous Windows (dev local) DejaVu n'existe pas, on retombe sur Arial.
SUBTITLE_FONT = "DejaVu Sans" if os.name != "nt" else "Arial"
SUBTITLE_FONT_SIZE = 96


def ensure_dirs() -> None:
    """Crée les dossiers de travail s'ils n'existent pas."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ffmpeg_path() -> str:
    """Chemin de l'exécutable ffmpeg, ou lève une erreur explicite."""
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg est introuvable. Installe-le et relance un nouveau terminal.\n"
            "Windows : winget install Gyan.FFmpeg\n"
            "Linux   : apt-get install ffmpeg"
        )
    return path


def ffprobe_path() -> str:
    """Chemin de l'exécutable ffprobe, ou lève une erreur explicite."""
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError(
            "ffprobe est introuvable (il est fourni avec ffmpeg). "
            "Installe ffmpeg et relance un nouveau terminal."
        )
    return path


# Cache de la détection d'encodeur : le test coûte ~1 s, inutile de le refaire.
_encoder_cache: str | None = None


def detect_video_encoder() -> str:
    """Renvoie le meilleur encodeur H.264 réellement fonctionnel.

    On ne se contente pas de lire `ffmpeg -encoders` : un encodeur peut être
    compilé dans ffmpeg sans que le matériel soit présent. On tente donc un
    vrai encodage d'une seconde de mire, et on garde le premier qui réussit.
    libx264 (CPU) est le dernier recours et marche partout.
    """
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    candidates = ["h264_nvenc", "h264_qsv", "h264_videotoolbox", "libx264"]

    for encoder in candidates:
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "test.mp4"
            command = [
                ffmpeg_path(), "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
                "-c:v", encoder,
                str(test_file),
            ]
            try:
                result = subprocess.run(command, capture_output=True, timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                continue
            if result.returncode == 0 and test_file.exists():
                _encoder_cache = encoder
                return encoder

    # Si même libx264 échoue, quelque chose de plus grave se passe.
    raise RuntimeError(
        "Aucun encodeur H.264 fonctionnel dans ffmpeg (libx264 inclus). "
        "L'installation de ffmpeg est probablement incomplète."
    )


# Plafond de débit vidéo.
# Sans lui, l'encodage à qualité constante produisait ~10 Mbps, soit 140 Mo pour
# moins de deux minutes : au-delà de ce qu'Instagram accepte pour un Reel, et
# pénible à téléverser. 7 Mbps reste très confortable en 1080x1920.
MAX_BITRATE = "7M"
BITRATE_BUFFER = "14M"


def encoder_options(encoder: str) -> list[str]:
    """Options de qualité/vitesse adaptées à chaque encodeur.

    Chaque encodeur garde son mode « qualité constante » (meilleur rendu à
    poids égal), mais borné par un débit maximum pour que le fichier reste
    publiable sur les réseaux sociaux.
    """
    plafond = ["-maxrate", MAX_BITRATE, "-bufsize", BITRATE_BUFFER]

    if encoder == "libx264":
        return ["-preset", "veryfast", "-crf", "23", *plafond]
    if encoder == "h264_nvenc":
        return ["-preset", "p4", "-cq", "23", *plafond]
    if encoder == "h264_qsv":
        return ["-preset", "veryfast", "-global_quality", "23", *plafond]
    if encoder == "h264_videotoolbox":
        return ["-b:v", "6M", *plafond]
    return plafond
