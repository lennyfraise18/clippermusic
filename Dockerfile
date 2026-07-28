# Image du Space Hugging Face.
# Construction locale (optionnelle) :  docker build -t clip-paroles .
# Lancement local                    :  docker run -p 7860:7860 --env-file .env clip-paroles

FROM python:3.12-slim

# --- Paquets système --------------------------------------------------------
# ffmpeg          : tout le montage vidéo.
# fonts-dejavu-core + fontconfig : PIÈGE N°1 DU PROJET. Sans police installée,
#   l'incrustation des sous-titres .ass affiche des carrés vides ou rien du tout,
#   SANS message d'erreur. La police déclarée dans modules/config.py
#   (« DejaVu Sans ») vient de ce paquet.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fontconfig \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# --- Utilisateur non-root ---------------------------------------------------
# Hugging Face Spaces impose l'UID 1000 : un conteneur qui tourne en root
# n'a pas le droit d'écrire dans son propre dossier de travail.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# --- Dépendances Python -----------------------------------------------------
COPY --chown=user requirements.txt .

# PIÈGE N°2 : sur Linux, `pip install torch` prend par défaut le wheel CUDA,
# soit ~2,5 Go de bibliothèques GPU parfaitement inutiles sur un Space gratuit
# (qui n'a pas de GPU) — et l'installation échoue souvent par manque de place.
# On installe donc explicitement la variante CPU EN PREMIER : le
# `pip install -r requirements.txt` suivant la verra déjà satisfaite.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Modèles spaCy (~15 Mo chacun) : français et anglais.
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

# Les modèles Whisper se téléchargent au premier appel. On les met dans
# l'image : sans ça, la toute première génération attend plusieurs minutes
# avant même de commencer — mauvais effet garanti en démo.
# « base » est le modèle par défaut ; « small » et « tiny » restent
# sélectionnables dans l'interface, autant qu'ils soient prêts aussi.
RUN python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('base'); whisper.load_model('small')"

# Le VAD silero est téléchargé depuis GitHub par whisper-timestamped.
# On tente de le mettre en cache, sans bloquer la construction si GitHub
# est indisponible : le code sait retomber sur une transcription sans VAD.
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)" || true

# --- Code de l'application --------------------------------------------------
COPY --chown=user . .

ENV GRADIO_SERVER_NAME=0.0.0.0

# 7860 est le port attendu par Hugging Face Spaces. Railway, Render et Fly.io
# imposent le leur via la variable PORT, que app.py lit en priorité.
EXPOSE 7860

CMD ["python", "app.py"]
