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

# Plus de PyTorch : faster-whisper s'appuie sur CTranslate2 (C++), sans CUDA ni
# torch. L'image passe d'environ 4 Go à moins de 1 Go, et surtout l'empreinte
# mémoire à l'exécution tombe de ~1900 Mo à ~390 Mo — ce qui permet enfin de
# tourner sur un hébergement gratuit.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Modèles spaCy (~15 Mo chacun) : français et anglais.
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

# Les modèles se téléchargent au premier appel. On les met dans l'image :
# sans ça, la toute première génération attend plusieurs minutes avant même
# de commencer — mauvais effet garanti en démo.
# « base » est le modèle par défaut ; « tiny » et « small » restent
# sélectionnables dans l'interface, autant qu'ils soient prêts aussi.
RUN python -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('tiny', 'base', 'small')]"

# --- Code de l'application --------------------------------------------------
COPY --chown=user . .

ENV GRADIO_SERVER_NAME=0.0.0.0

# 7860 est le port attendu par Hugging Face Spaces. Railway, Render et Fly.io
# imposent le leur via la variable PORT, que app.py lit en priorité.
EXPOSE 7860

CMD ["python", "app.py"]
