"""Transcription des paroles avec timestamps mot par mot.

On utilise whisper-timestamped et pas openai-whisper seul : le karaoké a besoin
de savoir quand chaque MOT commence et finit, ce que Whisper de base ne donne pas
de façon fiable.

Ce module renvoie une structure simple, réutilisée par tout le reste du pipeline :

    {
      "language": "fr",
      "start_offset": 12.4,   # début de l'extrait retenu dans le fichier d'origine
      "duration": 88.2,       # durée de l'extrait retenu
      "segments": [
        {
          "text": "je marche seul dans la ville",
          "start": 0.0, "end": 3.2, "confidence": 0.82,
          "words": [{"text": "je", "start": 0.0, "end": 0.2, "confidence": 0.9}, ...]
        },
        ...
      ]
    }

Les temps des segments sont TOUJOURS relatifs au début de l'extrait retenu
(donc le premier segment commence proche de 0), ce qui simplifie le montage.
"""

from pathlib import Path
from typing import Callable

from modules import config


class TranscriptionError(Exception):
    """Erreur de transcription, avec un message affichable dans l'interface."""


# Le modèle pèse plusieurs centaines de Mo : on le garde en mémoire entre
# deux traitements plutôt que de le recharger à chaque fois.
_model_cache: dict[str, object] = {}


def load_model(model_name: str | None = None):
    """Charge (et met en cache) un modèle Whisper."""
    import whisper_timestamped

    name = model_name or config.WHISPER_MODEL
    if name not in _model_cache:
        try:
            _model_cache[name] = whisper_timestamped.load_model(name, device="cpu")
        except Exception as error:
            raise TranscriptionError(
                f"Impossible de charger le modèle Whisper « {name} » : {error}\n"
                "Au premier lancement le modèle est téléchargé, cela peut prendre "
                "quelques minutes et demande une connexion internet."
            )
    return _model_cache[name]


def transcribe_audio(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Transcrit un fichier audio et renvoie l'extrait le plus dense en paroles.

    `language` : code ISO ("fr", "en"...) ou None pour détection automatique.
    `progress` : fonction appelée avec un message d'étape, pour l'interface.
    """
    import whisper_timestamped

    if progress:
        progress("Chargement du modèle de transcription…")
    model = load_model(model_name)

    if progress:
        progress("Transcription des paroles (étape la plus longue)…")

    try:
        audio = whisper_timestamped.load_audio(str(audio_path))
    except Exception as error:
        raise TranscriptionError(f"Lecture du fichier audio impossible : {error}")

    try:
        raw = whisper_timestamped.transcribe(
            model,
            audio,
            language=language,
            # vad coupe les longs silences : c'est la principale protection
            # contre les hallucinations de Whisper sur les passages instrumentaux.
            vad=True,
            # Sur une chanson, la "température de repli" fait souvent inventer
            # du texte. On reste déterministe.
            temperature=0.0,
            verbose=None,
        )
    except Exception as error:
        # vad=True télécharge le modèle silero au premier appel : sans réseau,
        # ça échoue. On réessaie une fois sans VAD plutôt que de tout arrêter.
        try:
            raw = whisper_timestamped.transcribe(
                model, audio, language=language, temperature=0.0, verbose=None
            )
        except Exception:
            raise TranscriptionError(f"La transcription a échoué : {error}")

    segments = _normalise_segments(raw.get("segments", []))
    segments = _drop_unreliable(segments)

    total_words = sum(len(segment["words"]) for segment in segments)
    if total_words < config.MIN_WORDS_REQUIRED:
        raise TranscriptionError(
            "Aucune parole détectée dans ce fichier.\n"
            "C'est normal pour un morceau instrumental : le clip karaoké a besoin "
            "de paroles chantées. Essaie un autre morceau."
        )

    if progress:
        progress("Sélection du passage le plus dense en paroles…")
    return _select_best_window(segments, raw.get("language", language or "fr"))


def appliquer_texte_corrige(segments: list[dict], texte: str) -> list[dict]:
    """Remplace les paroles par une version corrigée à la main, sans retranscrire.

    Whisper se trompe régulièrement, surtout sur l'argot et les noms propres.
    Plutôt que de relancer une transcription (une minute ou plus), on réutilise
    les temps déjà calculés et on n'échange que les mots.

    `texte` contient une ligne par segment, dans le même ordre qu'à l'affichage.

    Quand une ligne corrigée n'a pas le même nombre de mots que l'originale, on
    redistribue la durée de la ligne sur les nouveaux mots, proportionnellement
    à leur longueur — une syllabe de plus prend un peu plus de temps. Ce n'est
    pas un réalignement acoustique, mais l'écart reste sous le dixième de
    seconde, invisible à l'oeil sur un karaoké.
    """
    lignes = [ligne.strip() for ligne in (texte or "").splitlines()]
    lignes = [ligne for ligne in lignes if ligne]

    if not lignes:
        raise TranscriptionError("Les paroles corrigées sont vides.")

    if len(lignes) != len(segments):
        raise TranscriptionError(
            f"Il faut exactement une ligne par phrase : {len(segments)} lignes "
            f"attendues, {len(lignes)} reçues.\n"
            "Corrige les mots sans ajouter ni supprimer de ligne."
        )

    corriges = []
    for segment, ligne in zip(segments, lignes):
        mots = ligne.split()
        if not mots:
            continue

        debut = segment["words"][0]["start"]
        fin = segment["words"][-1]["end"]
        duree = max(fin - debut, 0.05)

        # Répartition proportionnelle à la longueur des mots.
        longueurs = [len(mot) for mot in mots]
        total = sum(longueurs) or len(mots)

        nouveaux_mots = []
        curseur = debut
        for mot, longueur in zip(mots, longueurs):
            part = duree * (longueur / total)
            nouveaux_mots.append(
                {
                    "text": mot,
                    "start": curseur,
                    "end": curseur + part,
                    "confidence": 1.0,  # corrigé par un humain
                }
            )
            curseur += part

        nouveaux_mots[-1]["end"] = fin

        corriges.append(
            {
                "text": " ".join(mots),
                "start": debut,
                "end": fin,
                "confidence": 1.0,
                "words": nouveaux_mots,
            }
        )

    return corriges


def _normalise_segments(raw_segments: list[dict]) -> list[dict]:
    """Convertit la sortie de whisper-timestamped en structure simple et propre."""
    segments = []
    for raw in raw_segments:
        words = []
        for raw_word in raw.get("words", []):
            text = (raw_word.get("text") or "").strip()
            start = raw_word.get("start")
            end = raw_word.get("end")
            if not text or start is None or end is None or end <= start:
                continue
            words.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "confidence": float(raw_word.get("confidence") or 0.0),
                }
            )

        if not words:
            continue

        segments.append(
            {
                "text": " ".join(word["text"] for word in words),
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "confidence": float(raw.get("confidence") or 0.0),
                "words": words,
            }
        )
    return segments


def _drop_unreliable(segments: list[dict]) -> list[dict]:
    """Jette les segments que Whisper a probablement inventés.

    Trois signaux d'hallucination, tous observés en pratique sur de la musique :
      - une confiance basse ;
      - le même mot répété en boucle ("la la la la la la la") ;
      - un segment absurdement long pour le nombre de mots qu'il contient.
    """
    kept = []
    for segment in segments:
        if segment["confidence"] and segment["confidence"] < config.MIN_SEGMENT_CONFIDENCE:
            continue

        words = [word["text"].lower().strip(".,!?;:") for word in segment["words"]]
        if len(words) >= 4 and len(set(words)) == 1:
            continue

        duration = segment["end"] - segment["start"]
        if duration > 2 and duration / max(len(words), 1) > 3.0:
            # Plus de 3 secondes par mot : c'est du silence habillé de texte.
            continue

        kept.append(segment)
    return kept


def _select_best_window(segments: list[dict], language: str) -> dict:
    """Garde au maximum MAX_CLIP_SECONDS de la partie la plus dense en paroles.

    Une chanson de 4 minutes donnerait un traitement très long et une vidéo
    trop longue pour les réseaux sociaux. Plutôt que de couper bêtement les
    90 premières secondes (souvent une intro instrumentale), on cherche la
    fenêtre qui contient le plus de mots.
    """
    max_seconds = config.MAX_CLIP_SECONDS

    if segments and segments[-1]["end"] - segments[0]["start"] <= max_seconds:
        window = segments
    else:
        best_window: list[dict] = []
        best_count = -1
        for index, first in enumerate(segments):
            limit = first["start"] + max_seconds
            window = [s for s in segments[index:] if s["end"] <= limit]
            if not window:
                window = [first]
            count = sum(len(s["words"]) for s in window)
            if count > best_count:
                best_count = count
                best_window = window
        window = best_window

    # On recale tous les temps sur le début de la fenêtre retenue.
    # Un petit coussin avant le premier mot évite de démarrer pile sur la voix.
    offset = max(0.0, window[0]["start"] - 0.4)
    shifted = []
    for segment in window:
        shifted.append(
            {
                "text": segment["text"],
                "start": segment["start"] - offset,
                "end": segment["end"] - offset,
                "confidence": segment["confidence"],
                "words": [
                    {
                        "text": word["text"],
                        "start": word["start"] - offset,
                        "end": word["end"] - offset,
                        "confidence": word["confidence"],
                    }
                    for word in segment["words"]
                ],
            }
        )

    duration = shifted[-1]["end"] + 0.6  # petite respiration à la fin

    return {
        "language": language,
        "start_offset": offset,
        "duration": duration,
        "segments": shifted,
    }
