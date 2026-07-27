"""Orchestration : de l'audio en entrée à la vidéo verticale en sortie.

C'est le seul module qui connaît l'enchaînement complet. Les autres ne savent
faire qu'une chose chacun, ce qui permet de les tester séparément (voir scripts/).

Deux garanties tenues ici :
  - le dossier de travail est TOUJOURS nettoyé, y compris quand une étape
    échoue (bloc `finally`) — sans ça un Space gratuit sature en quelques essais ;
  - toute erreur remonte avec un message lisible par un humain, jamais une trace
    Python brute.
"""

import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from modules import audio, config, editor, keywords, subtitles, transcribe, videos

# Nombre de vidéos finales conservées sur le disque du Space.
# Au-delà, les plus anciennes sont supprimées.
KEEP_LAST_OUTPUTS = 5


class PipelineError(Exception):
    """Erreur de pipeline, message affichable tel quel dans l'interface."""


def generate_clip(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = None,
    approach: str = "single",
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Fabrique le clip complet et renvoie un dictionnaire de résultat.

    Résultat :
      {
        "video": Path,          # la vidéo finale
        "lyrics": str,          # les paroles transcrites, pour affichage
        "credits": [str],       # auteurs des vidéos de fond
        "shots": int,           # nombre de plans
        "duration": float,      # durée de la vidéo
        "warnings": [str],      # ce qui s'est mal passé sans être bloquant
        "seconds": float,       # temps de traitement
      }
    """
    started_at = time.time()

    def step(message: str) -> None:
        if progress:
            progress(message)

    config.ensure_dirs()
    job_id = uuid.uuid4().hex[:10]
    work_dir = config.WORK_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    try:
        # --- 1. Validation de l'audio ---------------------------------------
        step("Vérification du fichier audio…")
        audio_path = Path(audio_path)
        audio.validate_audio(audio_path)

        # --- 2. Transcription alignée ---------------------------------------
        result = transcribe.transcribe_audio(
            audio_path, model_name=model_name, language=language, progress=step
        )
        segments = result["segments"]
        detected_language = result["language"]

        # --- 3. Mots-clés visuels -------------------------------------------
        step("Analyse des paroles et choix des images…")
        segments = keywords.build_queries(segments, detected_language)

        # --- 4. Découpage en plans ------------------------------------------
        shots = editor.plan_shots(segments, result["duration"])
        if not shots:
            raise PipelineError("Impossible de construire un montage à partir de ces paroles.")

        # --- 5. Vidéos de fond ----------------------------------------------
        # Sans clé API, on ne bloque pas : Wikimedia Commons sert de filet.
        # Mais on prévient, parce que le rendu s'en ressent nettement.
        if not videos.has_any_key():
            warnings.append(
                "Aucune clé Pexels/Pixabay : fonds repris de Wikimedia Commons, "
                "de qualité inégale."
            )

        step(f"Recherche de {len(shots)} vidéos de fond…")
        finder = videos.ClipFinder(work_dir)
        usable_shots = []

        for index, shot in enumerate(shots, start=1):
            step(f"Vidéo de fond {index}/{len(shots)}…")
            found = finder.find_and_download(shot["query"] or "abstract background")
            if found is None:
                warnings.append(f"Aucune vidéo trouvée pour « {shot['query']} ».")
                continue

            clip_path, candidate = found
            clip_duration = videos.probe_video_duration(clip_path)
            needed = shot["end"] - shot["start"]

            enriched = dict(shot)
            enriched["clip_name"] = clip_path.name
            enriched["clip_duration"] = clip_duration
            enriched["clip_offset"] = videos.pick_start_offset(clip_duration, needed)
            enriched["credit"] = candidate["credit"]
            enriched["clip_id"] = candidate["id"]
            usable_shots.append(enriched)

        if not usable_shots:
            raise PipelineError(
                "Aucune vidéo de fond n'a pu être récupérée.\n"
                "Vérifie tes clés API et ta connexion, puis réessaie."
            )

        # Un plan manquant laisserait un trou : on répartit son temps sur
        # les plans voisins plutôt que de faire un saut dans la musique.
        usable_shots = _close_gaps(usable_shots, result["duration"])

        # --- 6. Sous-titres karaoké -----------------------------------------
        step("Génération des sous-titres karaoké…")
        subtitle_path = subtitles.build_ass(segments, work_dir / "subs.ass")

        # --- 7. Extrait audio ------------------------------------------------
        step("Préparation de la bande son…")
        editor.extract_audio_segment(
            audio_path,
            work_dir,
            "track.m4a",
            result["start_offset"],
            result["duration"],
        )

        # --- 8. Montage -------------------------------------------------------
        output_name = f"clip_{job_id}.mp4"
        rendered = editor.render(
            usable_shots,
            audio_name="track.m4a",
            subtitle_name=subtitle_path.name,
            output_name=output_name,
            work_dir=work_dir,
            approach=approach,
            progress=step,
        )

        # La vidéo sort du dossier de travail avant que celui-ci soit effacé.
        final_path = config.OUTPUT_DIR / output_name
        shutil.move(str(rendered), str(final_path))
        _prune_old_outputs()

        lyrics = "\n".join(segment["text"] for segment in segments)

        return {
            "video": final_path,
            "lyrics": lyrics,
            "credits": finder.credits,
            # Un identifiant par plan : permet de vérifier l'anti-répétition
            # sans faire confiance aux crédits (Wikimedia renvoie toujours
            # le même auteur pour tous ses clips).
            "clip_ids": [shot["clip_id"] for shot in usable_shots],
            "shots": len(usable_shots),
            "duration": result["duration"],
            "language": detected_language,
            "warnings": warnings,
            "seconds": round(time.time() - started_at, 1),
        }

    except (audio.AudioError, transcribe.TranscriptionError,
            keywords.KeywordError, videos.VideoError, editor.EditError,
            RuntimeError) as error:
        # Erreurs déjà formulées pour un humain (RuntimeError = ffmpeg absent
        # ou sans encodeur) : on les laisse telles quelles.
        raise PipelineError(str(error)) from error

    finally:
        # Nettoyage garanti, même si tout a explosé au milieu.
        editor.cleanup(work_dir)


def _close_gaps(shots: list[dict], total_duration: float) -> list[dict]:
    """Recolle la timeline quand un plan a été abandonné faute de vidéo.

    Les plans étant joués les uns après les autres par ffmpeg, ce qui compte
    est leur DURÉE, pas leur position d'origine. On étire donc les plans
    restants pour retomber sur la durée totale.
    """
    if not shots:
        return shots

    kept_duration = sum(shot["end"] - shot["start"] for shot in shots)
    if kept_duration <= 0:
        return shots

    stretch = total_duration / kept_duration
    if abs(stretch - 1.0) < 0.02:
        return shots

    adjusted = []
    cursor = 0.0
    for shot in shots:
        length = (shot["end"] - shot["start"]) * stretch
        new_shot = dict(shot)
        new_shot["start"] = cursor
        new_shot["end"] = cursor + length
        cursor += length
        adjusted.append(new_shot)
    return adjusted


def _prune_old_outputs() -> None:
    """Garde seulement les dernières vidéos produites, pour ne pas saturer le disque."""
    try:
        files = sorted(
            config.OUTPUT_DIR.glob("clip_*.mp4"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in files[KEEP_LAST_OUTPUTS:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass
