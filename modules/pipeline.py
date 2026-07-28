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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from modules import audio, config, editor, keywords, subtitles, transcribe, videos

# Nombre de vidéos finales conservées sur le disque du Space.
# Au-delà, les plus anciennes sont supprimées.
KEEP_LAST_OUTPUTS = 5

# En dessous de cette mémoire, on vide le modèle de transcription avant de
# lancer le montage. Au-dessus, on le garde en cache : le recharger coûte
# quelques secondes à chaque traitement.
SEUIL_LIBERATION_MEMOIRE_MO = 2000


class PipelineError(Exception):
    """Erreur de pipeline, message affichable tel quel dans l'interface."""


def generate_clip(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = None,
    approach: str = "auto",
    progress: Callable[[str], None] | None = None,
    inclure_audio: bool = True,
    transcription_prete: dict | None = None,
    clips_a_eviter: set[str] | None = None,
    rang_passage: int = 0,
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
        # Une transcription déjà calculée (paroles corrigées à la main, ou
        # simple relance pour changer les visuels) évite de refaire l'étape la
        # plus longue du pipeline.
        if transcription_prete is not None:
            step("Reprise des paroles corrigées…")
            result = transcription_prete
        else:
            result = transcribe.transcribe_audio(
                audio_path, model_name=model_name, language=language,
                progress=step, rang_passage=rang_passage,
            )
        segments = result["segments"]
        detected_language = result["language"]

        instrumental = bool(result.get("instrumental"))

        # --- 3. Mots-clés visuels -------------------------------------------
        if instrumental:
            step("Morceau instrumental : montage d'ambiance…")
            segments = []
            shots = _plans_instrumentaux(result)
        else:
            step("Analyse des paroles et choix des images…")
            segments = keywords.build_queries(segments, detected_language)

            # --- 4. Découpage en plans --------------------------------------
            shots = editor.plan_shots(segments, result["duration"])

        # Les coupes glissent vers le temps fort le plus proche : c'est ce qui
        # fait qu'un montage « tombe juste » à l'oreille.
        temps_forts = result.get("temps_forts") or []
        if temps_forts:
            from modules import rythme

            avant = [s["start"] for s in shots]
            shots = rythme.caler_sur_temps_forts(shots, temps_forts)
            deplacees = sum(
                1 for a, s in zip(avant, shots) if abs(a - s["start"]) > 0.01
            )
            if deplacees:
                step(f"{deplacees} coupes calées sur le rythme…")
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

        # Sur une régénération, on écarte les clips déjà vus : l'utilisateur
        # qui reclique veut d'autres images, pas les mêmes.
        if clips_a_eviter:
            finder.used_ids.update(clips_a_eviter)

        # La sélection reste séquentielle — elle est quasi instantanée grâce au
        # cache, et l'anti-répétition a besoin d'un ordre déterminé. Seuls les
        # téléchargements partent en parallèle : ce sont eux qui attendent le
        # réseau, et ils sont indépendants les uns des autres.
        selection = []
        for shot in shots:
            candidate = finder.find(shot["query"] or "abstract background")
            if candidate is None:
                warnings.append(f"Aucune vidéo trouvée pour « {shot['query']} ».")
                continue
            selection.append((shot, candidate))

        step(f"Téléchargement de {len(selection)} clips…")

        def recuperer(paire):
            shot, candidate = paire
            chemin = finder.download(candidate)
            return shot, candidate, chemin

        with ThreadPoolExecutor(max_workers=min(6, max(len(selection), 1))) as pool:
            telecharges = list(pool.map(recuperer, selection))

        usable_shots = []
        for shot, candidate, clip_path in telecharges:
            if clip_path is None:
                warnings.append(f"Téléchargement échoué pour « {shot['query']} ».")
                continue

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

        # --- 5 bis. Libération de la mémoire --------------------------------
        # Le modèle de transcription a fini son travail, mais il occupe encore
        # plusieurs centaines de mégaoctets. Or c'est maintenant que ffmpeg va
        # décoder des vidéos, ce qui est l'étape la plus gourmande. Sur un
        # conteneur limité, garder le modèle en cache faisait tuer le montage.
        memoire = config.memoire_disponible_mo()
        if memoire is not None and memoire < SEUIL_LIBERATION_MEMOIRE_MO:
            libere = transcribe.decharger_modeles()
            if libere:
                step(f"Libération de la mémoire ({libere:.0f} Mo) avant le montage…")

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
            inclure_audio=inclure_audio,
        )

        # La vidéo sort du dossier de travail avant que celui-ci soit effacé.
        final_path = config.OUTPUT_DIR / output_name
        shutil.move(str(rendered), str(final_path))
        _prune_old_outputs()

        lyrics = "\n".join(segment["text"] for segment in segments)
        if instrumental:
            lyrics = (
                "Morceau instrumental : aucune parole à afficher.\n"
                "Le montage suit le rythme de la musique."
            )

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
            "muet": not inclure_audio,
            "instrumental": instrumental,
            # Où se situe l'extrait dans la chanson, et combien d'autres
            # passages sont disponibles : l'interface s'en sert pour proposer
            # d'en essayer un autre.
            "debut_dans_morceau": result["start_offset"],
            "passages_disponibles": result.get("passages_disponibles", 1),
            "passage_retenu": result.get("passage_retenu", 0),
            # Renvoyée telle quelle pour permettre une régénération sans
            # repasser par Whisper (correction des paroles, autres visuels).
            "transcription": result,
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


def _plans_instrumentaux(result: dict) -> list[dict]:
    """Découpe un morceau sans paroles en plans, calés sur ses temps forts.

    Sans texte, il n'y a plus de phrase pour rythmer le montage : ce sont les
    temps forts qui décident. On coupe donc sur eux, en gardant des plans
    d'une durée regardable — trop courts ils clignotent, trop longs ils
    endorment.
    """
    duree = result["duration"]
    temps_forts = result.get("temps_forts") or []

    cible = (config.MIN_SHOT_SECONDS + config.MAX_SHOT_SECONDS) / 2

    frontieres = [0.0]
    if temps_forts:
        for instant in temps_forts:
            if instant - frontieres[-1] >= cible and duree - instant >= config.MIN_SHOT_SECONDS:
                frontieres.append(instant)
    else:
        # Aucun temps fort détecté : découpage régulier, faute de mieux.
        instant = cible
        while duree - instant >= config.MIN_SHOT_SECONDS:
            frontieres.append(instant)
            instant += cible

    frontieres.append(duree)

    requetes = keywords.requetes_instrumentales(len(frontieres) - 1)
    return [
        {
            "start": frontieres[i],
            "end": frontieres[i + 1],
            "query": requetes[i],
            "keyword": "",
        }
        for i in range(len(frontieres) - 1)
    ]


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


# Un morceau téléchargé doit survivre à la génération : l'utilisateur peut
# vouloir corriger les paroles et refaire la vidéo sans tout retélécharger.
# On le garde donc une heure, puis on nettoie.
DUREE_VIE_AUDIO_TEMPORAIRE = 3600


def purger_audios_temporaires() -> None:
    """Supprime les morceaux téléchargés qui ne servent plus.

    Appelée à chaque génération : sans elle, le disque d'un Space gratuit se
    remplit d'un MP3 par recherche.
    """
    import time as _time

    try:
        maintenant = _time.time()
        for fichier in config.WORK_DIR.glob("jamendo_*.mp3"):
            if maintenant - fichier.stat().st_mtime > DUREE_VIE_AUDIO_TEMPORAIRE:
                fichier.unlink(missing_ok=True)
    except OSError:
        pass
