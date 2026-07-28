"""Montage vidéo avec ffmpeg : découpage en plans, assemblage, incrustation.

Deux approches sont implémentées, et le choix entre les deux est un vrai
arbitrage mesuré (voir scripts/benchmark_montage.py et la section
« Montage : pourquoi une seule passe » du README) :

  Approche A — deux passes : on normalise chaque clip en 1080x1920 séparément,
    on concatène en `-c copy` (rapide, aucun réencodage), puis une passe finale
    incruste les sous-titres. Piège : cette dernière passe réencode TOUT, donc
    au total chaque image est encodée deux fois. Le `-c copy` du milieu ne
    rattrape pas ça.

  Approche B — une seule passe : un `filter_complex` unique fait scale + crop +
    concat + sous-titres. Chaque image n'est encodée qu'une fois.

Le code appelle ffmpeg avec `cwd` positionné sur le dossier de travail et des
noms de fichiers relatifs. Ce n'est pas un détail de style : le filtre
`subtitles` de ffmpeg interprète `:` comme un séparateur d'options, donc un
chemin Windows comme `C:\\Users\\...` le casse. Un nom relatif évite le problème.
"""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from modules import config


class EditError(Exception):
    """Erreur de montage, message affichable dans l'interface."""


# Filtre appliqué à chaque plan : on agrandit jusqu'à couvrir le cadre vertical,
# puis on recadre au centre. force_original_aspect_ratio=increase garantit qu'il
# n'y a jamais de bandes noires.
def _scale_crop(
    largeur: int | None = None,
    hauteur: int | None = None,
    index: int = 0,
    duree: float = 0.0,
) -> str:
    """Filtre de mise au format, à la résolution demandée.

    Calculé à la demande et non figé une fois pour toutes : en cas de manque
    de mémoire, le montage est relancé à une résolution plus basse.

    Quand le zoom est actif, un mouvement de caméra lent est ajouté — zoom
    avant sur les plans pairs, arrière sur les impairs. Alterner évite l'effet
    « tout respire en même temps » qu'on obtient avec un zoom toujours
    identique, et donne l'impression d'un montage fait à la main.

    Le zoom est obtenu en agrandissant l'image puis en recadrant sur une zone
    qui varie avec le temps. C'est bien moins coûteux que le filtre `zoompan`,
    qui recalcule l'image entière à chaque trame.
    """
    largeur = largeur or config.VIDEO_WIDTH
    hauteur = hauteur or config.VIDEO_HEIGHT

    base = (
        f"scale={largeur}:{hauteur}:force_original_aspect_ratio=increase,"
        f"crop={largeur}:{hauteur}"
    )

    if config.ZOOM_ACTIF and duree > 0.4:
        marge = config.ZOOM_AMPLITUDE
        grande_l, grande_h = int(largeur * marge) // 2 * 2, int(hauteur * marge) // 2 * 2

        # `t` est le temps écoulé dans le plan. La progression va de 0 à 1,
        # dans un sens ou dans l'autre selon la parité du plan.
        progression = f"min(t/{duree:.2f},1)" if index % 2 == 0 else f"max(1-t/{duree:.2f},0)"
        decalage_x = f"(iw-ow)/2*{progression}"
        decalage_y = f"(ih-oh)/2*{progression}"

        base = (
            f"scale={grande_l}:{grande_h}:force_original_aspect_ratio=increase,"
            f"crop={largeur}:{hauteur}:x='{decalage_x}':y='{decalage_y}'"
        )

    return f"{base},fps={config.VIDEO_FPS},setsar=1,format=yuv420p"


# --- Découpage de la timeline en plans --------------------------------------


def plan_shots(segments: list[dict], total_duration: float) -> list[dict]:
    """Transforme les segments de paroles en une liste de plans à filmer.

    Un plan = un morceau de timeline avec une seule vidéo de fond.
    Trois règles :
      - un plan dure au moins MIN_SHOT_SECONDS (sinon ça clignote) ;
      - un plan dure au plus MAX_SHOT_SECONDS (sinon ça s'endort), un segment
        de paroles trop long est donc coupé en plusieurs plans ;
      - toute la timeline est couverte, y compris l'intro avant le premier mot
        et les trous entre deux phrases.
    """
    if not segments:
        return [{"start": 0.0, "end": total_duration, "query": "", "keyword": ""}]

    shots: list[dict] = []
    cursor = 0.0

    for index, segment in enumerate(segments):
        # Le plan s'étend jusqu'au début du segment suivant : pas de trou.
        next_start = (
            segments[index + 1]["start"] if index + 1 < len(segments) else total_duration
        )
        end = max(segment["end"], next_start)
        start = cursor

        length = end - start
        if length <= 0:
            continue

        # Segment trop long : on le coupe en plusieurs plans de durée égale.
        pieces = max(1, int(length // config.MAX_SHOT_SECONDS) + (1 if length % config.MAX_SHOT_SECONDS > 0.5 else 0))
        piece_length = length / pieces

        for piece in range(pieces):
            shots.append(
                {
                    "start": start + piece * piece_length,
                    "end": start + (piece + 1) * piece_length,
                    "query": segment.get("query", ""),
                    "keyword": segment.get("keyword", ""),
                }
            )
        cursor = end

    if cursor < total_duration - 0.2:
        shots.append(
            {
                "start": cursor,
                "end": total_duration,
                "query": shots[-1]["query"] if shots else "",
                "keyword": "",
            }
        )

    return _merge_short_shots(shots)


def _merge_short_shots(shots: list[dict]) -> list[dict]:
    """Fusionne les plans trop courts avec le précédent."""
    merged: list[dict] = []
    for shot in shots:
        duration = shot["end"] - shot["start"]
        if merged and duration < config.MIN_SHOT_SECONDS:
            merged[-1]["end"] = shot["end"]
            # On garde la requête du plan le plus long des deux.
            if not merged[-1]["query"]:
                merged[-1]["query"] = shot["query"]
        else:
            merged.append(dict(shot))
    return merged


# --- Appel de ffmpeg --------------------------------------------------------


def _run_ffmpeg(arguments: list[str], cwd: Path, timeout: int = 1800) -> None:
    """Lance ffmpeg et lève une EditError lisible si ça échoue."""
    command = [config.ffmpeg_path(), "-hide_banner", "-y", "-v", "error", *arguments]
    try:
        result = subprocess.run(
            command, cwd=str(cwd), capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise EditError("Le montage a dépassé le temps maximum autorisé.")
    except OSError as error:
        raise EditError(f"Impossible de lancer ffmpeg : {error}")

    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()

        # Un code négatif signifie que ffmpeg a été TUÉ par un signal, pas
        # qu'il s'est arrêté sur une erreur. Dans ce cas stderr est vide, et
        # afficher « ffmpeg a échoué » sans rien d'autre n'aide personne.
        # -9 = SIGKILL, envoyé par le système quand la mémoire manque.
        if result.returncode < 0 or (not message and result.returncode in (137, 143)):
            raise EditError(
                "Le montage a été interrompu par le système, faute de mémoire "
                f"(signal {abs(result.returncode)}).\n"
                "Réessaie avec un extrait plus court, ou avec un hébergement "
                "disposant de plus de mémoire."
            )

        if not message:
            raise EditError(
                f"ffmpeg s'est arrêté sans explication (code {result.returncode}). "
                "C'est presque toujours un manque de mémoire ou d'espace disque."
            )

        # On ne renvoie que les dernières lignes : ffmpeg est très bavard.
        tail = "\n".join(message.splitlines()[-6:])
        raise EditError(f"ffmpeg a échoué :\n{tail}")


def _input_arguments(shot: dict) -> list[str]:
    """Arguments d'entrée pour un plan : découpe dans le clip source.

    `-ss` placé AVANT `-i` fait un seek rapide (ffmpeg saute directement au
    bon endroit au lieu de décoder tout ce qui précède).
    Si le clip est plus court que le plan, on le boucle.
    """
    needed = shot["end"] - shot["start"]
    clip_duration = shot.get("clip_duration", 0.0)

    if clip_duration and clip_duration < needed + 0.2:
        return ["-stream_loop", "-1", "-t", f"{needed:.3f}", "-i", shot["clip_name"]]

    offset = shot.get("clip_offset", 0.0)
    return ["-ss", f"{offset:.3f}", "-t", f"{needed:.3f}", "-i", shot["clip_name"]]


# --- Approche B : une seule passe (utilisée par défaut) ---------------------


def render_single_pass(
    shots: list[dict],
    audio_name: str,
    subtitle_name: str,
    output_name: str,
    work_dir: Path,
    progress: Callable[[str], None] | None = None,
    inclure_audio: bool = True,
    largeur: int | None = None,
    hauteur: int | None = None,
) -> Path:
    """Monte la vidéo complète en une seule commande ffmpeg.

    Chaque plan doit contenir : clip_name, clip_offset, clip_duration, start, end.
    Tous les noms de fichiers sont RELATIFS à `work_dir`.

    `inclure_audio=False` produit une vidéo MUETTE, images et sous-titres
    seulement. C'est le format attendu par les monteurs d'edits : on ajoute
    ensuite la musique depuis la bibliothèque de TikTok ou d'Instagram, qui est
    sous licence — le son n'est alors ni coupé ni bloqué, contrairement à un
    enregistrement intégré au fichier.
    """
    if not shots:
        raise EditError("Aucun plan à monter.")

    if progress:
        muet = " (sans musique)" if not inclure_audio else ""
        progress(f"Montage de {len(shots)} plans en une passe{muet}…")

    arguments: list[str] = []
    for shot in shots:
        arguments += _input_arguments(shot)

    audio_index = len(shots)
    if inclure_audio:
        arguments += ["-i", audio_name]

    # Un filtre scale+crop par plan, puis un concat, puis les sous-titres.
    filters = []
    for index in range(len(shots)):
        duree = shots[index]["end"] - shots[index]["start"]
        filters.append(
            f"[{index}:v]{_scale_crop(largeur, hauteur, index, duree)}[v{index}]"
        )

    concat_inputs = "".join(f"[v{index}]" for index in range(len(shots)))
    filters.append(f"{concat_inputs}concat=n={len(shots)}:v=1:a=0[cat]")
    filters.append(f"[cat]subtitles={subtitle_name}[vout]")

    encoder = config.detect_video_encoder()
    arguments += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
    ]

    if inclure_audio:
        arguments += [
            "-map", f"{audio_index}:a",
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        arguments += ["-an"]

    arguments += [
        "-c:v", encoder,
        *config.encoder_options(encoder),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    # -shortest cale la vidéo sur la piste la plus courte. Sans piste audio,
    # l'option n'a aucun sens et ffmpeg s'en plaint.
    if inclure_audio:
        arguments += ["-shortest"]

    arguments += [output_name]

    _run_ffmpeg(arguments, cwd=work_dir)
    return work_dir / output_name


# --- Approche A : deux passes (gardée pour la comparaison) ------------------


def render_two_pass(
    shots: list[dict],
    audio_name: str,
    subtitle_name: str,
    output_name: str,
    work_dir: Path,
    progress: Callable[[str], None] | None = None,
    inclure_audio: bool = True,
    largeur: int | None = None,
    hauteur: int | None = None,
) -> Path:
    """Normalise chaque plan, concatène en copie, puis incruste les sous-titres.

    Plus lent que le montage en une passe, mais son pic de mémoire ne dépend
    pas du nombre de plans : c'est la seule stratégie tenable sur un conteneur
    limité (voir choisir_approche).
    """
    if not shots:
        raise EditError("Aucun plan à monter.")

    encoder = config.detect_video_encoder()

    if progress:
        progress(f"Normalisation de {len(shots)} plans…")

    def normalise(index_and_shot: tuple[int, dict]) -> str:
        index, shot = index_and_shot
        output = f"norm_{index:03d}.mp4"
        arguments = [
            *_input_arguments(shot),
            "-an",
            "-vf", _scale_crop(largeur, hauteur, index,
                               shot["end"] - shot["start"]),
            "-c:v", encoder,
            *config.encoder_options(encoder),
            output,
        ]
        _run_ffmpeg(arguments, cwd=work_dir)
        return output

    # Les plans sont indépendants, donc parallélisables — mais chaque ffmpeg
    # lancé consomme sa propre mémoire. Sur un conteneur limité, paralléliser
    # annulerait tout le bénéfice de cette approche.
    memoire = config.memoire_disponible_mo()
    if memoire is not None and memoire < MEMOIRE_MINIMALE_UNE_PASSE_MO:
        workers = 1
    else:
        workers = min(4, (os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        normalised = list(pool.map(normalise, enumerate(shots)))

    if progress:
        progress("Assemblage des plans…")

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{name}'\n" for name in normalised), encoding="utf-8"
    )
    _run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "concat.mp4"],
        cwd=work_dir,
    )

    if progress:
        progress("Incrustation des paroles…")

    arguments = ["-i", "concat.mp4"]
    if inclure_audio:
        arguments += ["-i", audio_name]

    arguments += [
        "-vf", f"subtitles={subtitle_name}",
        "-map", "0:v",
    ]
    if inclure_audio:
        arguments += ["-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    else:
        arguments += ["-an"]

    arguments += [
        "-c:v", encoder,
        *config.encoder_options(encoder),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if inclure_audio:
        arguments += ["-shortest"]
    arguments += [output_name]

    _run_ffmpeg(arguments, cwd=work_dir)
    return work_dir / output_name


# En dessous de cette mémoire, le montage en une passe ne tient pas : il ouvre
# tous les clips simultanément dans un même filter_complex.
MEMOIRE_MINIMALE_UNE_PASSE_MO = 1500


def choisir_approche(nombre_de_plans: int) -> tuple[str, str | None]:
    """Choisit la stratégie de montage selon la mémoire disponible.

    Le montage en une passe est le plus rapide (mesuré : 42 % plus rapide),
    mais il décode tous les clips en parallèle dans un seul filter_complex.
    Sur un conteneur limité, ffmpeg se fait tuer par le système — sans message
    d'erreur, puisqu'il ne s'arrête pas de lui-même.

    Le montage en deux passes traite les clips un par un : bien plus lent,
    mais son pic de mémoire ne dépend pas du nombre de plans.

    Renvoie (approche, explication ou None).
    """
    disponible = config.memoire_disponible_mo()
    if disponible is None or disponible >= MEMOIRE_MINIMALE_UNE_PASSE_MO:
        return "single", None

    return "two_pass", (
        f"Montage clip par clip : {disponible:.0f} Mo de mémoire disponibles, "
        f"insuffisant pour assembler {nombre_de_plans} plans en une seule passe."
    )


def render(
    shots: list[dict],
    audio_name: str,
    subtitle_name: str,
    output_name: str,
    work_dir: Path,
    approach: str = "single",
    progress: Callable[[str], None] | None = None,
    inclure_audio: bool = True,
    largeur: int | None = None,
    hauteur: int | None = None,
) -> Path:
    """Point d'entrée du montage. `approach` vaut "single" (B) ou "two_pass" (A).

    L'appelant peut passer "auto" pour laisser le choix se faire selon la
    mémoire réellement disponible.
    """
    if approach == "auto":
        approach, explication = choisir_approche(len(shots))
        if explication and progress:
            progress(explication)

    # Résolutions à tenter, de la meilleure à la plus modeste. Si le système
    # tue ffmpeg faute de mémoire, on recommence plus petit plutôt que de
    # renvoyer une erreur : un clip un peu moins défini vaut mieux que rien.
    resolutions = [(config.VIDEO_WIDTH, config.VIDEO_HEIGHT)]
    for reduction in (0.75, 0.5):
        largeur = int(config.VIDEO_WIDTH * reduction) // 2 * 2
        hauteur = int(config.VIDEO_HEIGHT * reduction) // 2 * 2
        if hauteur >= 640:
            resolutions.append((largeur, hauteur))

    derniere_erreur: EditError | None = None
    for essai, (largeur, hauteur) in enumerate(resolutions):
        if essai and progress:
            progress(
                f"Mémoire insuffisante : nouvelle tentative en {largeur}×{hauteur}…"
            )
        try:
            fonction = render_two_pass if approach == "two_pass" else render_single_pass
            return fonction(
                shots, audio_name, subtitle_name, output_name, work_dir, progress,
                inclure_audio=inclure_audio, largeur=largeur, hauteur=hauteur,
            )
        except EditError as erreur:
            derniere_erreur = erreur
            # On ne réessaie que sur un manque de mémoire : une erreur de
            # fichier ou de filtre se reproduira à l'identique.
            if "mémoire" not in str(erreur).lower():
                raise
            # Le montage précédent a laissé des fichiers à moitié écrits.
            for reste in work_dir.glob("norm_*.mp4"):
                reste.unlink(missing_ok=True)
            (work_dir / "concat.mp4").unlink(missing_ok=True)

    raise derniere_erreur or EditError("Le montage a échoué.")


def extract_audio_segment(
    source: Path, work_dir: Path, output_name: str, start: float, duration: float
) -> Path:
    """Extrait la portion d'audio correspondant à l'extrait retenu.

    Réencodé en AAC : couper un MP3 en copie brute décale souvent le son de
    quelques dizaines de millisecondes, ce qui suffit à désynchroniser le karaoké.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(Path(source).resolve()),
            "-vn",
            "-c:a", "aac", "-b:a", "192k",
            output_name,
        ],
        cwd=work_dir,
    )
    return work_dir / output_name


def cleanup(work_dir: Path) -> None:
    """Supprime le dossier de travail. Ne lève jamais d'exception.

    Appelé dans un `finally` : si le nettoyage échoue, ce n'est pas une raison
    de masquer l'erreur d'origine.
    """
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass
