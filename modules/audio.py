"""Entrée audio : validation d'un fichier uploadé, et mode Jamendo (Creative Commons).

Règle de ce module : on valide AVANT de lancer quoi que ce soit de long.
Découvrir qu'un fichier est corrompu après trois minutes de transcription,
c'est trois minutes perdues et un utilisateur perdu aussi.
"""

import json
import subprocess
from pathlib import Path

import requests

from modules import config


class AudioError(Exception):
    """Erreur d'entrée audio, avec un message affichable tel quel dans l'interface."""


# --- Validation d'un fichier local ------------------------------------------


def probe_duration(path: str | Path) -> float:
    """Renvoie la durée du fichier audio en secondes, via ffprobe.

    Lève AudioError si le fichier n'est pas lisible : c'est le vrai test de
    validité, bien plus fiable que de regarder l'extension du fichier.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"Fichier introuvable : {path.name}")

    try:
        ffprobe = config.ffprobe_path()
    except RuntimeError as error:
        # ffmpeg absent de la machine : c'est un message pour l'utilisateur,
        # pas une trace Python.
        raise AudioError(str(error))

    command = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "a:0",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AudioError("Analyse du fichier trop longue : il est probablement corrompu.")
    except OSError as error:
        raise AudioError(f"Impossible de lancer ffprobe : {error}")

    if result.returncode != 0:
        raise AudioError(
            "Ce fichier n'est pas un audio lisible. "
            "Formats acceptés : MP3, WAV, M4A, FLAC, OGG."
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AudioError("Réponse de ffprobe illisible : fichier probablement corrompu.")

    if not data.get("streams"):
        raise AudioError(
            "Aucune piste audio dans ce fichier. "
            "Si tu as envoyé une vidéo, extrais-en d'abord le son."
        )

    duration_text = data.get("format", {}).get("duration")
    if duration_text is None:
        raise AudioError("Durée du fichier indéterminable : fichier probablement corrompu.")

    try:
        return float(duration_text)
    except (TypeError, ValueError):
        raise AudioError("Durée du fichier illisible : fichier probablement corrompu.")


def validate_audio(path: str | Path) -> float:
    """Valide un fichier audio et renvoie sa durée. Lève AudioError sinon."""
    duration = probe_duration(path)

    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioError(
            f"Fichier trop court ({duration:.0f} s). "
            f"Il faut au moins {config.MIN_AUDIO_SECONDS} secondes de musique."
        )
    if duration > config.MAX_AUDIO_SECONDS:
        minutes = config.MAX_AUDIO_SECONDS // 60
        raise AudioError(
            f"Fichier trop long ({duration / 60:.1f} min). "
            f"Maximum accepté : {minutes} minutes."
        )
    return duration


# --- Mode Jamendo (Creative Commons) ----------------------------------------

JAMENDO_API = "https://api.jamendo.com/v3.0/tracks/"


def allows_derivatives(license_url: str) -> bool:
    """Vrai si la licence autorise la création d'une œuvre dérivée.

    Point juridique central de ce projet : « Creative Commons » ne veut pas dire
    « on peut tout faire ». Les licences **ND** (No Derivatives) — `by-nd` et
    `by-nc-nd` — autorisent le partage du morceau tel quel, mais INTERDISENT
    d'en tirer une œuvre dérivée. Or c'est exactement ce que fait cette
    application : elle produit une vidéo qui incorpore le morceau.

    Un tiers des résultats de l'API Jamendo est en ND, et l'API ne sait pas les
    exclure (`ccnd=false` renvoie zéro résultat). On filtre donc nous-mêmes,
    sinon le mode « légal par construction » ne tiendrait pas sa promesse.
    """
    if not license_url:
        # Sans licence identifiable, on ne peut rien garantir : on écarte.
        return False
    return "-nd" not in license_url.lower()


def _appel_jamendo(params: dict) -> list[dict]:
    """Appelle l'API Jamendo et renvoie la liste brute des résultats.

    Toute erreur réseau ou applicative devient une AudioError lisible.
    Une réponse valide mais vide renvoie simplement [] : c'est à l'appelant
    de décider s'il retente autrement.
    """
    try:
        response = requests.get(JAMENDO_API, params=params, timeout=30)
    except requests.RequestException as error:
        raise AudioError(f"Jamendo est injoignable (problème réseau) : {error}")

    if response.status_code == 401:
        raise AudioError("Identifiant Jamendo refusé. Vérifie JAMENDO_CLIENT_ID.")
    if response.status_code == 429:
        raise AudioError("Quota Jamendo dépassé. Réessaie dans quelques minutes.")
    if response.status_code != 200:
        raise AudioError(f"Jamendo a répondu une erreur {response.status_code}.")

    try:
        payload = response.json()
    except ValueError:
        raise AudioError("Réponse de Jamendo illisible.")

    # Jamendo renvoie ses erreurs applicatives dans le corps, pas dans le code HTTP.
    entetes = payload.get("headers", {})
    if entetes.get("status") == "failed":
        raise AudioError(f"Jamendo : {entetes.get('error_message', 'erreur inconnue')}")

    return payload.get("results", [])


def search_jamendo(query: str, limit: int = 8) -> list[dict]:
    """Cherche des morceaux Creative Commons sur Jamendo.

    Renvoie une liste de dictionnaires simples : id, titre, artiste, durée, url.
    Lève AudioError avec un message compréhensible en cas de souci réseau,
    de clé absente ou de quota dépassé — jamais une exception brute.
    """
    if not config.JAMENDO_CLIENT_ID:
        raise AudioError(
            "Le mode Creative Commons a besoin d'un identifiant Jamendo. "
            "Crée-le gratuitement sur https://devportal.jamendo.com/admin/applications "
            "puis renseigne JAMENDO_CLIENT_ID."
        )

    query = (query or "").strip()
    if not query:
        raise AudioError("Entre un mot-clé pour chercher un morceau (ex. « pop », « acoustic »).")

    base_params = {
        "client_id": config.JAMENDO_CLIENT_ID,
        "format": "json",
        # On demande large : environ un tiers des résultats sera écarté par le
        # filtre des licences No Derivatives (voir allows_derivatives).
        "limit": min(limit * 3, 200),
        "audioformat": "mp32",
        # On veut des morceaux chantés : sans paroles, pas de karaoké possible.
        "vocalinstrumental": "vocal",
        "include": "musicinfo",
    }

    # `search` cherche dans les titres et les noms d'artistes ; `tags` cherche
    # par genre et ambiance. Un mot comme « folk » ne donne rien avec le premier
    # mais beaucoup avec le second, alors qu'un titre précis fait l'inverse.
    # On tente donc les deux, dans cet ordre.
    resultats = _appel_jamendo({**base_params, "search": query})
    if not resultats:
        resultats = _appel_jamendo({**base_params, "tags": query})

    tracks = []
    ecartes_nd = 0

    for item in resultats:
        download_url = item.get("audiodownload") or item.get("audio")
        if not download_url:
            continue

        license_url = item.get("license_ccurl", "")
        if not allows_derivatives(license_url):
            ecartes_nd += 1
            continue

        tracks.append(
            {
                "id": item.get("id", ""),
                "name": item.get("name", "Sans titre"),
                "artist": item.get("artist_name", "Artiste inconnu"),
                "duration": int(item.get("duration") or 0),
                "url": download_url,
                "license": license_url,
                "share_url": item.get("shareurl", ""),
            }
        )

    if not tracks:
        if ecartes_nd:
            raise AudioError(
                f"Les {ecartes_nd} morceaux trouvés pour « {query} » sont tous "
                "sous licence « No Derivatives » : elle interdit d'en tirer une "
                "vidéo. Essaie un autre mot-clé."
            )
        raise AudioError(
            f"Aucun morceau chanté trouvé pour « {query} ». "
            "Essaie un mot-clé plus large (ex. « pop », « rock », « folk »)."
        )
    return tracks[:limit]


def download_jamendo_track(track: dict, destination: Path) -> Path:
    """Télécharge un morceau Jamendo vers `destination`, et valide le résultat."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(track["url"], timeout=120, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 16):
                file.write(chunk)
    except requests.RequestException as error:
        raise AudioError(f"Téléchargement du morceau Jamendo impossible : {error}")

    # Le fichier téléchargé passe par la même validation que les uploads :
    # un octet de travers et on le sait tout de suite.
    validate_audio(destination)
    return destination


def format_track_label(track: dict) -> str:
    """Libellé lisible pour la liste déroulante de l'interface."""
    minutes, seconds = divmod(track["duration"], 60)
    return f"{track['artist']} — {track['name']} ({minutes}:{seconds:02d})"
