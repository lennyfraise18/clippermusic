"""Télécharge un morceau chanté libre de droits pour les tests.

Source : Wikimedia Commons, fichiers du domaine public. Aucune clé API requise,
ce qui permet de tester le pipeline avant même d'avoir créé le moindre compte.

Lancement :
    .venv\\Scripts\\python.exe scripts\\fetch_test_audio.py
"""

import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DESTINATION = ROOT / "assets" / "test_song.mp3"

# Enregistrements du domaine public (avant 1930), donc utilisables sans aucune
# restriction. On en essaie plusieurs : Commons renomme parfois ses fichiers.
CANDIDATES = [
    "Let_Me_Call_You_Sweetheart_(1911).ogg",
    "Italian_Street_Song_(1916_sound_recording).mp3",
    "How_Can_They_Tell_That_I'm_Irish.ogg",
    "Down_by_the_Old_Mill_Stream_-_Harry_Macdonough_(1910).ogg",
]

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia renvoie 429 aux User-Agent génériques ou anonymes : leur politique
# impose un agent descriptif identifiant le projet. Sans ça, rien ne se télécharge.
HEADERS = {
    "User-Agent": (
        "clip-paroles-portfolio/1.0 "
        "(https://huggingface.co/spaces; projet de démonstration portfolio)"
    )
}


def resolve_url(filename: str) -> str | None:
    """Demande à l'API Commons l'URL réelle d'un fichier."""
    try:
        response = requests.get(
            COMMONS_API,
            params={
                "action": "query",
                "titles": f"File:{filename}",
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            },
            headers=HEADERS,
            timeout=30,
        )
        pages = response.json()["query"]["pages"]
    except Exception:
        return None

    for page in pages.values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]
    return None


def search_commons() -> list[str]:
    """Repli : cherche n'importe quel enregistrement chanté ancien sur Commons."""
    try:
        response = requests.get(
            COMMONS_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": 'filetype:audio "1910" song vocal',
                "srnamespace": "6",
                "srlimit": "15",
                "format": "json",
            },
            headers=HEADERS,
            timeout=30,
        )
        results = response.json()["query"]["search"]
    except Exception:
        return []
    return [item["title"].removeprefix("File:") for item in results]


def download_and_convert(url: str, destination: Path) -> bool:
    """Télécharge puis convertit en MP3 mono 44,1 kHz via ffmpeg."""
    temporary = destination.with_suffix(".download")
    try:
        response = requests.get(url, headers=HEADERS, timeout=180, stream=True)
        response.raise_for_status()
        with open(temporary, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 16):
                file.write(chunk)
    except Exception as error:
        print(f"  téléchargement impossible : {error}")
        temporary.unlink(missing_ok=True)
        return False

    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(temporary),
         "-ac", "1", "-ar", "44100", "-b:a", "160k", str(destination)],
        capture_output=True,
    )
    temporary.unlink(missing_ok=True)

    if result.returncode != 0:
        print("  conversion ffmpeg échouée :",
              result.stderr.decode("utf-8", errors="replace")[-300:])
        return False
    return True


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)

    for filename in CANDIDATES + search_commons():
        print(f"Essai : {filename}")
        url = resolve_url(filename)
        if not url:
            print("  introuvable sur Commons")
            continue
        if download_and_convert(url, DESTINATION):
            size_mo = DESTINATION.stat().st_size / (1024 * 1024)
            print(f"\nOK — {DESTINATION} ({size_mo:.1f} Mo)")
            print(f"Source : https://commons.wikimedia.org/wiki/File:{filename}")
            return 0

    print("\nAucun fichier de test n'a pu être récupéré.")
    print("Solution de repli : place toi-même un MP3 libre de droits dans "
          "assets/test_song.mp3")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
