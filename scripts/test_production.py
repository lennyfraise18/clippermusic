"""Test de bout en bout de l'application DÉPLOYÉE.

Contrairement aux autres scripts, celui-ci ne teste pas le code local mais le
service en ligne : c'est le seul moyen de vérifier ce qui ne se voit qu'en
conditions réelles — mémoire disponible, polices du conteneur, durée réelle
d'un traitement sur la machine d'hébergement.

Chaque contrôle a un critère de réussite explicite, affiché avec son résultat.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_production.py
    .venv\\Scripts\\python.exe scripts\\test_production.py https://mon-autre-url
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from gradio_client import Client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import config  # noqa: E402

URL_DEFAUT = "https://clippermusic-production.up.railway.app"
MP3_TEST = "https://archive.org/download/NineInchNails-TheSlip/04_Discipline.mp3"

resultats: list[tuple[str, bool, str]] = []


def controle(nom: str, reussi: bool, detail: str = "") -> bool:
    resultats.append((nom, reussi, detail))
    print(f"  {'OK  ' if reussi else 'ÉCHEC'}  {nom}" + (f" — {detail}" if detail else ""))
    return reussi


def attendre_service(url: str, minutes: int = 12) -> bool:
    """Attend que le service réponde, le temps qu'un déploiement se termine."""
    print(f"Attente du service ({minutes} min maximum)…")
    limite = time.time() + minutes * 60
    while time.time() < limite:
        try:
            if requests.get(url, timeout=20).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(15)
    return False


def test_disponibilite(url: str) -> bool:
    print("\n--- 1. Disponibilité ---")
    try:
        reponse = requests.get(url, timeout=30)
    except requests.RequestException as erreur:
        return controle("le service répond", False, str(erreur)[:60])

    ok = controle("le service répond", reponse.status_code == 200,
                  f"HTTP {reponse.status_code}")
    controle("l'interface est bien ClipperMusic", "ClipperMusic" in reponse.text)
    controle("l'identité visuelle est chargée", "egaliseur" in reponse.text)
    return ok


def test_recherche(client: Client) -> bool:
    print("\n--- 2. Recherche de musique libre (valide les clés API) ---")
    try:
        r = client.predict("rock", api_name="/analyser_entree")
    except Exception as erreur:
        return controle("la recherche répond", False, type(erreur).__name__)

    message = r[0].get("value") if isinstance(r[0], dict) else r[0]
    choix = r[1].get("choices") if isinstance(r[1], dict) else None
    controle("la recherche répond", True)
    return controle("des morceaux libres sont proposés", bool(choix),
                    f"{len(choix or [])} morceaux")


def test_lien_direct(client: Client) -> bool:
    print("\n--- 3. Reconnaissance d'un lien audio direct ---")
    try:
        r = client.predict(MP3_TEST, api_name="/analyser_entree")
    except Exception as erreur:
        return controle("le lien est analysé", False, type(erreur).__name__)

    message = str(r[0].get("value") if isinstance(r[0], dict) else r[0])
    return controle("le fichier audio est détecté", "détecté" in message,
                    message[:50])


def test_generation(client: Client) -> bool:
    """Le vrai test : c'est lui qui échouait par manque de mémoire."""
    print("\n--- 4. Génération complète (le test décisif) ---")
    debut = time.time()
    try:
        r = client.predict(
            None, MP3_TEST, "", False, config.WHISPER_MODEL,
            "détection automatique", api_name="/generer",
        )
    except Exception as erreur:
        ecoule = time.time() - debut
        indice = ("mémoire insuffisante (le conteneur redémarre)"
                  if ecoule < 60 else "traitement trop long")
        return controle("la génération aboutit", False,
                        f"{type(erreur).__name__} après {ecoule:.0f} s — {indice}")

    ecoule = time.time() - debut
    video = r[0]
    chemin = video.get("video") if isinstance(video, dict) else video
    message = str(r[1])

    if not chemin:
        # L'application a répondu proprement mais sans vidéo : le message
        # qu'elle renvoie contient la raison, c'est lui qu'il faut lire.
        controle("la génération aboutit", False,
                 f"{ecoule:.0f} s — message : {message[:200]}")
        return False

    controle("la génération aboutit", True, f"{ecoule:.0f} s")
    controle("le message de fin est un succès", message.startswith("✅"),
             message[:60].replace("\n", " "))
    return _verifier_video(chemin)


def _verifier_video(chemin_distant: str) -> bool:
    """Télécharge la vidéo produite et contrôle son format."""
    local = config.OUTPUT_DIR / "production.mp4"
    config.ensure_dirs()
    try:
        shutil.copy(chemin_distant, local)
    except Exception as erreur:
        return controle("la vidéo est récupérable", False, str(erreur)[:60])

    controle("la vidéo est récupérable", True, f"{local.stat().st_size/1e6:.1f} Mo")

    sonde = subprocess.run(
        [config.ffprobe_path(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(local)],
        capture_output=True,
    )
    donnees = json.loads(sonde.stdout)
    flux_video = next(s for s in donnees["streams"] if s["codec_type"] == "video")
    flux_audio = next((s for s in donnees["streams"] if s["codec_type"] == "audio"), None)
    duree = float(donnees["format"]["duration"])

    # On ne compare pas à une résolution fixe : le serveur réduit
    # volontairement en 720x1280 quand sa mémoire est limitée. Ce qui compte,
    # c'est que le format reste vertical 9:16 et assez défini pour les réseaux.
    largeur, hauteur = flux_video["width"], flux_video["height"]
    ratio = hauteur / largeur if largeur else 0
    controle(
        "format vertical 9:16",
        abs(ratio - 16 / 9) < 0.02,
        f"{largeur}x{hauteur} (ratio {ratio:.2f})",
    )
    controle(
        "définition suffisante pour les réseaux",
        hauteur >= 1280,
        f"{hauteur} px de haut (minimum 1280)",
    )
    controle("la bande son est présente", flux_audio is not None)
    controle(
        "durée conforme au format court",
        duree <= config.MAX_CLIP_SECONDS + 5,
        f"{duree:.1f} s (max attendu {config.MAX_CLIP_SECONDS + 5})",
    )
    return _verifier_sous_titres(local, duree)


def _verifier_sous_titres(video: Path, duree: float) -> bool:
    """Vérifie que du texte est réellement dessiné — le piège des polices.

    Sans police dans le conteneur, ffmpeg n'incruste rien du tout, sans lever
    d'erreur. On compte donc les pixels clairs d'une image prise au milieu du
    clip, là où il y a forcément des paroles.
    """
    image = config.OUTPUT_DIR / "production_frame.png"
    subprocess.run(
        [config.ffmpeg_path(), "-y", "-v", "error", "-ss", f"{duree/2:.1f}",
         "-i", str(video), "-frames:v", "1", str(image)],
        capture_output=True,
    )
    if not image.exists():
        return controle("les sous-titres sont dessinés", False, "image non extraite")

    from PIL import Image

    with Image.open(image) as ouverte:
        # Les sous-titres sont en bas : on ne regarde que ce tiers.
        largeur, hauteur = ouverte.size
        bas = ouverte.convert("L").crop((0, int(hauteur * 0.62), largeur, hauteur))
        clairs = sum(1 for valeur in bas.getdata() if valeur > 230)

    return controle(
        "les sous-titres sont dessinés (polices du conteneur)",
        clairs > 2000,
        f"{clairs} pixels de texte",
    )


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else URL_DEFAUT
    print(f"=== Test de production ===\n{url}\n")

    if not attendre_service(url):
        print("\nLe service n'a jamais répondu. Déploiement en échec ?")
        return 1

    if not test_disponibilite(url):
        return 1

    client = Client(url, verbose=False)
    test_recherche(client)
    test_lien_direct(client)
    test_generation(client)

    reussis = sum(1 for _, ok, _ in resultats if ok)
    print("\n" + "=" * 62)
    print(f"{reussis}/{len(resultats)} contrôles réussis")
    for nom, ok, detail in resultats:
        if not ok:
            print(f"  ÉCHEC : {nom} — {detail}")
    return 0 if reussis == len(resultats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
