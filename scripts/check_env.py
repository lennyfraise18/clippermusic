"""Vérification de l'environnement (étape 3 de l'ordre de construction).

Contrôle, dans l'ordre :
  1. ffmpeg et ffprobe présents ;
  2. un encodage réel fonctionne, et quel encodeur sera utilisé ;
  3. les POLICES : c'est le piège n°1 du projet. Si aucune police n'est
     installée dans le conteneur, ffmpeg incruste des carrés vides — ou rien
     du tout — sans jamais renvoyer d'erreur. On teste donc en incrustant
     vraiment du texte et en comptant les pixels allumés.
  4. le fichier audio de test ;
  5. les clés API configurées.

Lancement :
    .venv\\Scripts\\python.exe scripts\\check_env.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import audio, config, subtitles, videos  # noqa: E402

OK = "[ OK ]"
KO = "[FAIL]"
WARN = "[ !! ]"


def check_ffmpeg() -> bool:
    try:
        ffmpeg = config.ffmpeg_path()
        ffprobe = config.ffprobe_path()
    except RuntimeError as error:
        print(f"{KO} ffmpeg\n{error}")
        return False

    version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True
    ).stdout.decode("utf-8", errors="replace").splitlines()[0]
    print(f"{OK} {version}")
    print(f"{OK} ffprobe trouvé : {ffprobe}")
    return True


def check_encoding() -> bool:
    """Encodage trivial : une seconde de mire en 1080x1920."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "trivial.mp4"
        result = subprocess.run(
            [config.ffmpeg_path(), "-y", "-v", "error",
             "-f", "lavfi",
             "-i", f"testsrc=size={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}"
                   f":rate={config.VIDEO_FPS}:duration=1",
             "-c:v", "libx264", "-preset", "ultrafast", str(output)],
            capture_output=True,
        )
        if result.returncode != 0 or not output.exists():
            print(f"{KO} encodage 1080x1920 impossible")
            print(result.stderr.decode("utf-8", errors="replace")[-400:])
            return False
        taille = output.stat().st_size
    print(f"{OK} encodage 1080x1920 réussi ({taille // 1024} Ko)")

    encoder = config.detect_video_encoder()
    if encoder == "libx264":
        print(f"{OK} encodeur retenu : libx264 (CPU)")
    else:
        print(f"{OK} encodeur retenu : {encoder} (accélération matérielle détectée)")
    return True


def check_fonts() -> bool:
    """Incruste vraiment du texte et vérifie que des pixels s'allument."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        fake_words = [
            {"text": "TEST", "start": 0.0, "end": 0.5, "confidence": 1.0},
            {"text": "POLICE", "start": 0.5, "end": 1.0, "confidence": 1.0},
        ]
        subtitles.build_ass(
            [{"text": "TEST POLICE", "start": 0.0, "end": 1.0,
              "confidence": 1.0, "words": fake_words}],
            work / "probe.ass",
        )

        frame = work / "frame.png"
        result = subprocess.run(
            [config.ffmpeg_path(), "-y", "-v", "info",
             "-f", "lavfi",
             "-i", f"color=c=black:s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:d=1",
             "-vf", "subtitles=probe.ass",
             "-frames:v", "1", "-ss", "0.3",
             frame.name],
            cwd=str(work), capture_output=True,
        )
        log = result.stderr.decode("utf-8", errors="replace")

        if result.returncode != 0 or not frame.exists():
            print(f"{KO} incrustation de sous-titres impossible")
            print(log[-500:])
            return False

        # libass annonce explicitement quand il ne trouve pas de police.
        # Attention : « fontselect: (Arial, 700, 0) -> Arial-BoldMT » est un
        # message de SUCCÈS (la police demandée a été résolue). Seules les
        # lignes ci-dessous signalent un vrai problème.
        signaux = [
            "Glyph 0x",
            "failed to find any fallback",
            "No usable fontconfig",
            "Could not find font",
            "fontconfig: Cannot load",
        ]
        alertes = [ligne for ligne in log.splitlines()
                   if any(signal in ligne for signal in signaux)]

        from PIL import Image

        image = Image.open(frame).convert("L")
        pixels_allumes = sum(1 for valeur in image.getdata() if valeur > 128)

    if pixels_allumes < 200:
        print(f"{KO} aucun texte dessiné : la police « {config.SUBTITLE_FONT} » "
              f"est introuvable.")
        print("     Sous Linux/Docker : installe le paquet fonts-dejavu.")
        return False

    print(f"{OK} police « {config.SUBTITLE_FONT} » rendue "
          f"({pixels_allumes} pixels de texte)")
    if alertes:
        print(f"{WARN} libass signale des caractères manquants :")
        for ligne in alertes[:3]:
            print(f"       {ligne.strip()}")
    return True


def check_test_audio() -> bool:
    chemin = ROOT / "assets" / "test_song.mp3"
    if not chemin.exists():
        print(f"{WARN} fichier de test absent — lance : "
              f"python scripts/fetch_test_audio.py")
        return True
    try:
        duree = audio.validate_audio(chemin)
    except audio.AudioError as error:
        print(f"{KO} fichier de test invalide : {error}")
        return False
    print(f"{OK} fichier de test : {chemin.name} ({duree:.0f} s)")
    return True


def check_api_keys() -> bool:
    cles = [
        ("PEXELS_API_KEY", config.PEXELS_API_KEY, "https://www.pexels.com/api/new/"),
        ("PIXABAY_API_KEY", config.PIXABAY_API_KEY, "https://pixabay.com/api/docs/"),
        ("JAMENDO_CLIENT_ID", config.JAMENDO_CLIENT_ID,
         "https://devportal.jamendo.com/admin/applications"),
    ]
    for nom, valeur, lien in cles:
        if valeur:
            print(f"{OK} {nom} configurée")
        else:
            print(f"{WARN} {nom} absente — à créer ici : {lien}")

    if not videos.has_any_key():
        print(f"{WARN} aucune clé Pexels/Pixabay : le pipeline tournera quand même,")
        print("       en reprenant les fonds sur Wikimedia Commons (qualité inégale).")
    return True


def main() -> int:
    print("=== Vérification de l'environnement ===\n")

    etapes = [
        ("ffmpeg", check_ffmpeg),
        ("encodage", check_encoding),
        ("polices", check_fonts),
        ("audio de test", check_test_audio),
        ("clés API", check_api_keys),
    ]

    bloquant = False
    for nom, fonction in etapes:
        print(f"\n--- {nom} ---")
        try:
            if not fonction():
                bloquant = True
        except Exception as error:
            print(f"{KO} {nom} : {error}")
            bloquant = True

    print("\n" + "=" * 40)
    if bloquant:
        print("Des points bloquants restent à régler (voir les lignes [FAIL]).")
        return 1
    print("Environnement prêt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
