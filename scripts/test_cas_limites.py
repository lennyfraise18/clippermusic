"""Test des cas qui cassent (checklist finale).

Aucune clé API n'est nécessaire : tous les fichiers d'entrée sont fabriqués sur
place par ffmpeg. On vérifie qu'un cas anormal produit un MESSAGE CLAIR et pas
une trace Python, et que rien ne traîne sur le disque après coup.

Cas couverts :
  1. fichier inexistant
  2. fichier corrompu (des octets au hasard renommés en .mp3)
  3. fichier vidéo sans piste audio
  4. fichier trop court
  5. fichier instrumental (un simple sinus) → « aucune parole détectée »
  6. nettoyage des fichiers temporaires après une erreur

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_cas_limites.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import audio, config, pipeline, transcribe  # noqa: E402

BAC = None  # dossier temporaire, créé dans main()


def _ffmpeg(arguments: list[str]) -> None:
    subprocess.run(
        [config.ffmpeg_path(), "-y", "-v", "error", *arguments],
        check=True, capture_output=True,
    )


def cas(numero: int, titre: str, action, attendu: str) -> bool:
    """Lance `action` et vérifie qu'elle lève une erreur au message lisible."""
    print(f"\n--- Cas {numero} : {titre} ---")
    try:
        action()
    except (audio.AudioError, transcribe.TranscriptionError,
            pipeline.PipelineError) as erreur:
        message = str(erreur)
        print(f"  message affiché : « {message.splitlines()[0]} »")
        if attendu.lower() in message.lower():
            print("  OK — message clair et attendu.")
            return True
        print(f"  ÉCHEC — le message ne mentionne pas « {attendu} ».")
        return False
    except Exception as erreur:
        print(f"  ÉCHEC — exception brute, pas un message pour l'utilisateur :")
        print(f"    {type(erreur).__name__}: {erreur}")
        return False

    print("  ÉCHEC — aucune erreur levée, alors que le fichier est invalide.")
    return False


def cas_1_inexistant() -> bool:
    return cas(1, "fichier inexistant",
               lambda: audio.validate_audio(BAC / "nexiste_pas.mp3"),
               "introuvable")


def cas_2_corrompu() -> bool:
    chemin = BAC / "corrompu.mp3"
    chemin.write_bytes(os.urandom(200_000))
    # Selon les octets tirés, ffprobe échoue franchement ou croit voir un flux
    # sans pouvoir en lire la durée. Les deux messages parlent de corruption,
    # c'est ce qui compte pour l'utilisateur.
    return cas(2, "fichier corrompu (octets au hasard)",
               lambda: audio.validate_audio(chemin),
               "corrompu")


def cas_3_sans_audio() -> bool:
    chemin = BAC / "muet.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
             "-c:v", "libx264", "-preset", "ultrafast", str(chemin)])
    return cas(3, "vidéo sans piste audio",
               lambda: audio.validate_audio(chemin),
               "aucune piste audio")


def cas_4_trop_court() -> bool:
    chemin = BAC / "court.mp3"
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:a", "libmp3lame", str(chemin)])
    return cas(4, "fichier trop court (2 s)",
               lambda: audio.validate_audio(chemin),
               "trop court")


def cas_5_instrumental() -> bool:
    """Le cas le plus important : Whisper ne doit pas inventer des paroles."""
    chemin = BAC / "instrumental.mp3"
    # Un accord de trois sinus : de la musique, aucune voix.
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=220:duration=25",
        "-f", "lavfi", "-i", "sine=frequency=277:duration=25",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=25",
        "-filter_complex", "[0][1][2]amix=inputs=3",
        "-c:a", "libmp3lame", str(chemin),
    ])
    print("\n--- Cas 5 : fichier instrumental (transcription réelle, ~30 s) ---")
    try:
        resultat = transcribe.transcribe_audio(chemin)
    except transcribe.TranscriptionError as erreur:
        message = str(erreur)
        print(f"  message affiché : « {message.splitlines()[0]} »")
        if "aucune parole" in message.lower():
            print("  OK — Whisper n'a pas inventé de paroles.")
            return True
        print("  ÉCHEC — message inattendu.")
        return False
    except Exception as erreur:
        print(f"  ÉCHEC — exception brute : {type(erreur).__name__}: {erreur}")
        return False

    mots = sum(len(s["words"]) for s in resultat["segments"])
    print(f"  ÉCHEC — {mots} mots « détectés » sur un instrumental :")
    for segment in resultat["segments"][:3]:
        print(f"    « {segment['text']} »")
    return False


def cas_6_nettoyage() -> bool:
    """Une erreur en cours de route ne doit rien laisser derrière elle."""
    print("\n--- Cas 6 : nettoyage des fichiers temporaires après erreur ---")
    config.ensure_dirs()
    avant = {chemin.name for chemin in config.WORK_DIR.iterdir()}

    try:
        pipeline.generate_clip(BAC / "corrompu.mp3")
    except pipeline.PipelineError:
        pass
    except Exception as erreur:
        print(f"  ÉCHEC — exception brute : {erreur}")
        return False

    apres = {chemin.name for chemin in config.WORK_DIR.iterdir()}
    restes = apres - avant

    if restes:
        print(f"  ÉCHEC — {len(restes)} dossier(s) laissé(s) derrière : {restes}")
        return False
    print("  OK — dossier de travail propre, malgré l'erreur.")
    return True


def main() -> int:
    global BAC

    import tempfile

    config.ensure_dirs()
    with tempfile.TemporaryDirectory() as dossier:
        BAC = Path(dossier)
        print("=== Cas limites ===")

        resultats = [
            cas_1_inexistant(),
            cas_2_corrompu(),
            cas_3_sans_audio(),
            cas_4_trop_court(),
            cas_5_instrumental(),
            cas_6_nettoyage(),
        ]

    reussis = sum(resultats)
    print("\n" + "=" * 50)
    print(f"{reussis}/{len(resultats)} cas gérés proprement.")
    return 0 if reussis == len(resultats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
