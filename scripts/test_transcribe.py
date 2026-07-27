"""Test isolé de la transcription (étape 4 de l'ordre de construction).

Vérifie que whisper-timestamped produit bien des timestamps MOT PAR MOT sur le
fichier de test, et que la sélection du passage le plus dense fonctionne.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_transcribe.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import audio, config, transcribe  # noqa: E402

AUDIO = ROOT / "assets" / "test_song.mp3"
CACHE = ROOT / "assets" / "test_transcription.json"


def main() -> int:
    if not AUDIO.exists():
        print(f"Fichier de test absent : {AUDIO}")
        print("Lance d'abord : python scripts/fetch_test_audio.py")
        return 1

    print(f"Fichier   : {AUDIO.name}")
    duration = audio.validate_audio(AUDIO)
    print(f"Durée     : {duration:.1f} s")
    print(f"Modèle    : {config.WHISPER_MODEL}")
    print("Transcription en cours (plusieurs minutes au premier lancement,")
    print("le modèle Whisper doit être téléchargé)…\n")

    started = time.time()
    result = transcribe.transcribe_audio(AUDIO, progress=lambda m: print(f"  → {m}"))
    elapsed = time.time() - started

    segments = result["segments"]
    words = sum(len(segment["words"]) for segment in segments)

    print(f"\nTerminé en {elapsed:.0f} s")
    print(f"Langue détectée      : {result['language']}")
    print(f"Extrait retenu       : à partir de {result['start_offset']:.1f} s, "
          f"durée {result['duration']:.1f} s")
    print(f"Segments conservés   : {len(segments)}")
    print(f"Mots avec timestamps : {words}")

    print("\n--- Cinq premiers segments ---")
    for segment in segments[:5]:
        print(f"[{segment['start']:6.2f} → {segment['end']:6.2f}] {segment['text']}")

    print("\n--- Timestamps mot par mot du premier segment ---")
    for word in segments[0]["words"]:
        print(f"  {word['start']:6.2f} → {word['end']:6.2f}  "
              f"{word['text']:<20} (confiance {word['confidence']:.2f})")

    # On garde le résultat sous la main : les tests suivants (mots-clés,
    # sous-titres, montage) peuvent ainsi tourner sans relancer Whisper.
    CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRésultat mis en cache dans {CACHE.name}")

    if words < config.MIN_WORDS_REQUIRED:
        print("\nÉCHEC : trop peu de mots détectés.")
        return 1

    print("\nOK — les timestamps mot par mot sont exploitables pour le karaoké.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
