"""Transcription exécutée dans un processus séparé.

Pourquoi un processus à part plutôt qu'un simple appel de fonction
-----------------------------------------------------------------
Le modèle de transcription occupe plusieurs centaines de mégaoctets. Une fois
la transcription finie, il ne sert plus à rien — mais le libérer depuis Python
ne suffit pas : `del` puis `gc.collect()` rendent la mémoire à l'allocateur de
Python, qui la garde en réserve au lieu de la restituer au système. Vu du
conteneur, elle reste occupée.

Or c'est ensuite ffmpeg qui a besoin de mémoire, et lui est un programme
externe : il ne peut pas puiser dans la réserve de Python. Sur un hébergement
limité, il se faisait tuer par le système (signal 9).

En exécutant la transcription dans un processus séparé, sa mémoire est rendue
au système **à coup sûr** quand il se termine — c'est le noyau qui s'en charge,
pas l'allocateur de Python.

Ce fichier est lancé en ligne de commande par transcribe.py :
    python -m modules.transcribe_worker <audio> <sortie.json> [modèle] [langue]
"""

import json
import sys
from pathlib import Path

# Permet d'exécuter ce fichier directement, sans que le projet soit installé.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import transcribe  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: transcribe_worker <audio> <sortie.json> [modèle] [langue]",
              file=sys.stderr)
        return 2

    audio_path = sys.argv[1]
    sortie = Path(sys.argv[2])
    modele = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    langue = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    rang = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 0

    try:
        resultat = transcribe.transcrire_directement(
            audio_path, model_name=modele, language=langue, rang_passage=rang
        )
    except transcribe.TranscriptionError as erreur:
        sortie.write_text(
            json.dumps({"erreur": str(erreur)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1
    except Exception as erreur:  # noqa: BLE001
        sortie.write_text(
            json.dumps({"erreur": f"{type(erreur).__name__}: {erreur}"},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        return 1

    sortie.write_text(json.dumps(resultat, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
