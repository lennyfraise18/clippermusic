"""Test isolé de l'extraction de mots-clés (étape 5 de l'ordre de construction).

Teste sur des paroles réelles et volontairement difficiles : argot, anglicismes,
mots abstraits, phrases sans aucun mot concret. Le module ne doit jamais planter
et doit toujours renvoyer une requête exploitable.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_keywords.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import keywords  # noqa: E402

# Cas volontairement pénibles : c'est là que les extracteurs naïfs cassent.
CAS_FRANCAIS = [
    "je marche seul dans la ville la nuit",
    "wesh gros ça va ou quoi tranquille",              # argot pur, rien de concret
    "mon coeur est en miettes depuis que t'es partie",  # abstrait
    "j'ai la haine, la rage, ça part en freestyle",     # argot + anglicisme
    "la la la la la la la",                             # aucun contenu
    "on roule sur l'autoroute vers la mer",
    "ma mère m'a dit petit reste droit",
    "y'a rien à faire ici",                             # que des mots vides
    "l'amour c'est la liberté",                         # deux abstraits
    "chaque matin je regarde par la fenêtre",
]

CAS_ANGLAIS = [
    "I'm walking alone down the empty street",
    "my heart is broken since you left",
    "yeah yeah baby gonna make it right",              # que des tics de langage
    "we drive all night under the neon lights",
    "dreaming of the ocean and the sun",
]


def tester(cas: list[str], langue: str) -> int:
    print(f"\n=== Paroles en « {langue} » ===")
    nlp = keywords._load_nlp(langue)

    segments = [{"text": texte, "start": 0.0, "end": 3.0, "words": []} for texte in cas]
    resultats = keywords.build_queries(segments, langue)

    problemes = 0
    for texte, resultat in zip(cas, resultats):
        mot = resultat["keyword"] or "—"
        origine = "dictionnaire" if resultat["query"] not in keywords.FALLBACK_QUERIES else "REPLI"
        print(f"\n  « {texte} »")
        print(f"      mot-clé : {mot}")
        print(f"      requête : {resultat['query']}  ({origine})")

        if not resultat["query"]:
            print("      ÉCHEC : requête vide")
            problemes += 1

    del nlp
    return problemes


def tester_sur_transcription() -> int:
    """Rejoue l'extraction sur la vraie transcription du fichier de test."""
    cache = ROOT / "assets" / "test_transcription.json"
    if not cache.exists():
        print("\n(transcription de test absente — lance scripts/test_transcribe.py)")
        return 0

    donnees = json.loads(cache.read_text(encoding="utf-8"))
    print(f"\n=== Vraies paroles du fichier de test ({donnees['language']}) ===")

    resultats = keywords.build_queries(donnees["segments"], donnees["language"])
    replis = 0
    for resultat in resultats:
        marque = "" if resultat["query"] not in keywords.FALLBACK_QUERIES else "  [repli]"
        print(f"  {resultat['text'][:52]:<54} → {resultat['query']}{marque}")
        if marque:
            replis += 1

    print(f"\n  {len(resultats) - replis}/{len(resultats)} segments ont trouvé "
          f"un visuel dans le dictionnaire.")
    return 0


def main() -> int:
    problemes = 0
    problemes += tester(CAS_FRANCAIS, "fr")
    problemes += tester(CAS_ANGLAIS, "en")
    problemes += tester_sur_transcription()

    print("\n" + "=" * 60)
    if problemes:
        print(f"ÉCHEC : {problemes} cas sans requête exploitable.")
        return 1
    print("OK — aucun cas ne casse, tous les segments ont une requête.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
