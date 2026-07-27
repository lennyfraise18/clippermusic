"""Test de la reconnaissance de liens musicaux.

Vérifie que coller un lien YouTube, Spotify ou Deezer permet d'identifier le
morceau (via les API oEmbed officielles, sans rien télécharger) et de proposer
un équivalent libre de droits.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_liens.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import liens  # noqa: E402

LIENS = [
    "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
    "https://open.spotify.com/track/6habFhsOp2NvshLv26DqMb",
    "https://www.deezer.com/track/3135556",
]

TITRES_A_NETTOYER = [
    "Luis Fonsi - Despacito ft. Daddy Yankee (Official Video) [4K]",
    "Stromae - Alors On Danse (Official Music Video)",
    "Daft Punk - Harder, Better, Faster, Stronger | Official Audio HD",
    "Adele - Someone Like You (Lyrics Video)",
]


def test_nettoyage() -> bool:
    print("--- Nettoyage des titres ---")
    for titre in TITRES_A_NETTOYER:
        print(f"  « {titre[:52]} »")
        print(f"      -> « {liens.nettoyer_titre(titre)} »")
    return True


def test_detection() -> bool:
    print("\n--- Détection de plateforme ---")
    cas = [
        ("https://www.youtube.com/watch?v=abc", "youtube"),
        ("https://youtu.be/abc", "youtube"),
        ("https://open.spotify.com/track/abc", "spotify"),
        ("https://www.deezer.com/track/123", "deezer"),
        ("https://example.com/musique.mp3", None),
        ("pop", None),
    ]
    ok = True
    for texte, attendu in cas:
        obtenu = liens.detecter_plateforme(texte)
        marque = "OK  " if obtenu == attendu else "ÉCHEC"
        print(f"  {marque} {texte[:44]:<46} -> {obtenu}")
        if obtenu != attendu:
            ok = False
    return ok


def test_lecture() -> bool:
    print("\n--- Lecture des vrais liens ---")
    ok = True
    for url in LIENS:
        try:
            infos = liens.lire_titre(url)
        except liens.LienError as erreur:
            print(f"  ÉCHEC {url[:46]} : {erreur}")
            ok = False
            continue
        print(f"  {infos['plateforme']:<9} « {infos['titre'][:44]} »")
        print(f"            artiste   : {infos['artiste'] or '(non fourni)'}")
        print(f"            recherche : {infos['recherche']}")
    return ok


def test_equivalent() -> bool:
    print("\n--- Équivalent libre de droits ---")
    try:
        infos = liens.lire_titre(LIENS[0])
        propositions = liens.chercher_equivalent_libre(infos)
    except liens.LienError as erreur:
        print(f"  ÉCHEC : {erreur}")
        return False

    print(f"  Morceau identifié : {infos['titre'][:50]}")
    print(f"  {len(propositions)} morceaux libres proposés :")
    for morceau in propositions[:5]:
        print(f"    {morceau['artist'][:22]:<24} {morceau['name'][:30]}")
    return len(propositions) > 0


def test_lien_invalide() -> bool:
    print("\n--- Lien non reconnu ---")
    try:
        liens.lire_titre("https://exemple.fr/ma-musique")
    except liens.LienError as erreur:
        print(f"  OK — message clair : « {str(erreur).splitlines()[0]} »")
        return True
    print("  ÉCHEC — aucune erreur levée.")
    return False


def main() -> int:
    print("=== Reconnaissance de liens musicaux ===\n")
    resultats = [
        test_nettoyage(),
        test_detection(),
        test_lecture(),
        test_equivalent(),
        test_lien_invalide(),
    ]
    print("\n" + "=" * 54)
    if all(resultats):
        print("OK — la reconnaissance de liens fonctionne.")
        return 0
    print("Certains contrôles ont échoué.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
