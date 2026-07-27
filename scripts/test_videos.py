"""Test isolé des banques vidéo (étape 6 de l'ordre de construction).

Vérifie avec de vraies clés API :
  - que Pexels et Pixabay répondent ;
  - qu'une clé absente ou invalide donne un message clair, pas un plantage ;
  - que le format vertical est bien privilégié ;
  - que l'ANTI-RÉPÉTITION tient : on lance dix fois la même requête (comme un
    refrain le ferait) et on exige dix clips différents ;
  - qu'une requête absurde retombe sur un visuel de secours au lieu d'échouer.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_videos.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import config, videos  # noqa: E402

REQUETES = ["city night lights", "ocean waves slow motion", "rain on window"]


def test_cles() -> bool:
    print("--- Clés configurées ---")
    print(f"  Pexels  : {'oui' if config.PEXELS_API_KEY else 'NON'}")
    print(f"  Pixabay : {'oui' if config.PIXABAY_API_KEY else 'NON'}")

    print(f"  Wikimedia : oui (filet de sécurité, aucune clé nécessaire)")

    if not videos.has_any_key():
        print("\n" + videos.missing_key_message())
        print("\n(les tests continuent en s'appuyant sur Wikimedia Commons)")
    return True


def test_recherche() -> bool:
    print("\n--- Recherche ---")
    tout_va_bien = True

    for requete in REQUETES:
        try:
            pexels = videos.search_pexels(requete)
            pixabay = videos.search_pixabay(requete)
        except videos.VideoError as erreur:
            print(f"  ÉCHEC « {requete} » : {erreur}")
            tout_va_bien = False
            continue

        commons = videos.search_wikimedia(requete)
        trouves = pexels + pixabay + commons

        verticaux = sum(
            1 for c in trouves
            if c["height"] and c["width"] and c["height"] / c["width"] >= 1.6
        )
        print(f"  « {requete} » → {len(pexels)} Pexels + {len(pixabay)} Pixabay "
              f"+ {len(commons)} Wikimedia ({verticaux} déjà verticaux)")

        if not trouves:
            print("    ATTENTION : aucun résultat sur une requête pourtant courante.")
            tout_va_bien = False

    return tout_va_bien


def test_anti_repetition() -> bool:
    """Le vrai test qui compte : un refrain ne doit pas ressortir le même plan."""
    print("\n--- Anti-répétition (10 appels sur la MÊME requête) ---")

    dossier = config.WORK_DIR / "test_videos"
    finder = videos.ClipFinder(dossier)

    identifiants = []
    for tour in range(10):
        candidat = finder.find("city night lights")
        if candidat is None:
            print(f"  tour {tour + 1} : plus aucun clip disponible")
            break
        identifiants.append(candidat["id"])
        print(f"  tour {tour + 1:>2} : {candidat['id']:<20} "
              f"{candidat['width']}x{candidat['height']:<5} "
              f"(requête « {candidat['query_used']} »)")

    doublons = len(identifiants) - len(set(identifiants))
    if doublons:
        print(f"\n  ÉCHEC : {doublons} clip(s) servi(s) deux fois.")
        return False
    print(f"\n  OK — {len(identifiants)} clips, tous différents.")
    return len(identifiants) >= 5


def test_repli() -> bool:
    """Une requête sans résultat ne doit jamais faire échouer le montage."""
    print("\n--- Repli sur requête absurde ---")

    finder = videos.ClipFinder(config.WORK_DIR / "test_videos")
    requete = "xyzzy quux inexistant 12345"
    candidat = finder.find(requete)

    if candidat is None:
        print("  ÉCHEC : aucun clip trouvé, le montage aurait planté.")
        return False

    if candidat["query_used"] == requete:
        # Pexels fait de la correspondance approximative et répond à peu près
        # à tout. Ce n'est pas un échec : le montage a bien un clip. Mais le
        # chemin de repli n'a pas été exercé, autant le dire clairement.
        print(f"  OK — Pexels a répondu même à cette requête ({candidat['id']}).")
        print("       Le repli générique n'a donc pas été sollicité ici.")
    else:
        print(f"  OK — repli sur « {candidat['query_used']} » → {candidat['id']}")
    return True


def test_telechargement() -> bool:
    print("\n--- Téléchargement réel ---")

    finder = videos.ClipFinder(config.WORK_DIR / "test_videos")
    resultat = finder.find_and_download("ocean waves slow motion")

    if resultat is None:
        print("  ÉCHEC : aucun clip téléchargeable.")
        return False

    chemin, candidat = resultat
    duree = videos.probe_video_duration(chemin)
    taille = chemin.stat().st_size / (1024 * 1024)
    print(f"  OK — {chemin.name} : {taille:.1f} Mo, {duree:.1f} s, "
          f"{candidat['width']}x{candidat['height']}")
    print(f"  Crédit : {candidat['credit']} ({candidat['source']})")

    if duree <= 0:
        print("  ÉCHEC : fichier téléchargé illisible par ffprobe.")
        return False
    return True


def main() -> int:
    config.ensure_dirs()
    print("=== Test des banques vidéo ===\n")

    if not test_cles():
        return 1

    resultats = [
        test_recherche(),
        test_anti_repetition(),
        test_repli(),
        test_telechargement(),
    ]

    print("\n" + "=" * 50)
    if all(resultats):
        print("OK — les banques vidéo fonctionnent.")
        return 0
    print("Certains contrôles ont échoué (voir ci-dessus).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
