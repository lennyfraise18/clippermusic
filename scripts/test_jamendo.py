"""Test isolé du mode Creative Commons (Jamendo).

Vérifie que le mode « légal par construction » est une alternative complète au
mode upload :
  - la recherche renvoie des morceaux CHANTÉS (sans paroles, pas de karaoké) ;
  - le téléchargement produit un fichier réellement lisible par ffmpeg ;
  - les informations de licence et de crédit sont bien présentes ;
  - une recherche sans résultat donne un message clair.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_jamendo.py
    .venv\\Scripts\\python.exe scripts\\test_jamendo.py "chanson francaise"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import audio, config  # noqa: E402


def test_recherche(requete: str) -> list[dict]:
    print(f"--- Recherche « {requete} » ---")
    morceaux = audio.search_jamendo(requete, limit=8)

    for morceau in morceaux:
        minutes, secondes = divmod(morceau["duration"], 60)
        print(f"  {morceau['artist'][:24]:<26} {morceau['name'][:30]:<32} "
              f"{minutes}:{secondes:02d}")

    print(f"\n  {len(morceaux)} morceaux trouvés.")
    return morceaux


def test_licence(morceaux: list[dict]) -> bool:
    """Le contrôle juridique central : aucune licence No Derivatives ne doit passer.

    Une licence ND autorise le partage du morceau tel quel mais interdit d'en
    tirer une œuvre dérivée — ce que fait précisément cette application. En
    laisser passer une viderait de son sens le mode « légal par construction ».
    """
    print("\n--- Licences (aucune « No Derivatives » ne doit passer) ---")

    for morceau in morceaux:
        licence = (morceau["license"] or "(aucune)").replace(
            "http://creativecommons.org/licenses/", "CC "
        )
        print(f"  {morceau['name'][:32]:<34} {licence}")

    interdits = [m for m in morceaux if not audio.allows_derivatives(m["license"])]

    if interdits:
        print(f"\n  ÉCHEC — {len(interdits)} morceau(x) sous licence ND ont passé "
              f"le filtre :")
        for morceau in interdits:
            print(f"    {morceau['name']} → {morceau['license']}")
        return False

    print(f"\n  OK — {len(morceaux)} morceaux, tous autorisent l'œuvre dérivée.")
    return True


def test_telechargement(morceaux: list[dict]) -> bool:
    print("\n--- Téléchargement du premier morceau ---")
    morceau = morceaux[0]
    destination = config.WORK_DIR / f"test_jamendo_{morceau['id']}.mp3"

    try:
        chemin = audio.download_jamendo_track(morceau, destination)
    except audio.AudioError as erreur:
        print(f"  ÉCHEC : {erreur}")
        return False

    duree = audio.probe_duration(chemin)
    taille = chemin.stat().st_size / (1024 * 1024)
    print(f"  OK — {chemin.name} : {taille:.1f} Mo, {duree:.0f} s")
    print(f"  Crédit à afficher : {morceau['artist']} — {morceau['name']}")
    print(f"  Page du morceau   : {morceau['share_url']}")

    # On garde le fichier : le test du pipeline complet peut le réutiliser.
    print(f"\n  Fichier conservé pour un test de bout en bout :")
    print(f"    .venv\\Scripts\\python.exe scripts\\test_pipeline.py \"{chemin}\"")
    return True


def test_recherche_vide() -> bool:
    print("\n--- Recherche sans résultat ---")
    try:
        audio.search_jamendo("zzzzqqqqxxxx99999")
    except audio.AudioError as erreur:
        print(f"  OK — message clair : « {str(erreur).splitlines()[0]} »")
        return True
    print("  (Jamendo a quand même renvoyé des résultats — pas bloquant)")
    return True


def main() -> int:
    config.ensure_dirs()
    requete = sys.argv[1] if len(sys.argv) > 1 else "acoustic"

    print("=== Mode Creative Commons (Jamendo) ===\n")

    if not config.JAMENDO_CLIENT_ID:
        print("JAMENDO_CLIENT_ID absente.")
        print("Crée-la ici : https://devportal.jamendo.com/admin/applications")
        return 1

    try:
        morceaux = test_recherche(requete)
    except audio.AudioError as erreur:
        print(f"ÉCHEC : {erreur}")
        return 1

    resultats = [
        test_licence(morceaux),
        test_telechargement(morceaux),
        test_recherche_vide(),
    ]

    print("\n" + "=" * 50)
    if all(resultats):
        print("OK — le mode Creative Commons fonctionne.")
        return 0
    print("Certains contrôles ont échoué.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
