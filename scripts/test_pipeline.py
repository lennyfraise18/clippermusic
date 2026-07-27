"""Test du pipeline complet, de l'audio à la vidéo (étape 8).

Nécessite au moins une clé de banque vidéo (Pexels ou Pixabay).

Contrôles automatiques à l'arrivée :
  - la vidéo existe, fait bien 1080x1920 et a une piste audio ;
  - sa durée correspond à celle de l'extrait retenu ;
  - AUCUN clip de fond n'a été utilisé deux fois ;
  - le dossier de travail a été nettoyé.

Le script extrait aussi trois images pour la vérification à l'oeil : c'est le
seul moyen de constater que les paroles s'affichent avec la bonne police et au
bon moment.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_pipeline.py
    .venv\\Scripts\\python.exe scripts\\test_pipeline.py mon_fichier.mp3
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import config, pipeline, videos  # noqa: E402


def infos_video(chemin: Path) -> dict:
    resultat = subprocess.run(
        [config.ffprobe_path(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(chemin)],
        capture_output=True,
    )
    donnees = json.loads(resultat.stdout)
    video = next(s for s in donnees["streams"] if s["codec_type"] == "video")
    audio = next((s for s in donnees["streams"] if s["codec_type"] == "audio"), None)
    return {
        "largeur": video["width"],
        "hauteur": video["height"],
        "fps": video.get("r_frame_rate", "?"),
        "duree": float(donnees["format"]["duration"]),
        "audio": bool(audio),
        "taille_mo": chemin.stat().st_size / (1024 * 1024),
    }


def extraire_images(chemin: Path, duree: float) -> list[Path]:
    """Trois images réparties dans la vidéo, pour le contrôle visuel."""
    images = []
    for numero, instant in enumerate([duree * 0.15, duree * 0.5, duree * 0.8], start=1):
        image = chemin.with_name(f"{chemin.stem}_image{numero}.png")
        subprocess.run(
            [config.ffmpeg_path(), "-y", "-v", "error", "-ss", f"{instant:.2f}",
             "-i", str(chemin), "-frames:v", "1", "-vf", "scale=540:-1", str(image)],
            capture_output=True,
        )
        if image.exists():
            images.append(image)
    return images


def main() -> int:
    config.ensure_dirs()

    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "test_song.mp3"
    if not source.exists():
        print(f"Fichier introuvable : {source}")
        print("Lance d'abord : python scripts/fetch_test_audio.py")
        return 1

    if not videos.has_any_key():
        print(videos.missing_key_message())
        print()

    print("=== Pipeline complet ===")
    print(f"Entrée   : {source.name}")
    print(f"Modèle   : {config.WHISPER_MODEL}")
    print(f"Encodeur : {config.detect_video_encoder()}\n")

    # Instantané du dossier de travail AVANT le traitement : on ne peut pas
    # exiger qu'il soit vide (d'autres tests y déposent leurs fichiers), on
    # exige seulement que le pipeline ne laisse rien de NOUVEAU derrière lui.
    avant = {chemin.name for chemin in config.WORK_DIR.iterdir()}

    try:
        resultat = pipeline.generate_clip(
            source, progress=lambda message: print(f"  → {message}")
        )
    except pipeline.PipelineError as erreur:
        print(f"\nÉCHEC : {erreur}")
        return 1

    chemin = resultat["video"]
    infos = infos_video(chemin)

    print(f"\n--- Résultat ---")
    print(f"  fichier   : {chemin}")
    print(f"  format    : {infos['largeur']}x{infos['hauteur']} @ {infos['fps']} fps")
    print(f"  durée     : {infos['duree']:.1f} s (attendu {resultat['duration']:.1f} s)")
    print(f"  taille    : {infos['taille_mo']:.1f} Mo")
    print(f"  plans     : {resultat['shots']}")
    print(f"  langue    : {resultat['language']}")
    print(f"  traitement: {resultat['seconds']:.0f} s")
    print(f"  crédits   : {', '.join(resultat['credits'])}")
    if resultat["warnings"]:
        print(f"  avertissements :")
        for avertissement in resultat["warnings"]:
            print(f"    - {avertissement}")

    print("\n--- Paroles transcrites ---")
    for ligne in resultat["lyrics"].splitlines():
        print(f"  {ligne}")

    print("\n--- Contrôles ---")
    problemes = []

    if (infos["largeur"], infos["hauteur"]) != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
        problemes.append(
            f"format {infos['largeur']}x{infos['hauteur']} au lieu de "
            f"{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}"
        )
    if not infos["audio"]:
        problemes.append("pas de piste audio")
    if abs(infos["duree"] - resultat["duration"]) > 2.0:
        problemes.append(
            f"durée {infos['duree']:.1f} s au lieu de {resultat['duration']:.1f} s"
        )

    # Anti-répétition : on compare les identifiants de clips, pas les crédits.
    # Wikimedia renvoie « Wikimedia Commons » pour tous ses clips, donc compter
    # les auteurs distincts ne prouverait rien.
    identifiants = resultat["clip_ids"]
    doublons = len(identifiants) - len(set(identifiants))
    if doublons:
        repetes = [i for i in set(identifiants) if identifiants.count(i) > 1]
        problemes.append(
            f"{doublons} plan(s) réutilisent un clip déjà vu : {repetes[:3]}"
        )
    else:
        print(f"  OK — {len(identifiants)} plans, {len(set(identifiants))} clips "
              f"distincts : aucune répétition.")

    apres = {chemin.name for chemin in config.WORK_DIR.iterdir()}
    restes = apres - avant
    if restes:
        problemes.append(f"le pipeline a laissé des fichiers derrière lui : {restes}")

    if problemes:
        for probleme in problemes:
            print(f"  ÉCHEC — {probleme}")
    else:
        print("  OK — format, durée, audio et nettoyage conformes.")

    images = extraire_images(chemin, infos["duree"])
    print("\n--- À vérifier À L'OEIL ---")
    print(f"  Ouvre la vidéo : {chemin}")
    print("  Points à contrôler :")
    print("    • les paroles s'affichent (pas de carrés vides)")
    print("    • elles sont synchronisées avec le chant")
    print("    • le mot en cours change de couleur")
    print("    • aucun plan de fond ne revient deux fois")
    for image in images:
        print(f"  image : {image}")

    return 1 if problemes else 0


if __name__ == "__main__":
    raise SystemExit(main())
