"""Test isolé du montage + arbitrage entre les deux approches (étape 7).

Le module de montage est testé SANS dépendre des banques vidéo : les clips
sources sont fabriqués par ffmpeg dans trois formats différents (paysage,
portrait, carré) et à trois cadences différentes. C'est justement ce qui
permet de vérifier ce qui casse en vrai :
  - un clip paysage doit être recadré sans bande noire ;
  - un clip 24 fps et un clip 30 fps doivent se concaténer sans saccade ;
  - un clip plus court que le plan doit être bouclé.

Le script mesure ensuite les deux approches de montage sur le même cas, pour
trancher par la mesure et pas par principe.

Lancement :
    .venv\\Scripts\\python.exe scripts\\test_montage.py
    .venv\\Scripts\\python.exe scripts\\test_montage.py --complet   (les deux approches)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import config, editor, keywords, subtitles, videos  # noqa: E402

# Formats volontairement hétérogènes : c'est le cas réel d'une banque de vidéos.
FORMATS = [
    ("paysage", 1920, 1080, 30, 8),
    ("portrait", 1080, 1920, 25, 6),
    ("carre", 720, 720, 24, 4),
    ("petit", 640, 360, 30, 2),      # plus court que la plupart des plans : sera bouclé
    ("hd_portrait", 720, 1280, 30, 7),
]


def fabriquer_clips(work_dir: Path) -> list[dict]:
    """Génère des clips de test avec ffmpeg. Renvoie leurs caractéristiques."""
    clips = []
    for index, (nom, largeur, hauteur, fps, duree) in enumerate(FORMATS):
        nom_fichier = f"source_{nom}.mp4"
        commande = [
            config.ffmpeg_path(), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc2=size={largeur}x{hauteur}:rate={fps}:duration={duree}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            nom_fichier,
        ]
        subprocess.run(commande, cwd=str(work_dir), check=True, capture_output=True)
        clips.append({"name": nom_fichier, "duration": float(duree)})
        print(f"  clip {index + 1} : {nom_fichier} "
              f"({largeur}x{hauteur}, {fps} fps, {duree} s)")
    return clips


def charger_segments() -> tuple[list[dict], float, float]:
    """Reprend la transcription du fichier de test, ou fabrique un exemple."""
    cache = ROOT / "assets" / "test_transcription.json"
    if cache.exists():
        donnees = json.loads(cache.read_text(encoding="utf-8"))
        segments = keywords.build_queries(donnees["segments"], donnees["language"])
        return segments, donnees["duration"], donnees["start_offset"]

    print("  (pas de transcription en cache — segments d'exemple)")
    faux = []
    for index in range(6):
        debut = index * 4.0
        mots = [
            {"text": mot, "start": debut + position * 0.5,
             "end": debut + position * 0.5 + 0.45, "confidence": 0.9}
            for position, mot in enumerate(["test", "de", "montage", "numero", str(index)])
        ]
        faux.append({"text": " ".join(m["text"] for m in mots), "start": debut,
                     "end": debut + 3.0, "confidence": 0.9, "words": mots,
                     "query": "abstract", "keyword": "test"})
    return faux, 24.0, 0.0


def preparer(work_dir: Path) -> tuple[list[dict], str, str]:
    """Prépare tout ce qu'il faut pour lancer un montage."""
    print("\n--- Fabrication des clips sources ---")
    clips = fabriquer_clips(work_dir)

    print("\n--- Préparation de la timeline ---")
    segments, duree, debut = charger_segments()
    plans = editor.plan_shots(segments, duree)
    print(f"  {len(segments)} segments de paroles → {len(plans)} plans")

    # Attribution en rotation : ici on teste le montage, pas la recherche.
    for index, plan in enumerate(plans):
        clip = clips[index % len(clips)]
        besoin = plan["end"] - plan["start"]
        plan["clip_name"] = clip["name"]
        plan["clip_duration"] = clip["duration"]
        plan["clip_offset"] = videos.pick_start_offset(clip["duration"], besoin)

    durees = [p["end"] - p["start"] for p in plans]
    print(f"  durée des plans : min {min(durees):.1f} s, max {max(durees):.1f} s, "
          f"total {sum(durees):.1f} s")

    print("\n--- Sous-titres karaoké ---")
    chemin_ass = subtitles.build_ass(segments, work_dir / "subs.ass")
    lignes = chemin_ass.read_text(encoding="utf-8").count("Dialogue:")
    print(f"  {lignes} lignes de karaoké écrites dans {chemin_ass.name}")

    print("\n--- Bande son ---")
    source_audio = ROOT / "assets" / "test_song.mp3"
    if source_audio.exists():
        editor.extract_audio_segment(source_audio, work_dir, "track.m4a", debut, duree)
        print(f"  extrait de {duree:.1f} s à partir de {debut:.1f} s")
    else:
        subprocess.run(
            [config.ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi",
             "-i", f"sine=frequency=440:duration={duree}",
             "-c:a", "aac", "track.m4a"],
            cwd=str(work_dir), check=True, capture_output=True,
        )
        print("  (pas de MP3 de test — bip de remplacement)")

    return plans, "track.m4a", chemin_ass.name


def mesurer(approche: str, plans, audio_nom, ass_nom, work_dir: Path) -> dict:
    """Lance un montage et mesure le temps et la taille du résultat."""
    sortie = f"resultat_{approche}.mp4"
    debut = time.time()
    chemin = editor.render(
        plans, audio_nom, ass_nom, sortie, work_dir,
        approach=approche, progress=lambda m: print(f"    {m}"),
    )
    duree = time.time() - debut

    infos = subprocess.run(
        [config.ffprobe_path(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(chemin)],
        capture_output=True,
    )
    donnees = json.loads(infos.stdout)
    flux_video = next(s for s in donnees["streams"] if s["codec_type"] == "video")
    flux_audio = next((s for s in donnees["streams"] if s["codec_type"] == "audio"), None)

    return {
        "approche": approche,
        "secondes": duree,
        "taille_mo": chemin.stat().st_size / (1024 * 1024),
        "largeur": flux_video["width"],
        "hauteur": flux_video["height"],
        "duree_video": float(donnees["format"]["duration"]),
        "audio": bool(flux_audio),
        "chemin": chemin,
    }


def main() -> int:
    complet = "--complet" in sys.argv

    config.ensure_dirs()
    work_dir = config.WORK_DIR / "test_montage"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=== Test du module de montage ===")
    print(f"Encodeur : {config.detect_video_encoder()}")

    try:
        plans, audio_nom, ass_nom = preparer(work_dir)

        resultats = []
        print("\n--- Approche B : une seule passe ---")
        resultats.append(mesurer("single", plans, audio_nom, ass_nom, work_dir))

        if complet:
            print("\n--- Approche A : deux passes ---")
            resultats.append(mesurer("two_pass", plans, audio_nom, ass_nom, work_dir))

        print("\n" + "=" * 62)
        print(f"{'approche':<12} {'temps':>8} {'taille':>9} {'format':>12} {'durée':>8} {'audio':>6}")
        for resultat in resultats:
            print(f"{resultat['approche']:<12} {resultat['secondes']:>7.1f}s "
                  f"{resultat['taille_mo']:>8.1f}Mo "
                  f"{resultat['largeur']}x{resultat['hauteur']:>4} "
                  f"{resultat['duree_video']:>7.1f}s "
                  f"{'oui' if resultat['audio'] else 'NON':>6}")

        if len(resultats) == 2:
            rapide = min(resultats, key=lambda r: r["secondes"])
            ecart = abs(resultats[0]["secondes"] - resultats[1]["secondes"])
            pourcent = 100 * ecart / max(r["secondes"] for r in resultats)
            print(f"\nPlus rapide : {rapide['approche']} "
                  f"(écart {ecart:.1f} s, soit {pourcent:.0f} %)")

        # Vérifications automatiques
        print("\n--- Contrôles ---")
        problemes = 0
        for resultat in resultats:
            if (resultat["largeur"], resultat["hauteur"]) != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
                print(f"  ÉCHEC {resultat['approche']} : format "
                      f"{resultat['largeur']}x{resultat['hauteur']} au lieu de "
                      f"{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}")
                problemes += 1
            if not resultat["audio"]:
                print(f"  ÉCHEC {resultat['approche']} : pas de piste audio")
                problemes += 1

        if problemes == 0:
            print("  OK — format vertical correct et bande son présente.")

        print("\nVidéos à vérifier À L'OEIL (paroles lisibles, pas de carrés vides) :")
        for resultat in resultats:
            print(f"  {resultat['chemin']}")
        print("\n(le dossier de travail n'est PAS nettoyé ici, exprès, "
              "pour que tu puisses regarder le résultat)")

        return 1 if problemes else 0

    except editor.EditError as erreur:
        print(f"\nÉCHEC du montage :\n{erreur}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
