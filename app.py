"""Interface Gradio du générateur de clips paroles.

Une seule application : l'interface, le pipeline et les fichiers vivent dans le
même processus, hébergé sur un Space Hugging Face gratuit. Pas de frontend
séparé, pas de backend séparé, pas de stockage cloud.

Deux modes assumés, et c'est le coeur du propos du projet :

  • Mode démo — l'utilisateur envoie son propre MP3. Techniquement identique,
    mais juridiquement risqué : un avertissement doit être accepté avant de
    lancer la génération.

  • Mode Creative Commons — le morceau vient de Jamendo, sous licence CC.
    La vidéo produite est diffusable publiquement sans risque.

Lancement en local :
    .venv\\Scripts\\python.exe app.py
"""

import os
import sys
import traceback
from pathlib import Path

import gradio as gr

from modules import audio, config, pipeline, videos

# --- Textes de l'interface ---------------------------------------------------

TITRE = "🎬 Générateur de clips paroles"

INTRODUCTION = (
    "Transforme une chanson en vidéo verticale avec les paroles en karaoké, "
    "illustrée automatiquement."
)

# Version courte affichée d'emblée : une phrase, un lien vers le détail.
AVERTISSEMENT_COURT = (
    "⚠️ **Musique protégée.** La vidéo sera générée, mais ne la publie pas sur "
    "les réseaux : elle serait bloquée automatiquement, et tu t'exposerais à un "
    "risque juridique. Garde ce mode pour un usage privé ou une démonstration."
)

# Détail replié : disponible pour qui veut comprendre, invisible pour les autres.
AVERTISSEMENT_DETAIL = """
**Le blocage automatique.** Content ID (YouTube) et les systèmes équivalents
d'Instagram et TikTok reconnaissent un enregistrement en quelques secondes, même
ralenti, coupé ou recouvert de voix. La vidéo est retirée, coupée du son, ou
monétisée au profit de l'ayant droit.

**Le risque juridique.** Deux droits distincts s'appliquent : celui de la
**composition** (auteur, compositeur, éditeur) et celui de l'**enregistrement**
(le master, détenu par le label). Utiliser un extrait sans licence porte atteinte
aux deux, et la courte durée n'est pas une exception en droit français.

Pour une vidéo réellement publiable, utilise le mode **Creative Commons**.
"""

EXPLICATION_CC = (
    "✅ **Musique libre de droits.** Ces morceaux viennent de "
    "[Jamendo](https://www.jamendo.com) et sont diffusables partout, à condition "
    "de créditer l'artiste — le crédit s'affiche avec ta vidéo, prêt à recopier."
)

EXPLICATION_CC_DETAIL = """
« Creative Commons » ne veut pas dire « tout est permis ». Les licences **ND**
(*No Derivatives*) autorisent le partage du morceau tel quel mais **interdisent
d'en tirer une œuvre dérivée** — exactement ce que fait cette application.
Environ un tiers du catalogue Jamendo est concerné : ces morceaux sont
automatiquement écartés de la recherche.

Les vidéos de fond viennent de Pexels et Pixabay, dont les licences autorisent
explicitement l'intégration dans une œuvre dérivée.
"""


# --- Fonctions branchées sur l'interface ------------------------------------


def chercher_jamendo(recherche: str):
    """Cherche des morceaux Creative Commons et remplit la liste déroulante."""
    try:
        morceaux = audio.search_jamendo(recherche)
    except audio.AudioError as erreur:
        return (
            gr.update(choices=[], value=None),
            [],
            f"❌ {erreur}",
        )

    libelles = [audio.format_track_label(morceau) for morceau in morceaux]
    return (
        gr.update(choices=libelles, value=libelles[0]),
        morceaux,
        f"✅ {len(morceaux)} morceaux trouvés. Choisis-en un ci-dessous.",
    )


def _preparer_audio_jamendo(morceaux: list, libelle_choisi: str) -> Path:
    """Télécharge le morceau Jamendo sélectionné et renvoie son chemin."""
    if not morceaux or not libelle_choisi:
        raise pipeline.PipelineError(
            "Aucun morceau sélectionné. Lance d'abord une recherche, "
            "puis choisis un titre dans la liste."
        )

    for morceau in morceaux:
        if audio.format_track_label(morceau) == libelle_choisi:
            config.ensure_dirs()
            destination = config.WORK_DIR / f"jamendo_{morceau['id']}.mp3"
            return audio.download_jamendo_track(morceau, destination), morceau

    raise pipeline.PipelineError("Morceau introuvable. Relance la recherche.")


def generer(
    mode: str,
    fichier_upload: str | None,
    morceaux_jamendo: list,
    libelle_jamendo: str,
    avertissement_accepte: bool,
    modele: str,
    langue: str,
    progress=gr.Progress(),
):
    """Point d'entrée unique du bouton « Générer ».

    Renvoie : (vidéo, paroles, message, crédits).
    """
    etapes = {"n": 0}

    def avancement(message: str) -> None:
        # On n'a pas de progression exacte (le temps dépend de la chanson) :
        # une progression qui avance par paliers vaut mieux qu'une barre figée.
        etapes["n"] = min(etapes["n"] + 1, 40)
        progress(etapes["n"] / 45, desc=message)

    # Fichier temporaire à supprimer à la fin (morceau Jamendo téléchargé).
    # Sans ça, chaque recherche laisse un MP3 sur le disque du Space, qui sature
    # au bout de quelques dizaines de générations.
    fichier_temporaire: Path | None = None

    try:
        credit_musique = ""

        if mode == "Creative Commons (Jamendo)":
            progress(0.01, desc="Téléchargement du morceau Creative Commons…")
            chemin_audio, morceau = _preparer_audio_jamendo(
                morceaux_jamendo, libelle_jamendo
            )
            fichier_temporaire = chemin_audio
            # Le crédit doit être recopiable tel quel dans la description du
            # post : nom, titre, licence exacte et lien vers le morceau.
            licence = (morceau.get("license") or "").replace(
                "http://creativecommons.org/licenses/", "CC BY-"
            ).rstrip("/").upper()
            credit_musique = (
                f"🎵 **Musique :** {morceau['artist']} — *{morceau['name']}*"
            )
            if licence:
                credit_musique += f" — licence {licence}"
            if morceau.get("share_url"):
                credit_musique += f"\n\n{morceau['share_url']}"
            credit_musique += (
                "\n\n*Recopie ce bloc dans la description de ta publication : "
                "la licence Creative Commons impose de créditer l'artiste.*"
            )
        else:
            if not fichier_upload:
                return None, "", "❌ Envoie d'abord un fichier audio.", ""
            if not avertissement_accepte:
                return (
                    None, "",
                    "❌ Coche la case d'avertissement avant de lancer la génération.",
                    "",
                )
            chemin_audio = Path(fichier_upload)
            credit_musique = "🎵 **Musique :** fichier fourni par l'utilisateur."

        resultat = pipeline.generate_clip(
            chemin_audio,
            model_name=modele,
            language=None if langue == "détection automatique" else langue,
            progress=avancement,
        )

        progress(1.0, desc="Terminé.")

        message = (
            f"✅ Clip généré en {resultat['seconds']:.0f} s — "
            f"{resultat['shots']} plans, {resultat['duration']:.0f} s de vidéo, "
            f"langue détectée : {resultat['language']}."
        )
        if resultat["warnings"]:
            message += "\n\n⚠️ " + " • ".join(resultat["warnings"][:3])

        credits = credit_musique + "\n\n🎥 **Vidéos de fond :** " + ", ".join(
            resultat["credits"]
        )
        if mode != "Creative Commons (Jamendo)":
            credits += (
                "\n\n⚠️ *Cette vidéo utilise un enregistrement protégé : "
                "ne la diffuse pas publiquement.*"
            )

        return str(resultat["video"]), resultat["lyrics"], message, credits

    except pipeline.PipelineError as erreur:
        return None, "", f"❌ {erreur}", ""
    except Exception as erreur:  # filet de sécurité : jamais de trace brute à l'écran
        traceback.print_exc()
        return None, "", f"❌ Erreur inattendue : {erreur}", ""

    finally:
        # Le morceau Jamendo a servi, la vidéo est montée : on le supprime.
        # Dans un `finally`, donc y compris quand la génération a échoué.
        if fichier_temporaire is not None:
            try:
                fichier_temporaire.unlink(missing_ok=True)
            except OSError:
                pass


def basculer_mode(mode: str):
    """Affiche le bloc correspondant au mode choisi."""
    est_demo = mode != "Creative Commons (Jamendo)"
    return gr.update(visible=est_demo), gr.update(visible=not est_demo)


# --- Construction de l'interface ---------------------------------------------


def construire_interface() -> gr.Blocks:
    # Note : Gradio 5 avertit que `theme` passera dans launch() en version 6,
    # mais launch() ne l'accepte pas encore. On reste sur Blocks().
    with gr.Blocks(title="Générateur de clips paroles", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITRE}")
        gr.Markdown(INTRODUCTION)

        etat_jamendo = gr.State([])

        mode = gr.Radio(
            choices=["Creative Commons (Jamendo)", "Mon fichier (démo)"],
            value="Creative Commons (Jamendo)",
            label="Ta musique",
            info="Creative Commons = vidéo publiable. Mon fichier = privé seulement.",
        )

        # --- Bloc Creative Commons ---
        with gr.Group(visible=True) as bloc_cc:
            gr.Markdown(EXPLICATION_CC)
            with gr.Accordion("Pourquoi certains morceaux sont écartés", open=False):
                gr.Markdown(EXPLICATION_CC_DETAIL)
            with gr.Row():
                recherche = gr.Textbox(
                    label="Chercher un style de musique",
                    placeholder="pop, rock, acoustic, chanson, hip hop…",
                    info="Un style ou une ambiance — pas un lien YouTube.",
                    scale=3,
                )
                bouton_recherche = gr.Button("Chercher", variant="secondary", scale=1)
            # allow_custom_value : sans lui, Gradio refuse toute valeur qui
            # n'est pas dans `choices`. Or en mode démo la liste est vide et
            # jamais remplie — le bouton « Générer » serait rejeté avant même
            # d'atteindre notre code.
            choix_morceau = gr.Dropdown(
                choices=[], label="Morceaux trouvés",
                interactive=True, allow_custom_value=True,
            )
            # Ces zones de message partent avec un contenu : un gr.Markdown
            # initialisé à "" occupe une hauteur nulle, et la première mise à
            # jour passe alors facilement inaperçue.
            message_recherche = gr.Markdown(
                "*Lance une recherche pour voir les morceaux disponibles.*"
            )

        # --- Bloc démo ---
        with gr.Group(visible=False) as bloc_demo:
            gr.Markdown(AVERTISSEMENT_COURT)
            with gr.Accordion("Pourquoi ce n'est pas publiable", open=False):
                gr.Markdown(AVERTISSEMENT_DETAIL)
            fichier = gr.Audio(
                label="Ton fichier audio (MP3, WAV, M4A…)",
                type="filepath",
                sources=["upload"],
            )
            avertissement = gr.Checkbox(
                label="J'ai compris : usage privé ou démonstration seulement.",
                value=False,
            )

        with gr.Accordion("Réglages avancés", open=False):
            with gr.Row():
                modele = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium"],
                    value=config.WHISPER_MODEL,
                    label="Modèle de transcription",
                    info="Plus gros = plus précis mais plus lent. « small » est "
                         "le bon compromis sur processeur.",
                )
                langue = gr.Dropdown(
                    choices=["détection automatique", "fr", "en", "es", "de", "it"],
                    value="détection automatique",
                    label="Langue des paroles",
                    info="Forcer la langue évite les erreurs sur les morceaux "
                         "qui mélangent deux langues.",
                )

        bouton_generer = gr.Button("🎬 Générer le clip", variant="primary", size="lg")

        message = gr.Markdown("*Compte une à trois minutes de traitement.*")
        video = gr.Video(label="Ton clip", height=520)
        credits = gr.Markdown("")

        with gr.Accordion("Paroles transcrites", open=False):
            paroles = gr.Textbox(
                label="", lines=8, show_copy_button=True, container=False
            )

        gr.Markdown(
            "<sub>Transcription Whisper · thèmes visuels spaCy · "
            "vidéos Pexels & Pixabay · montage ffmpeg</sub>"
        )

        # --- Branchements ---
        mode.change(basculer_mode, inputs=mode, outputs=[bloc_demo, bloc_cc])

        bouton_recherche.click(
            chercher_jamendo,
            inputs=recherche,
            outputs=[choix_morceau, etat_jamendo, message_recherche],
        )
        recherche.submit(
            chercher_jamendo,
            inputs=recherche,
            outputs=[choix_morceau, etat_jamendo, message_recherche],
        )

        bouton_generer.click(
            generer,
            inputs=[mode, fichier, etat_jamendo, choix_morceau,
                    avertissement, modele, langue],
            outputs=[video, paroles, message, credits],
        )

    return demo


if __name__ == "__main__":
    config.ensure_dirs()

    # Un avertissement au démarrage vaut mieux qu'une erreur au bout de
    # trois minutes de traitement.
    if not videos.has_any_key():
        print("ATTENTION : " + videos.missing_key_message())

    # Lien public temporaire, activé par l'argument --share :
    #     .venv\Scripts\python.exe app.py --share
    #
    # Sur un Space Hugging Face, l'adresse publique vient de la plateforme et
    # cette option reste inutile. En local, elle crée un tunnel *.gradio.live
    # valable environ une semaine — de quoi montrer la démo à distance sans
    # dépendre d'un quota d'hébergement.
    #
    # La variable d'environnement GRADIO_SHARE reste acceptée, mais l'argument
    # est plus simple : il n'a pas de syntaxe différente entre CMD et PowerShell.
    partager = (
        "--share" in sys.argv
        or os.getenv("GRADIO_SHARE", "").strip() in {"1", "true", "yes"}
    )

    print("\n" + "=" * 58)
    print("  Ouvre cette adresse dans ton navigateur :")
    print("      http://localhost:7860")
    print()
    if partager:
        print("  Un lien public https://....gradio.live va aussi s'afficher")
        print("  juste en dessous. Il fonctionne tant que ce terminal reste")
        print("  ouvert ; ferme-le pour couper l'accès.")
    else:
        print("  Pour obtenir en plus un lien public partageable, relance avec :")
        print("      .venv\\Scripts\\python.exe app.py --share")
    print("=" * 58 + "\n")

    construire_interface().queue(max_size=8).launch(
        # 0.0.0.0 = « écoute sur toutes les interfaces », nécessaire dans un
        # conteneur. Ce n'est PAS une adresse à taper dans un navigateur :
        # côté utilisateur, c'est http://localhost:7860 (voir le message ci-dessus).
        server_name="0.0.0.0",
        server_port=7860,
        share=partager,
        show_error=True,
    )
