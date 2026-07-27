"""ClipperMusic — interface Gradio du générateur de clips paroles.

Une seule application : l'interface, le pipeline et les fichiers vivent dans le
même processus, hébergé sur un Space Hugging Face gratuit.

Parcours utilisateur, volontairement réduit à trois gestes :
    1. déposer un fichier audio, ou coller un lien YouTube / Spotify / Deezer ;
    2. cliquer sur un bouton ;
    3. télécharger l'edit vertical.

Le reste (choix du modèle, langue, options de montage) est replié : personne
n'a besoin de savoir ce qu'est un modèle Whisper pour se servir de l'outil.

Lancement en local :
    .venv\\Scripts\\python.exe app.py
    .venv\\Scripts\\python.exe app.py --share    (lien public temporaire)
"""

import os
import sys
import traceback
from pathlib import Path

import gradio as gr

from modules import audio, config, liens, pipeline, transcribe, videos

# --- Textes de l'interface ---------------------------------------------------

TITRE = "🎬 ClipperMusic"

INTRODUCTION = (
    "Dépose une chanson, récupère un edit vertical avec les paroles en karaoké "
    "et des visuels choisis automatiquement.<br>"
    "<sub>Transcription et montage entièrement automatiques — aucune "
    "intervention manuelle.</sub>"
)

# Affiché quand l'utilisateur colle un lien YouTube / Spotify / Deezer.
# Ton factuel : on explique la méthode, on ne fait pas la morale.
EXPLICATION_LIEN = """
**Ton morceau est identifié, mais son audio n'est pas récupéré depuis la
plateforme — et c'est à ton avantage.**

Une musique intégrée au fichier vidéo se fait couper : Content ID (YouTube) et
ses équivalents sur Instagram et TikTok reconnaissent un enregistrement en
quelques secondes, même ralenti ou recouvert de voix. Son coupé, vidéo retirée,
ou vues monétisées au profit du label.

**La méthode des comptes d'edits qui tiennent :** exporter la vidéo **sans
musique**, la publier, puis ajouter le son depuis la bibliothèque de TikTok ou
d'Instagram. Ces plateformes ont des accords de licence avec les labels : le son
y est autorisé et n'est jamais coupé.

Deux façons de continuer :
- **tu as le fichier audio du morceau** → dépose-le ci-dessus, coche *Sans
  musique*, et tu auras un edit calé sur les vraies paroles ;
- **tu ne l'as pas** → choisis un des morceaux libres proposés ci-dessous.
"""

AIDE_SANS_MUSIQUE = """
**Coché** — la vidéo sort **muette** (visuels + paroles). Tu la postes sur
TikTok ou Instagram, puis tu ajoutes la musique depuis leur bibliothèque
intégrée. C'est la méthode qui ne se fait jamais couper le son.

**Décoché** — la musique est intégrée au fichier. Pratique pour vérifier la
synchro ou garder la vidéo pour toi, mais à ne pas publier si le morceau est
protégé.
"""


# --- Fonctions branchées sur l'interface ------------------------------------


def analyser_entree(lien: str):
    """Réagit à un lien collé : identifie le morceau et propose des alternatives.

    Renvoie : (message, détail replié, liste déroulante, état, visibilité du bloc)
    """
    lien = (lien or "").strip()

    if not lien:
        return (
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
            [],
        )

    # Un mot-clé simple ("pop", "rock") : on cherche directement des morceaux libres.
    if not liens.est_un_lien(lien) and liens.detecter_plateforme(lien) is None:
        try:
            morceaux = audio.search_jamendo(lien)
        except audio.AudioError as erreur:
            return (
                gr.update(value=f"❌ {erreur}", visible=True),
                gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False),
                [],
            )
        libelles = [audio.format_track_label(m) for m in morceaux]
        return (
            gr.update(
                value=f"✅ {len(libelles)} morceaux libres de droits trouvés.",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(choices=libelles, value=libelles[0], visible=True),
            morceaux,
        )

    # Un lien de plateforme : on lit le titre, puis on propose des équivalents.
    try:
        infos = liens.lire_titre(lien)
    except liens.LienError as erreur:
        return (
            gr.update(value=f"❌ {erreur}", visible=True),
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
            [],
        )

    entete = f"🎧 Morceau identifié : **{infos['titre']}**"
    if infos["artiste"]:
        entete += f" — {infos['artiste']}"
    entete += f"  *(via {infos['plateforme']})*"

    try:
        morceaux = liens.chercher_equivalent_libre(infos)
    except liens.LienError:
        morceaux = []

    libelles = [audio.format_track_label(m) for m in morceaux]
    return (
        gr.update(value=entete, visible=True),
        gr.update(visible=True),
        gr.update(
            choices=libelles,
            value=libelles[0] if libelles else None,
            visible=bool(libelles),
        ),
        morceaux,
    )


def _preparer_morceau_libre(morceaux: list, libelle_choisi: str):
    """Télécharge le morceau libre sélectionné."""
    if not morceaux:
        raise pipeline.PipelineError(
            "Dépose un fichier audio, ou colle un lien pour qu'on te propose "
            "des morceaux libres de droits."
        )

    choisi = None
    for morceau in morceaux:
        if audio.format_track_label(morceau) == libelle_choisi:
            choisi = morceau
            break
    if choisi is None:
        choisi = morceaux[0]

    config.ensure_dirs()
    destination = config.WORK_DIR / f"jamendo_{choisi['id']}.mp3"
    return audio.download_jamendo_track(choisi, destination), choisi


def refaire(
    session: dict,
    paroles_corrigees: str,
    sans_musique: bool,
    progress=gr.Progress(),
):
    """Refait la vidéo à partir des paroles corrigées, avec de nouveaux visuels.

    Ne repasse pas par Whisper : la transcription de la première génération est
    réutilisée, seul le texte change. Le traitement dure une trentaine de
    secondes au lieu de plusieurs minutes.
    """
    if not session or not session.get("audio"):
        return None, "❌ Génère d'abord une première version.", "", session

    chemin_audio = Path(session["audio"])
    if not chemin_audio.exists():
        return (
            None,
            "❌ Le fichier audio n'est plus disponible. Redépose-le et relance.",
            "", session,
        )

    transcription = dict(session["transcription"])

    try:
        transcription["segments"] = transcribe.appliquer_texte_corrige(
            transcription["segments"], paroles_corrigees
        )
    except transcribe.TranscriptionError as erreur:
        return None, f"❌ {erreur}", "", session

    etapes = {"n": 0}

    def avancement(message: str) -> None:
        etapes["n"] = min(etapes["n"] + 1, 30)
        progress(etapes["n"] / 34, desc=message)

    try:
        resultat = pipeline.generate_clip(
            chemin_audio,
            progress=avancement,
            inclure_audio=not sans_musique,
            transcription_prete=transcription,
            clips_a_eviter=set(session.get("clips", [])),
        )
    except pipeline.PipelineError as erreur:
        return None, f"❌ {erreur}", "", session
    except Exception as erreur:
        traceback.print_exc()
        return None, f"❌ Erreur inattendue : {erreur}", "", session

    progress(1.0, desc="Terminé.")

    session["transcription"] = resultat["transcription"]
    session["clips"] = list(session.get("clips", [])) + resultat["clip_ids"]

    message = (
        f"✅ **Nouvelle version** — paroles corrigées, visuels renouvelés "
        f"({resultat['seconds']:.0f} s)."
    )
    credits = session.get("credit_musique", "") + "\n\n🎥 **Visuels :** " + ", ".join(
        resultat["credits"][:6]
    )
    if sans_musique:
        credits += (
            "\n\n🔇 **Vidéo sans musique.** Ajoute le son sur TikTok ou Instagram."
        )

    return str(resultat["video"]), message, credits, session


def generer(
    fichier_upload: str | None,
    morceaux_proposes: list,
    libelle_choisi: str,
    sans_musique: bool,
    modele: str,
    langue: str,
    progress=gr.Progress(),
):
    """Point d'entrée du bouton principal.

    Renvoie : (vidéo, message, crédits, paroles, session).
    """
    etapes = {"n": 0}

    def avancement(message: str) -> None:
        # Pas de progression exacte possible (le temps dépend de la chanson) :
        # une barre qui avance par paliers vaut mieux qu'une barre figée.
        etapes["n"] = min(etapes["n"] + 1, 40)
        progress(etapes["n"] / 45, desc=message)

    # Les morceaux téléchargés ne sont plus supprimés tout de suite : ils
    # doivent survivre pour permettre de corriger les paroles et refaire la
    # vidéo. Une purge horaire s'en charge (pipeline.purger_audios_temporaires).
    pipeline.purger_audios_temporaires()

    try:
        if fichier_upload:
            chemin_audio = Path(fichier_upload)
            credit_musique = "🎵 **Musique :** ton fichier."
            musique_protegee = True
        else:
            progress(0.01, desc="Téléchargement du morceau…")
            chemin_audio, morceau = _preparer_morceau_libre(
                morceaux_proposes, libelle_choisi
            )
            musique_protegee = False

            licence = (
                (morceau.get("license") or "")
                .replace("http://creativecommons.org/licenses/", "CC BY-")
                .rstrip("/")
                .upper()
            )
            credit_musique = f"🎵 **Musique :** {morceau['artist']} — *{morceau['name']}*"
            if licence:
                credit_musique += f" — {licence}"
            if morceau.get("share_url"):
                credit_musique += f"\n\n{morceau['share_url']}"
            credit_musique += (
                "\n\n*Recopie ce bloc dans ta description : la licence impose "
                "de créditer l'artiste.*"
            )

        resultat = pipeline.generate_clip(
            chemin_audio,
            model_name=modele,
            language=None if langue == "détection automatique" else langue,
            progress=avancement,
            inclure_audio=not sans_musique,
        )

        progress(1.0, desc="Terminé.")

        message = (
            f"✅ **Ton edit est prêt** — {resultat['duration']:.0f} s, "
            f"{resultat['shots']} plans, généré en {resultat['seconds']:.0f} s."
        )
        if resultat["warnings"]:
            message += "\n\n⚠️ " + " • ".join(resultat["warnings"][:2])

        credits = credit_musique + "\n\n🎥 **Visuels :** " + ", ".join(
            resultat["credits"][:6]
        )

        if sans_musique:
            credits += (
                "\n\n🔇 **Vidéo sans musique.** Poste-la sur TikTok ou Instagram, "
                "puis ajoute le son depuis leur bibliothèque : il ne sera pas coupé."
            )
        elif musique_protegee:
            credits += (
                "\n\n⚠️ **Musique intégrée au fichier.** Si le morceau est protégé, "
                "garde cette vidéo pour toi — coche *Sans musique* pour une "
                "version publiable."
            )

        # Mémorisé pour permettre « Corriger et refaire » sans retranscrire.
        session = {
            "audio": str(chemin_audio),
            "transcription": resultat["transcription"],
            "clips": resultat["clip_ids"],
            "credit_musique": credit_musique,
        }

        return str(resultat["video"]), message, credits, resultat["lyrics"], session

    except pipeline.PipelineError as erreur:
        return None, f"❌ {erreur}", "", "", {}
    except Exception as erreur:  # filet : jamais de trace Python à l'écran
        traceback.print_exc()
        return None, f"❌ Erreur inattendue : {erreur}", "", "", {}


# --- Construction de l'interface ---------------------------------------------

CSS = """
.gradio-container {max-width: 860px !important; margin: auto !important;}
#titre {text-align: center; margin-bottom: 0;}
#accroche {text-align: center; color: #6b7280; margin-top: 0.2rem;}
#bouton-principal {font-size: 1.1rem !important; padding: 0.9rem !important;}
"""


def construire_interface() -> gr.Blocks:
    # Note : Gradio 5 avertit que `theme` passera dans launch() en version 6,
    # mais launch() ne l'accepte pas encore. On reste sur Blocks().
    with gr.Blocks(
        title="ClipperMusic", theme=gr.themes.Soft(), css=CSS
    ) as demo:
        gr.Markdown(f"# {TITRE}", elem_id="titre")
        gr.Markdown(INTRODUCTION, elem_id="accroche")

        etat_morceaux = gr.State([])
        # Retient la dernière génération : fichier audio, transcription et
        # clips déjà utilisés. C'est ce qui permet de refaire la vidéo sans
        # repasser par Whisper.
        etat_session = gr.State({})

        # --- 1. La musique ---
        # Le dépôt de fichier est le geste principal : il occupe l'écran.
        # Chercher une musique libre est un cas secondaire, donc replié.
        fichier = gr.Audio(
            label="🎵 Dépose ton fichier audio ici",
            type="filepath",
            sources=["upload"],
        )

        with gr.Accordion(
            "Je n'ai pas de fichier — trouver une musique libre de droits",
            open=False,
        ):
            lien = gr.Textbox(
                label="Un style, ou un lien YouTube / Spotify / Deezer",
                placeholder="pop · rock · acoustic · ou colle un lien",
                info="On identifie le morceau et on propose des musiques "
                     "libres de droits utilisables directement.",
            )
            message_lien = gr.Markdown("", visible=False)

            with gr.Accordion(
                "Pourquoi le son ne vient pas du lien", open=False, visible=False
            ) as detail_lien:
                gr.Markdown(EXPLICATION_LIEN)

            choix_morceau = gr.Dropdown(
                choices=[],
                label="Morceaux libres de droits",
                interactive=True,
                allow_custom_value=True,
                visible=False,
            )

        # --- 2. L'option qui compte ---
        sans_musique = gr.Checkbox(
            label="🔇 Sans musique — recommandé pour poster sur TikTok / Instagram",
            value=False,
        )
        with gr.Accordion("À quoi sert cette option", open=False):
            gr.Markdown(AIDE_SANS_MUSIQUE)

        # --- 3. Le bouton ---
        bouton = gr.Button(
            "🎬 Créer mon edit", variant="primary", size="lg",
            elem_id="bouton-principal",
        )

        # --- 4. Le résultat ---
        message = gr.Markdown("*Compte une à trois minutes.*")
        video = gr.Video(label="Ton edit", height=520)
        credits = gr.Markdown("")

        with gr.Accordion("✏️ Corriger les paroles", open=False):
            gr.Markdown(
                "La transcription se trompe parfois, surtout sur l'argot et les "
                "noms propres. Corrige les mots ici, puis relance : les visuels "
                "sont renouvelés au passage.\n\n"
                "*Garde une ligne par phrase — n'en ajoute pas, n'en supprime pas.*"
            )
            paroles = gr.Textbox(
                label="", lines=10, show_copy_button=True,
                container=False, interactive=True,
            )
            bouton_refaire = gr.Button("🔄 Refaire avec ces paroles", variant="secondary")

        with gr.Accordion("Réglages avancés", open=False):
            with gr.Row():
                modele = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium"],
                    value=config.WHISPER_MODEL,
                    label="Précision de la transcription",
                    info="Plus élevé = plus précis, mais plus lent.",
                )
                langue = gr.Dropdown(
                    choices=["détection automatique", "fr", "en", "es", "de", "it"],
                    value="détection automatique",
                    label="Langue des paroles",
                )

        gr.Markdown(
            "<sub>Transcription Whisper · thèmes visuels spaCy · "
            "visuels Pexels & Pixabay · montage ffmpeg</sub>"
        )

        # --- Branchements ---
        lien.submit(
            analyser_entree,
            inputs=lien,
            outputs=[message_lien, detail_lien, choix_morceau, etat_morceaux],
        )
        lien.blur(
            analyser_entree,
            inputs=lien,
            outputs=[message_lien, detail_lien, choix_morceau, etat_morceaux],
        )

        bouton.click(
            generer,
            inputs=[fichier, etat_morceaux, choix_morceau, sans_musique,
                    modele, langue],
            outputs=[video, message, credits, paroles, etat_session],
        )

        bouton_refaire.click(
            refaire,
            inputs=[etat_session, paroles, sans_musique],
            outputs=[video, message, credits, etat_session],
        )

    return demo


if __name__ == "__main__":
    config.ensure_dirs()

    if not videos.has_any_key():
        print("ATTENTION : " + videos.missing_key_message())

    # Lien public temporaire, activé par --share.
    # Sur un Space Hugging Face, l'adresse publique vient de la plateforme et
    # cette option est inutile. En local, elle crée un tunnel *.gradio.live
    # valable environ une semaine.
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
        # conteneur. Ce n'est PAS une adresse à taper dans un navigateur.
        server_name="0.0.0.0",
        server_port=7860,
        share=partager,
        show_error=True,
    )
