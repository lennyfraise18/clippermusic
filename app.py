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


# --- Fonctions branchées sur l'interface ------------------------------------


def _signature_moteur() -> str:
    """Décrit le moteur de transcription réellement installé.

    Affiché en pied de page : c'est ce qui permet de vérifier à distance quelle
    version tourne réellement, sans accès aux logs de l'hébergeur.
    """
    try:
        import faster_whisper

        return (
            f"Transcription faster-whisper {faster_whisper.__version__} "
            f"(modèle {config.WHISPER_MODEL})"
        )
    except ImportError:
        return "Transcription Whisper"


def _diagnostic() -> str:
    """État de l'environnement d'exécution, affiché dans l'interface.

    Sert au diagnostic à distance : quand un traitement échoue sur un serveur
    dont on ne lit pas les logs, la mémoire allouée est la première chose à
    vérifier — c'est elle qui fait tuer le conteneur au chargement du modèle.
    """
    from modules import transcribe

    lignes = [f"- **Moteur** : {_signature_moteur()}"]

    memoire = config.memoire_disponible_mo()
    if memoire is None:
        lignes.append("- **Mémoire allouée** : non mesurable (hors conteneur Linux)")
    else:
        lignes.append(f"- **Mémoire allouée** : {memoire:.0f} Mo")

    retenu, avertissement = transcribe.modele_tenable(config.WHISPER_MODEL)
    if avertissement:
        lignes.append(f"- **Modèle réellement utilisé** : {retenu} ⚠️ {avertissement}")
    else:
        lignes.append(f"- **Modèle réellement utilisé** : {retenu}")

    lignes.append(f"- **Extrait analysé** : {config.MAX_TRANSCRIBE_SECONDS} s max")
    lignes.append(
        f"- **Clip produit** : {config.MAX_CLIP_SECONDS} s max, "
        f"{config.VIDEO_WIDTH}×{config.VIDEO_HEIGHT}"
    )

    try:
        encodeur = config.detect_video_encoder()
    except Exception as erreur:
        encodeur = f"aucun ({erreur})"
    lignes.append(f"- **Encodeur vidéo** : {encodeur}")

    cles = []
    if config.PEXELS_API_KEY:
        cles.append("Pexels")
    if config.PIXABAY_API_KEY:
        cles.append("Pixabay")
    if config.JAMENDO_CLIENT_ID:
        cles.append("Jamendo")
    lignes.append(f"- **Clés configurées** : {', '.join(cles) if cles else 'aucune'}")

    return "\n".join(lignes)


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

    # Lien direct vers un fichier audio : rien à identifier, on l'utilisera tel
    # quel au moment de générer.
    if liens.est_fichier_audio_direct(lien):
        return (
            gr.update(
                value="🎵 Fichier audio détecté. Clique sur **Créer mon edit**.",
                visible=True,
            ),
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
    lien: str,
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

        elif liens.est_fichier_audio_direct(lien or ""):
            progress(0.01, desc="Téléchargement du fichier audio…")
            config.ensure_dirs()
            nom = Path((lien or "").split("?")[0]).name or "audio_distant.mp3"
            chemin_audio = liens.telecharger_audio_direct(
                lien, config.WORK_DIR / f"jamendo_{nom}"
            )
            credit_musique = f"🎵 **Musique :** fichier récupéré depuis {lien}"
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

        return (
            str(resultat["video"]), message, credits, resultat["lyrics"],
            session, gr.update(visible=True),
        )

    except (pipeline.PipelineError, liens.LienError) as erreur:
        return None, f"❌ {erreur}", "", "", {}, gr.update(visible=False)
    except Exception as erreur:  # filet : jamais de trace Python à l'écran
        traceback.print_exc()
        return (
            None, f"❌ Erreur inattendue : {erreur}", "", "", {},
            gr.update(visible=False),
        )


# --- Construction de l'interface ---------------------------------------------

# Identité visuelle : clavier de piano sur fond nocturne, réveillé par des
# néons violet/magenta/cyan et un égaliseur qui bat en continu.
#
# Tout est en CSS pur — aucune image, aucune police externe, aucun script.
# C'est un choix contraint autant qu'esthétique : le conteneur bloque les
# requêtes sortantes, et une ressource qui ne charge pas casse la page.
CSS = """
:root {
    --violet: #a855f7;
    --magenta: #ec4899;
    --cyan: #22d3ee;
    --ambre: #fbbf24;
}

.gradio-container {
    max-width: 620px !important;
    margin: auto !important;
    padding: 0 0.6rem !important;
    background:
        radial-gradient(ellipse 70% 50% at 20% 0%, rgba(168,85,247,0.16), transparent 60%),
        radial-gradient(ellipse 60% 45% at 85% 8%, rgba(34,211,238,0.13), transparent 60%),
        radial-gradient(ellipse at top, #16161f 0%, #08080c 65%) !important;
}

/* --- Bandeau titre --- */
#entete {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 1.7rem 0.9rem 1.4rem;
    margin-bottom: 1rem;
    background: linear-gradient(160deg, #1a1a26 0%, #0b0b11 100%);
    border: 1px solid rgba(168,85,247,0.22);
    box-shadow: 0 0 60px rgba(168,85,247,0.10) inset,
                0 12px 40px rgba(0,0,0,0.5);
}

/* Halo coloré qui balaie lentement le bandeau, comme un projecteur de scène. */
#entete::before {
    content: "";
    position: absolute;
    inset: -50%;
    background: conic-gradient(from 0deg,
        transparent 0deg, rgba(168,85,247,0.16) 60deg,
        transparent 130deg, rgba(34,211,238,0.14) 200deg,
        transparent 280deg, rgba(236,72,153,0.14) 330deg, transparent 360deg);
    animation: balayage 14s linear infinite;
    pointer-events: none;
}
@keyframes balayage { to { transform: rotate(360deg); } }

#entete h1, #entete p, #entete .clavier, #entete .egaliseur {
    position: relative;
    z-index: 1;
}
#entete h1 {
    margin: 0;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background-image: linear-gradient(100deg, #ffffff 0%, #a855f7 42%,
                                      #ec4899 62%, #22d3ee 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 18px rgba(168,85,247,0.4));
}
#entete p {
    text-align: center;
    color: #9ba1ad;
    margin: 0.45rem auto 0;
    max-width: 26rem;
    font-size: 0.92rem;
    line-height: 1.45;
}

/* --- Égaliseur : des barres qui battent, comme sur une platine --- */
.egaliseur {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 4px;
    height: 30px;
    margin-top: 1rem;
}
.barre {
    width: 5px;
    border-radius: 3px;
    background-image: linear-gradient(180deg, #22d3ee, #a855f7 55%, #ec4899);
    box-shadow: 0 0 12px rgba(168,85,247,0.55);
    animation: battre 1.1s ease-in-out infinite;
}
@keyframes battre {
    0%, 100% { height: 18%; opacity: 0.65; }
    50%      { height: 100%; opacity: 1; }
}
.barre:nth-child(1) { animation-delay: 0.0s;  }
.barre:nth-child(2) { animation-delay: 0.18s; }
.barre:nth-child(3) { animation-delay: 0.36s; }
.barre:nth-child(4) { animation-delay: 0.12s; }
.barre:nth-child(5) { animation-delay: 0.48s; }
.barre:nth-child(6) { animation-delay: 0.26s; }
.barre:nth-child(7) { animation-delay: 0.55s; }
.barre:nth-child(8) { animation-delay: 0.08s; }
.barre:nth-child(9) { animation-delay: 0.40s; }

/* --- Clavier de piano sous le titre --- */
.clavier {
    display: flex;
    justify-content: center;
    gap: 2px;
    margin-top: 0.7rem;
    height: 26px;
}
.touche {
    width: 13px;
    border-radius: 0 0 4px 4px;
    background: linear-gradient(180deg, #ffffff 0%, #cfcfda 100%);
    animation: enfoncer 3.4s ease-in-out infinite;
}
.touche.noire {
    width: 11px;
    height: 62%;
    background: linear-gradient(180deg, #2e2e3c 0%, #0e0e16 100%);
    margin: 0 -7px;
    z-index: 2;
}
@keyframes enfoncer {
    0%, 88%, 100% { transform: translateY(0); opacity: 0.85; }
    92%           { transform: translateY(4px); opacity: 1; }
}
.touche:nth-child(2n)  { animation-delay: 0.4s; }
.touche:nth-child(3n)  { animation-delay: 0.9s; }
.touche:nth-child(5n)  { animation-delay: 1.6s; }
.touche:nth-child(7n)  { animation-delay: 2.3s; }

/* --- Notes qui montent en fond --- */
.note {
    position: absolute;
    bottom: -22px;
    color: rgba(255,255,255,0.16);
    font-size: 1.5rem;
    animation: monter 9s linear infinite;
    pointer-events: none;
}
@keyframes monter {
    0%   { transform: translateY(0) rotate(0deg);      opacity: 0; }
    12%  { opacity: 0.9; }
    80%  { opacity: 0.45; }
    100% { transform: translateY(-230px) rotate(22deg); opacity: 0; }
}
.note:nth-child(1) { left: 11%; animation-delay: 0s;   }
.note:nth-child(2) { left: 27%; animation-delay: 2.1s; font-size: 1.1rem; }
.note:nth-child(3) { left: 49%; animation-delay: 4.3s; }
.note:nth-child(4) { left: 71%; animation-delay: 1.2s; font-size: 1.9rem; }
.note:nth-child(5) { left: 87%; animation-delay: 6s;   font-size: 1.2rem; }

/* --- Bouton principal ---
   Sélecteur volontairement très spécifique : Gradio applique sa propre
   couleur de bouton primaire avec !important, et une règle simple sur
   #bouton-principal se fait écraser. */
.gradio-container button#bouton-principal,
.gradio-container #bouton-principal,
#bouton-principal {
    font-size: 1.18rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
    padding: 1.05rem !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    border: none !important;
    /* `background-image` et non le raccourci `background` : Gradio définit sa
       couleur de bouton primaire, ce qui remet background-image à `none` et
       écrase un dégradé posé via le raccourci. */
    background-color: transparent !important;
    background-image: linear-gradient(110deg, #a855f7, #ec4899 45%,
                                      #f97316 75%, #a855f7) !important;
    background-size: 260% 100% !important;
    animation: defiler 6s ease infinite;
    box-shadow: 0 8px 28px rgba(168,85,247,0.34) !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}
@keyframes defiler {
    0%, 100% { background-position:   0% 50%; }
    50%      { background-position: 100% 50%; }
}
.gradio-container button#bouton-principal:hover,
#bouton-principal:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 40px rgba(236,72,153,0.48) !important;
}

/* --- Badges des plateformes visées --- */
.plateformes {
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.1rem;
}
.tag {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    padding: 5px 13px;
    border-radius: 999px;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.16);
}
.tag-tiktok  { background-image: linear-gradient(100deg, #ff0050, #00f2ea); }
.tag-reels   { background-image: linear-gradient(100deg, #f9ce34, #ee2a7b 55%, #6228d7); }
.tag-shorts  { background-image: linear-gradient(100deg, #ff0000, #ff6b6b); }
.tag-format  { background: rgba(255,255,255,0.09); color: #cbd5e1; }

/* --- Séparateur « ou » entre les deux façons de fournir la musique --- */
.separateur {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0.9rem 0 0.2rem;
    color: #8b93a3;
    font-size: 0.86rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}
.separateur::before, .separateur::after {
    content: "";
    flex: 1;
    height: 1px;
    background-image: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), transparent);
}

/* --- Bloc de partage --- */
#partage {
    border: 1px solid rgba(34,211,238,0.24);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    background: linear-gradient(160deg, rgba(34,211,238,0.07) 0%, #0d0d14 70%);
}
#partage h3 {
    margin-top: 0;
    background-image: linear-gradient(100deg, #22d3ee, #a855f7);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
#partage ul { margin: 0.4rem 0 0.6rem 1.1rem; }
#partage li { margin: 0.18rem 0; }

/* --- Blocs de contenu : un liseré coloré discret --- */
.gradio-container .block,
.gradio-container .form {
    border-radius: 14px !important;
}
.gradio-container label > span { color: #cbd5e1 !important; }

/* La zone de dépôt est l'action principale : on la met en évidence. */
.gradio-container .block:has(input[type="file"]) {
    border: 1px dashed rgba(168,85,247,0.4) !important;
    background: linear-gradient(160deg, rgba(168,85,247,0.06), transparent 70%) !important;
}

/* Le lecteur ne doit pas manger tout l'écran d'un téléphone. */
.gradio-container video {
    max-height: 62vh !important;
    width: auto !important;
    margin: auto !important;
}

/* --- Téléphones ---
   L'outil produit des vidéos verticales : une bonne partie des visites vient
   d'un téléphone, souvent pour envoyer le clip juste après l'avoir créé. */
@media (max-width: 640px) {
    .gradio-container { padding: 0 0.4rem !important; }

    #entete {
        padding: 1.2rem 0.7rem 1rem;
        border-radius: 14px;
    }
    #entete h1 { font-size: 1.55rem; }
    #entete p  { font-size: 0.85rem; }

    .egaliseur { height: 22px; gap: 3px; margin-top: 0.8rem; }
    .barre     { width: 4px; }
    .clavier   { height: 20px; margin-top: 0.5rem; }
    .touche    { width: 10px; }
    .touche.noire { width: 7px; margin: 0 -4px; }

    .tag { font-size: 0.68rem; padding: 4px 10px; }
    .plateformes { gap: 6px; margin-top: 0.8rem; }

    .separateur { font-size: 0.75rem; letter-spacing: 0.12em; }

    .gradio-container button#bouton-principal {
        font-size: 1.02rem !important;
        padding: 0.85rem !important;
    }

    #partage { padding: 0.9rem 1rem; }
    #partage ul { margin-left: 0.9rem; }

    /* Les réglages passent l'un sous l'autre au lieu de se serrer. */
    .gradio-container .form > .block { min-width: 100% !important; }

    /* Le clavier disparaît : l'égaliseur suffit comme signature musicale,
       et ces 30 pixels comptent quand l'écran fait 812 px de haut. */
    .clavier { display: none; }
}
"""

# Bandeau HTML du haut : clavier animé + notes qui montent.
ENTETE_HTML = """
<div id="entete">
  <span class="note">&#9835;</span>
  <span class="note">&#9834;</span>
  <span class="note">&#9839;</span>
  <span class="note">&#9836;</span>
  <span class="note">&#9834;</span>
  <h1>&#127916; ClipperMusic</h1>
  <p>Ta musique devient un edit vertical avec les paroles en karaoké.
     <b>Prêt à poster en 20 secondes.</b></p>
  <div class="plateformes">
    <span class="tag tag-tiktok">TikTok</span>
    <span class="tag tag-reels">Reels</span>
    <span class="tag tag-shorts">Shorts</span>
    <span class="tag tag-format">1080&times;1920</span>
  </div>
  <div class="egaliseur">
    <div class="barre"></div><div class="barre"></div><div class="barre"></div>
    <div class="barre"></div><div class="barre"></div><div class="barre"></div>
    <div class="barre"></div><div class="barre"></div><div class="barre"></div>
  </div>
  <div class="clavier">
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div><div class="touche noire"></div>
    <div class="touche"></div>
  </div>
</div>
"""

# Affiché sous la vidéo une fois le clip prêt.
PARTAGE_HTML = """
<div id="partage">
  <h3>&#128229; Récupérer et publier</h3>
  <p><b>1.</b> Survole la vidéo et clique sur l'icône de téléchargement
     (&#11015;&#65039;) en haut à droite du lecteur.</p>
  <p><b>2.</b> Envoie le fichier sur ton téléphone — AirDrop, Google Drive,
     ou en te l'envoyant par message.</p>
  <p><b>3.</b> Publie&nbsp;:</p>
  <ul>
    <li><b>TikTok</b> &rarr; Créer &rarr; Importer &rarr; ta vidéo</li>
    <li><b>Instagram Reels</b> &rarr; + &rarr; Reel &rarr; ta vidéo</li>
    <li><b>YouTube Shorts</b> &rarr; + &rarr; Créer un Short</li>
  </ul>
  <p style="color:#9ba1ad;font-size:0.92em;margin-bottom:0">
     Le format 1080&times;1920 est déjà le bon&nbsp;: aucun recadrage à faire.</p>
</div>
"""


def construire_interface() -> gr.Blocks:
    # Note : Gradio 5 avertit que `theme` passera dans launch() en version 6,
    # mais launch() ne l'accepte pas encore. On reste sur Blocks().
    # Thème sombre imposé : l'identité visuelle repose sur le contraste
    # noir/ivoire d'un clavier, elle n'a pas de sens en clair.
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0a0a0c",
        body_background_fill_dark="#0a0a0c",
        block_background_fill="#141419",
        block_background_fill_dark="#141419",
        block_border_color="rgba(255,255,255,0.10)",
        block_label_text_color="#e8e8ec",
        body_text_color="#e8e8ec",
        body_text_color_subdued="#9ba1ad",
        input_background_fill="#1b1b21",
        border_color_primary="rgba(255,255,255,0.12)",
    )

    with gr.Blocks(title="ClipperMusic", theme=theme, css=CSS) as demo:
        gr.HTML(ENTETE_HTML)

        etat_morceaux = gr.State([])
        # Retient la dernière génération : fichier audio, transcription et
        # clips déjà utilisés. C'est ce qui permet de refaire la vidéo sans
        # repasser par Whisper.
        etat_session = gr.State({})

        # --- 1. La musique : les deux façons d'en fournir, côte à côte ---
        # Rien n'est replié ici. C'est la seule décision que l'utilisateur a à
        # prendre, elle doit tenir en un coup d'oeil.
        fichier = gr.Audio(
            label="🎵 Dépose ton MP3",
            type="filepath",
            sources=["upload"],
        )

        gr.HTML('<div class="separateur"><span>ou</span></div>')

        lien = gr.Textbox(
            label="🔗 Colle un lien",
            placeholder="YouTube · Spotify · Deezer · lien .mp3 · ou un style : « pop »",
            info="On identifie ton morceau et on te propose des musiques "
                 "libres de droits, utilisables sans risque.",
        )
        message_lien = gr.Markdown("", visible=False)

        with gr.Accordion(
            "Pourquoi le son ne vient pas du lien", open=False, visible=False
        ) as detail_lien:
            gr.Markdown(EXPLICATION_LIEN)

        choix_morceau = gr.Dropdown(
            choices=[],
            label="🎶 Morceaux libres de droits",
            interactive=True,
            allow_custom_value=True,
            visible=False,
        )

        # --- 2. L'option qui compte ---
        sans_musique = gr.Checkbox(
            label="🔇 Sans musique — pour ajouter le son sur TikTok / Instagram",
            value=False,
            info="Coche si tu publies : le son ajouté depuis la plateforme "
                 "n'est jamais coupé.",
        )

        # --- 3. Le bouton ---
        bouton = gr.Button(
            "🚀 Créer mon edit", variant="primary", size="lg",
            elem_id="bouton-principal",
        )

        # --- 4. Le résultat ---
        message = gr.Markdown("*Compte une à trois minutes.*")
        # Pas de hauteur fixe : le CSS la borne à 62 % de la fenêtre, ce qui
        # évite qu'un clip vertical remplisse tout l'écran d'un téléphone.
        video = gr.Video(label="Ton clip", show_download_button=True)

        # Le mode d'emploi de publication n'apparaît qu'une fois le clip prêt :
        # avant, il n'aurait rien à quoi se rapporter.
        bloc_partage = gr.HTML(PARTAGE_HTML, visible=False)

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

        # Réglages et diagnostic regroupés : deux blocs repliés valaient un
        # encombrement inutile pour des options que personne n'ouvre.
        with gr.Accordion("⚙️ Réglages et diagnostic", open=False):
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
            gr.Markdown(_diagnostic())

        # Le pied de page affiche le moteur réellement chargé. Sans ça,
        # impossible de savoir depuis l'extérieur quelle version tourne — ce
        # qui rend tout diagnostic à distance impossible.
        gr.Markdown(
            f"<sub>{_signature_moteur()} · thèmes visuels spaCy · "
            "visuels Pexels &amp; Pixabay · montage ffmpeg</sub>"
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
            inputs=[fichier, lien, etat_morceaux, choix_morceau, sans_musique,
                    modele, langue],
            outputs=[video, message, credits, paroles, etat_session, bloc_partage],
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

    # Le port doit être configurable : Railway, Render et Fly.io imposent le
    # leur via la variable PORT et refusent le conteneur s'il écoute ailleurs.
    # Hugging Face Spaces attend 7860, qui reste donc la valeur par défaut.
    port = int(
        os.getenv("PORT")
        or os.getenv("GRADIO_SERVER_PORT")
        or 7860
    )

    print("\n" + "=" * 58)
    print("  Ouvre cette adresse dans ton navigateur :")
    print(f"      http://localhost:{port}")
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
        server_port=port,
        share=partager,
        show_error=True,
    )
