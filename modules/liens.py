"""Reconnaissance d'un lien musical collé par l'utilisateur.

Ce module lit le TITRE d'un morceau à partir d'un lien YouTube, Spotify ou
Deezer, puis cherche un équivalent libre de droits.

Ce qu'il ne fait pas, et pourquoi
---------------------------------
Il ne télécharge jamais l'audio de ces plateformes.

  • Spotify et Deezer diffusent leurs flux protégés par DRM. Les contourner
    n'est pas une simple violation de conditions d'utilisation : c'est un délit
    pénal en France (article L.335-3-1 du code de la propriété intellectuelle).

  • YouTube interdit le téléchargement dans ses conditions d'utilisation, et
    bloque de toute façon les adresses IP des hébergeurs — la fonctionnalité
    échouerait la plupart du temps depuis un serveur.

  • Et surtout, ça casserait la promesse du produit : une vidéo montée sur un
    enregistrement commercial est identifiée par Content ID en quelques heures,
    puis coupée du son, retirée, ou monétisée au profit de l'ayant droit.
    « Publiable sur les réseaux » et « musique commerciale » s'excluent.

On utilise donc les API oEmbed **officielles et publiques** de ces trois
plateformes, prévues pour ça : elles renvoient le titre et l'artiste, rien de
plus. Ce titre sert ensuite à proposer un morceau libre de droits dans le même
esprit — et cette vidéo-là, elle, est réellement publiable.
"""

import html
import re

import requests

from modules import audio

HEADERS = {
    "User-Agent": (
        "clip-paroles-portfolio/1.0 "
        "(https://huggingface.co/spaces; projet de démonstration portfolio)"
    )
}

# API oEmbed officielles. Aucune clé, aucune authentification.
PLATEFORMES = {
    "youtube": {
        "nom": "YouTube",
        "api": "https://www.youtube.com/oembed",
        "motifs": ("youtube.com", "youtu.be", "music.youtube.com"),
        "params": lambda url: {"url": url, "format": "json"},
    },
    "spotify": {
        "nom": "Spotify",
        "api": "https://open.spotify.com/oembed",
        "motifs": ("open.spotify.com", "spotify.link"),
        "params": lambda url: {"url": url},
    },
    "deezer": {
        "nom": "Deezer",
        "api": "https://api.deezer.com/oembed",
        "motifs": ("deezer.com", "dzr.page.link"),
        "params": lambda url: {"url": url, "format": "json"},
    },
}


class LienError(Exception):
    """Erreur de lecture d'un lien, message affichable dans l'interface."""


# Extensions reconnues comme un fichier audio directement téléchargeable.
EXTENSIONS_AUDIO = (
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
)

# Un fichier audio de plus de 60 Mo n'est pas une chanson : on refuse avant
# de saturer le disque de l'hébergeur.
TAILLE_MAX_AUDIO = 60 * 1024 * 1024


def est_fichier_audio_direct(url: str) -> bool:
    """Vrai si le lien pointe directement vers un fichier audio.

    Cas courant et parfaitement légitime : un MP3 hébergé sur archive.org, un
    site d'artiste, ou n'importe quel serveur public. Rien à contourner, le
    fichier est simplement téléchargeable.
    """
    sans_parametres = (url or "").split("?")[0].split("#")[0].lower()
    return sans_parametres.endswith(EXTENSIONS_AUDIO)


def telecharger_audio_direct(url: str, destination) -> "Path":
    """Télécharge un fichier audio accessible publiquement.

    Le fichier est ensuite validé par audio.validate_audio(), donc un lien qui
    renvoie une page HTML au lieu d'un son est rejeté avec un message clair.
    """
    from pathlib import Path

    from modules import audio as _audio

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        reponse = requests.get(url, headers=HEADERS, timeout=180, stream=True)
        reponse.raise_for_status()
    except requests.RequestException as erreur:
        raise LienError(f"Téléchargement impossible depuis ce lien : {erreur}")

    annonce = reponse.headers.get("content-length")
    if annonce and int(annonce) > TAILLE_MAX_AUDIO:
        raise LienError(
            f"Ce fichier fait {int(annonce) / 1e6:.0f} Mo, c'est trop lourd "
            f"(maximum {TAILLE_MAX_AUDIO // 1024 // 1024} Mo)."
        )

    ecrit = 0
    try:
        with open(destination, "wb") as fichier:
            for morceau in reponse.iter_content(chunk_size=1 << 16):
                ecrit += len(morceau)
                if ecrit > TAILLE_MAX_AUDIO:
                    raise LienError("Fichier trop lourd, téléchargement interrompu.")
                fichier.write(morceau)
    except OSError as erreur:
        raise LienError(f"Écriture du fichier impossible : {erreur}")

    # Le vrai test : ffprobe sait-il le lire ?
    try:
        _audio.validate_audio(destination)
    except _audio.AudioError as erreur:
        destination.unlink(missing_ok=True)
        raise LienError(
            f"Le lien a bien répondu, mais le fichier n'est pas un audio "
            f"exploitable.\n{erreur}"
        )

    return destination


def detecter_plateforme(texte: str) -> str | None:
    """Renvoie la clé de la plateforme reconnue, ou None."""
    minuscules = (texte or "").lower()
    for cle, plateforme in PLATEFORMES.items():
        if any(motif in minuscules for motif in plateforme["motifs"]):
            return cle
    return None


def est_un_lien(texte: str) -> bool:
    """Vrai si le texte ressemble à une adresse web."""
    return (texte or "").strip().lower().startswith(("http://", "https://", "www."))


def lire_titre(url: str) -> dict:
    """Lit le titre et l'artiste d'un lien musical.

    Renvoie {"plateforme", "titre", "artiste", "recherche"} où "recherche" est
    la version nettoyée du titre, prête à servir de requête.
    """
    url = (url or "").strip()
    cle = detecter_plateforme(url)

    if cle is None:
        raise LienError(
            "Lien non reconnu. Ce champ accepte :\n"
            "• un lien YouTube, Spotify ou Deezer (pour identifier un morceau) ;\n"
            "• un lien direct vers un fichier audio (.mp3, .wav, .m4a…) ;\n"
            "• un style de musique (« pop », « rock », « acoustic »).\n"
            "Sinon, dépose ton fichier dans la zone au-dessus."
        )

    plateforme = PLATEFORMES[cle]

    try:
        reponse = requests.get(
            plateforme["api"],
            params=plateforme["params"](url),
            headers=HEADERS,
            timeout=25,
        )
    except requests.RequestException:
        raise LienError(
            f"{plateforme['nom']} est injoignable pour le moment. "
            "Réessaie dans un instant."
        )

    if reponse.status_code == 404:
        raise LienError(
            f"Ce lien {plateforme['nom']} ne correspond à aucun morceau. "
            "Vérifie que tu as copié l'adresse complète."
        )
    if reponse.status_code != 200:
        raise LienError(
            f"{plateforme['nom']} a refusé la demande (erreur {reponse.status_code}). "
            "Le morceau est peut-être privé ou indisponible dans ton pays."
        )

    try:
        donnees = reponse.json()
    except ValueError:
        raise LienError(f"Réponse illisible de {plateforme['nom']}.")

    titre = html.unescape(donnees.get("title") or "").strip()
    artiste = html.unescape(donnees.get("author_name") or "").strip()

    if not titre:
        raise LienError(f"Impossible de lire le titre depuis ce lien {plateforme['nom']}.")

    return {
        "plateforme": plateforme["nom"],
        "titre": titre,
        "artiste": artiste,
        "recherche": nettoyer_titre(titre),
    }


# Mentions ajoutées par les plateformes, sans rapport avec le morceau lui-même.
PARASITES = re.compile(
    r"\b(official\s*(music\s*)?video|official\s*audio|lyric[s]?\s*video|lyrics|"
    r"clip\s*officiel|audio\s*officiel|visualizer|hd|4k|hq|remaster(ed)?|"
    r"music\s*video|full\s*album|live|cover|extended|radio\s*edit)\b",
    re.IGNORECASE,
)


def nettoyer_titre(titre: str) -> str:
    """Réduit un titre de plateforme à ses mots utiles.

    « Luis Fonsi - Despacito ft. Daddy Yankee (Official Video) [4K] »
    devient « Luis Fonsi Despacito ».
    """
    texte = titre

    # Tout ce qui est entre parenthèses ou crochets est presque toujours du bruit.
    texte = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", texte)

    # Les featuring n'aident pas à retrouver une ambiance.
    texte = re.split(r"\b(ft|feat|featuring|avec|with)\b\.?", texte, flags=re.IGNORECASE)[0]

    texte = PARASITES.sub(" ", texte)
    texte = re.sub(r"[|/\\_–—•·:,\"']+", " ", texte)
    texte = re.sub(r"\s*-\s*", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()

    # Un titre réduit à rien : on garde l'original plutôt qu'une chaîne vide.
    return texte or titre.strip()


def chercher_equivalent_libre(infos: dict, limite: int = 8) -> list[dict]:
    """Cherche des morceaux libres de droits proches du morceau identifié.

    Trois tentatives, de la plus précise à la plus large. On ne renvoie jamais
    une liste vide sans avoir vraiment essayé : l'utilisateur a collé un lien,
    il attend un résultat, pas un message d'échec.
    """
    tentatives = [infos["recherche"]]

    # Le nom de l'artiste seul ouvre souvent des morceaux du même style.
    if infos.get("artiste"):
        tentatives.append(infos["artiste"])

    # Puis les mots du titre pris isolément (les plus longs d'abord : ce sont
    # les plus porteurs de sens).
    mots = sorted(
        (m for m in infos["recherche"].split() if len(m) > 3),
        key=len,
        reverse=True,
    )
    tentatives.extend(mots[:3])

    # Dernier recours : un genre très courant, pour toujours proposer quelque chose.
    tentatives.append("pop")

    deja_vu = set()
    for tentative in tentatives:
        if not tentative or tentative.lower() in deja_vu:
            continue
        deja_vu.add(tentative.lower())
        try:
            resultats = audio.search_jamendo(tentative, limit=limite)
        except audio.AudioError:
            continue
        if resultats:
            return resultats

    raise LienError(
        "Aucun morceau libre de droits n'a pu être trouvé. "
        "Réessaie dans un instant, ou dépose ton propre fichier audio."
    )
