"""Transcription des paroles avec timestamps mot par mot.

On utilise **faster-whisper**, et pas openai-whisper ni whisper-timestamped.

Pourquoi ce choix, mesuré sur le fichier de test (154 s d'audio) :

    whisper-timestamped (PyTorch)   ~1 900 Mo de RAM,  ~70 s
    faster-whisper (CTranslate2)      ~390 Mo de RAM,   ~9 s

Le facteur limitant n'était pas le modèle mais **PyTorch** : à lui seul il
réserve plus d'un gigaoctet. faster-whisper s'appuie sur CTranslate2, écrit en
C++, qui n'a besoin ni de PyTorch ni de CUDA. Le projet a donc perdu sa plus
grosse dépendance, ce qui a débloqué l'hébergement : sur un conteneur limité en
mémoire, la version PyTorch se faisait tuer au chargement du modèle.

Les timestamps par mot, indispensables au karaoké, sont fournis nativement via
`word_timestamps=True`.

Ce module renvoie une structure simple, réutilisée par tout le reste du pipeline :

    {
      "language": "fr",
      "start_offset": 12.4,   # début de l'extrait retenu dans le fichier d'origine
      "duration": 88.2,       # durée de l'extrait retenu
      "segments": [
        {
          "text": "je marche seul dans la ville",
          "start": 0.0, "end": 3.2, "confidence": 0.82,
          "words": [{"text": "je", "start": 0.0, "end": 0.2, "confidence": 0.9}, ...]
        },
        ...
      ]
    }

Les temps des segments sont TOUJOURS relatifs au début de l'extrait retenu
(donc le premier segment commence proche de 0), ce qui simplifie le montage.
"""

import os
import sys
from pathlib import Path
from typing import Callable

from modules import config


class TranscriptionError(Exception):
    """Erreur de transcription, avec un message affichable dans l'interface."""


# Le modèle pèse plusieurs centaines de Mo : on le garde en mémoire entre
# deux traitements plutôt que de le recharger à chaque fois.
_model_cache: dict[str, object] = {}


# Mémoire réellement consommée par chaque modèle, en mégaoctets : pic mesuré
# sur une transcription complète, quantification int8 comprise.
#
# Ces chiffres ont été mesurés, pas estimés — et les premières estimations
# étaient fausses de 130 Mo sur « small », ce qui l'écartait à tort sur un
# hébergement où il tient très bien.
BESOIN_MEMOIRE_MO = {"tiny": 340, "base": 390, "small": 580, "medium": 1400}

# Ce que consomme l'application pendant que la transcription travaille dans son
# sous-processus : Gradio, spaCy, l'interpréteur principal et ses tampons.
#
# 420 et non 300 : la mesure « à froid » donnait 250 Mo, mais en conditions
# réelles s'ajoutent les tampons de Gradio, le fichier audio en mémoire et la
# fragmentation de l'allocateur. Avec 300, « small » passait deux fois sur
# trois — et la troisième tuait le conteneur, ce qui est pire qu'un modèle
# moins précis. La marge couvre cet écart entre mesure et réalité.
MEMOIRE_RESERVEE_MO = int(os.getenv("MEMOIRE_RESERVEE_MO", "420"))


def decharger_modeles() -> float:
    """Libère les modèles gardés en mémoire. Renvoie les Mo estimés récupérés.

    Le modèle reste normalement en cache d'un traitement à l'autre, pour éviter
    de le recharger à chaque fois. Mais pendant le montage, il ne sert plus à
    rien : il occupe simplement la mémoire dont ffmpeg a besoin pour décoder
    les vidéos. Sur un conteneur limité, c'est cette occupation inutile qui
    faisait tuer le montage.
    """
    import gc

    recupere = sum(BESOIN_MEMOIRE_MO.get(nom, 0) for nom in _model_cache)
    _model_cache.clear()
    gc.collect()
    return float(recupere)


def modele_tenable(demande: str) -> tuple[str, str | None]:
    """Rétrograde le modèle demandé s'il ne tient pas dans la mémoire allouée.

    Sur un hébergement limité, charger un modèle trop gros ne produit pas une
    erreur Python : le système tue le processus. L'utilisateur voit alors la
    page se figer sans explication. Mieux vaut transcrire un peu moins
    finement que ne rien transcrire du tout.

    Renvoie (modèle retenu, message d'avertissement ou None).
    """
    disponible = config.memoire_disponible_mo()
    if disponible is None:
        return demande, None

    budget = disponible - MEMOIRE_RESERVEE_MO
    if BESOIN_MEMOIRE_MO.get(demande, 0) <= budget:
        return demande, None

    # On descend jusqu'au plus gros modèle qui tienne.
    for candidat in ("small", "base", "tiny"):
        if BESOIN_MEMOIRE_MO[candidat] <= budget:
            return candidat, (
                f"Modèle « {demande} » remplacé par « {candidat} » : "
                f"{disponible:.0f} Mo de mémoire disponibles, ce qui est "
                f"insuffisant pour « {demande} »."
            )

    return "tiny", (
        f"Mémoire très limitée ({disponible:.0f} Mo) : modèle « tiny » imposé. "
        "La transcription sera approximative."
    )


def load_model(model_name: str | None = None):
    """Charge (et met en cache) un modèle faster-whisper."""
    from faster_whisper import WhisperModel

    name = model_name or config.WHISPER_MODEL
    name, avertissement = modele_tenable(name)
    if avertissement:
        print(f"[transcription] {avertissement}")
    if name not in _model_cache:
        try:
            _model_cache[name] = WhisperModel(
                name,
                device="cpu",
                # int8 : quantification 8 bits. Divise par deux la mémoire et
                # accélère nettement, pour une perte de précision imperceptible
                # sur du chant.
                compute_type="int8",
                cpu_threads=config.CPU_THREADS,
            )
        except Exception as error:
            raise TranscriptionError(
                f"Impossible de charger le modèle « {name} » : {error}\n"
                "Au premier lancement le modèle est téléchargé, cela peut prendre "
                "quelques minutes et demande une connexion internet."
            )
    return _model_cache[name]


def transcribe_audio(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = None,
    progress: Callable[[str], None] | None = None,
    rang_passage: int = 0,
) -> dict:
    """Transcrit un fichier audio et renvoie le moment fort de la chanson.

    `language` : code ISO ("fr", "en"...) ou None pour détection automatique.
    `progress` : fonction appelée avec un message d'étape, pour l'interface.

    Sur un hébergement à la mémoire limitée, le travail est délégué à un
    processus séparé : c'est le seul moyen que la mémoire du modèle soit
    réellement rendue au système avant le montage (voir transcribe_worker).
    """
    memoire = config.memoire_disponible_mo()
    en_sous_processus = memoire is not None and memoire < SEUIL_SOUS_PROCESSUS_MO

    if en_sous_processus:
        if progress:
            progress("Transcription des paroles (étape la plus longue)…")

        # Le modèle retenu peut malgré tout ne pas tenir : les besoins varient
        # avec la durée du morceau. Plutôt qu'échouer, on redescend d'un cran.
        demande = model_name or config.WHISPER_MODEL
        replis = [demande] + [
            m for m in ("base", "tiny")
            if BESOIN_MEMOIRE_MO.get(m, 0) < BESOIN_MEMOIRE_MO.get(demande, 0)
        ]

        derniere: TranscriptionError | None = None
        for essai, modele in enumerate(replis):
            if essai and progress:
                progress(f"Mémoire insuffisante : reprise avec le modèle « {modele} »…")
            try:
                return _transcrire_via_sous_processus(
                    audio_path, modele, language, rang_passage
                )
            except TranscriptionError as erreur:
                derniere = erreur
                if "mémoire" not in str(erreur).lower():
                    raise
        raise derniere or TranscriptionError("La transcription a échoué.")

    if progress:
        progress("Chargement du modèle de transcription…")
    return transcrire_directement(
        audio_path, model_name, language, progress, rang_passage
    )


# En dessous de cette mémoire, la transcription part dans un processus séparé.
SEUIL_SOUS_PROCESSUS_MO = 2000


def _transcrire_via_sous_processus(
    audio_path: str | Path,
    model_name: str | None,
    language: str | None,
    rang_passage: int = 0,
) -> dict:
    """Lance la transcription dans un processus séparé et lit son résultat."""
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as dossier:
        sortie = Path(dossier) / "transcription.json"
        commande = [
            sys.executable, "-m", "modules.transcribe_worker",
            str(audio_path), str(sortie),
            model_name or "", language or "", str(rang_passage),
        ]
        try:
            resultat = subprocess.run(
                commande,
                cwd=str(config.ROOT_DIR),
                capture_output=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            raise TranscriptionError("La transcription a dépassé le temps maximum.")

        # Un code négatif signifie que le processus a été TUÉ par un signal
        # (mémoire insuffisante), pas qu'il s'est arrêté sur une erreur.
        if resultat.returncode < 0 or resultat.returncode in (137, 143):
            raise TranscriptionError(
                "La transcription a été interrompue par le système, faute de "
                "mémoire. Essaie un modèle plus léger dans les réglages "
                "avancés (« tiny »)."
            )

        if not sortie.exists():
            details = resultat.stderr.decode("utf-8", errors="replace").strip()
            tail = "\n".join(details.splitlines()[-4:])
            raise TranscriptionError(
                f"La transcription a échoué sans produire de résultat.\n{tail}"
            )

        donnees = json.loads(sortie.read_text(encoding="utf-8"))

    if "erreur" in donnees:
        raise TranscriptionError(donnees["erreur"])
    return donnees


def transcrire_directement(
    audio_path: str | Path,
    model_name: str | None = None,
    language: str | None = None,
    progress: Callable[[str], None] | None = None,
    rang_passage: int = 0,
) -> dict:
    """Fait réellement la transcription, dans le processus courant."""
    from faster_whisper.audio import decode_audio

    model = load_model(model_name)

    if progress:
        progress("Transcription des paroles (étape la plus longue)…")

    # On ne transcrit qu'une portion : voir config.MAX_TRANSCRIBE_SECONDS.
    decalage = _decalage_analyse(audio_path)

    try:
        audio = decode_audio(str(audio_path), sampling_rate=16000)
    except Exception as error:
        raise TranscriptionError(f"Lecture du fichier audio impossible : {error}")

    if decalage > 0 or len(audio) > 16000 * config.MAX_TRANSCRIBE_SECONDS:
        debut = int(decalage * 16000)
        fin = debut + int(config.MAX_TRANSCRIBE_SECONDS * 16000)
        audio = audio[debut:fin]

    try:
        segments_bruts, info = model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            # Sur une chanson, la « température de repli » fait inventer du
            # texte. On reste déterministe.
            temperature=0.0,
            # Pas de VAD : sur des enregistrements bruités ou anciens, il
            # supprime la totalité des paroles (constaté sur le fichier de
            # test de 1911, où il ne restait aucun segment). Les hallucinations
            # sont écartées ensuite par _drop_unreliable.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        # model.transcribe renvoie un générateur : rien n'est calculé tant
        # qu'on ne le parcourt pas.
        segments_bruts = list(segments_bruts)
    except Exception as error:
        raise TranscriptionError(f"La transcription a échoué : {error}")

    langue_detectee = info.language or language or "fr"

    segments = _normalise_segments(segments_bruts)
    segments = _drop_unreliable(segments)

    total_words = sum(len(segment["words"]) for segment in segments)
    if total_words < config.MIN_WORDS_REQUIRED:
        raise TranscriptionError(
            "Aucune parole détectée dans ce fichier.\n"
            "C'est normal pour un morceau instrumental : le clip karaoké a besoin "
            "de paroles chantées. Essaie un autre morceau."
        )

    if progress:
        progress("Sélection du moment fort…")
    resultat = _select_best_window(segments, langue_detectee, rang_passage)

    # Les temps calculés sont relatifs au morceau analysé. On les recale sur le
    # fichier d'origine, sinon la bande son serait extraite au mauvais endroit.
    resultat["start_offset"] += decalage
    return resultat


def appliquer_texte_corrige(segments: list[dict], texte: str) -> list[dict]:
    """Remplace les paroles par une version corrigée à la main, sans retranscrire.

    Whisper se trompe régulièrement, surtout sur l'argot et les noms propres.
    Plutôt que de relancer une transcription (une minute ou plus), on réutilise
    les temps déjà calculés et on n'échange que les mots.

    `texte` contient une ligne par segment, dans le même ordre qu'à l'affichage.

    Quand une ligne corrigée n'a pas le même nombre de mots que l'originale, on
    redistribue la durée de la ligne sur les nouveaux mots, proportionnellement
    à leur longueur — une syllabe de plus prend un peu plus de temps. Ce n'est
    pas un réalignement acoustique, mais l'écart reste sous le dixième de
    seconde, invisible à l'oeil sur un karaoké.
    """
    lignes = [ligne.strip() for ligne in (texte or "").splitlines()]
    lignes = [ligne for ligne in lignes if ligne]

    if not lignes:
        raise TranscriptionError("Les paroles corrigées sont vides.")

    if len(lignes) != len(segments):
        raise TranscriptionError(
            f"Il faut exactement une ligne par phrase : {len(segments)} lignes "
            f"attendues, {len(lignes)} reçues.\n"
            "Corrige les mots sans ajouter ni supprimer de ligne."
        )

    corriges = []
    for segment, ligne in zip(segments, lignes):
        mots = ligne.split()
        if not mots:
            continue

        debut = segment["words"][0]["start"]
        fin = segment["words"][-1]["end"]
        duree = max(fin - debut, 0.05)

        # Répartition proportionnelle à la longueur des mots.
        longueurs = [len(mot) for mot in mots]
        total = sum(longueurs) or len(mots)

        nouveaux_mots = []
        curseur = debut
        for mot, longueur in zip(mots, longueurs):
            part = duree * (longueur / total)
            nouveaux_mots.append(
                {
                    "text": mot,
                    "start": curseur,
                    "end": curseur + part,
                    "confidence": 1.0,  # corrigé par un humain
                }
            )
            curseur += part

        nouveaux_mots[-1]["end"] = fin

        corriges.append(
            {
                "text": " ".join(mots),
                "start": debut,
                "end": fin,
                "confidence": 1.0,
                "words": nouveaux_mots,
            }
        )

    return corriges


def _normalise_segments(raw_segments) -> list[dict]:
    """Convertit la sortie de faster-whisper en structure simple et propre.

    faster-whisper renvoie des objets (Segment, Word) et non des dictionnaires.
    On les aplatit ici pour que le reste du pipeline n'ait pas à connaître la
    bibliothèque utilisée — c'est ce qui a permis de changer de moteur de
    transcription sans toucher aux autres modules.
    """
    segments = []
    for raw in raw_segments:
        words = []
        for raw_word in (getattr(raw, "words", None) or []):
            text = (raw_word.word or "").strip()
            start, end = raw_word.start, raw_word.end
            if not text or start is None or end is None or end <= start:
                continue
            words.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    # faster-whisper donne une probabilité par mot ; c'est
                    # l'équivalent direct de l'ancienne « confidence ».
                    "confidence": float(raw_word.probability or 0.0),
                }
            )

        if not words:
            continue

        # Pas de confiance fournie au niveau du segment : on prend la moyenne
        # de ses mots, ce qui donne la même échelle qu'avant (0 à 1).
        confiance = sum(w["confidence"] for w in words) / len(words)

        segments.append(
            {
                "text": " ".join(word["text"] for word in words),
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "confidence": confiance,
                "words": words,
            }
        )
    return segments


def _drop_unreliable(segments: list[dict]) -> list[dict]:
    """Jette les segments que Whisper a probablement inventés.

    Trois signaux d'hallucination, tous observés en pratique sur de la musique :
      - une confiance basse ;
      - le même mot répété en boucle ("la la la la la la la") ;
      - un segment absurdement long pour le nombre de mots qu'il contient.
    """
    kept = []
    for segment in segments:
        if segment["confidence"] and segment["confidence"] < config.MIN_SEGMENT_CONFIDENCE:
            continue

        words = [word["text"].lower().strip(".,!?;:") for word in segment["words"]]
        if len(words) >= 4 and len(set(words)) == 1:
            continue

        duration = segment["end"] - segment["start"]
        if duration > 2 and duration / max(len(words), 1) > 3.0:
            # Plus de 3 secondes par mot : c'est du silence habillé de texte.
            continue

        kept.append(segment)
    return kept


def _decalage_analyse(audio_path) -> float:
    """À quelle seconde commencer l'analyse, pour sauter l'intro.

    Sur une chanson plus longue que ce qu'on analyse, démarrer un peu après le
    début évite l'intro instrumentale — où il n'y a rien à transcrire.
    Renvoie 0 si le morceau est assez court pour être analysé en entier.
    """
    from modules import audio as _audio

    try:
        duree = _audio.probe_duration(audio_path)
    except Exception:
        return 0.0

    if duree <= config.MAX_TRANSCRIBE_SECONDS:
        return 0.0

    marge = duree - config.MAX_TRANSCRIBE_SECONDS
    return round(min(duree * config.TRANSCRIBE_START_RATIO, marge), 2)


def _poids_refrain(segments: list[dict]) -> dict[int, float]:
    """Donne à chaque segment un poids selon qu'il appartient au refrain.

    Un refrain, c'est ce qui revient. On compte donc combien de fois chaque
    phrase apparaît dans la chanson : celles qui reviennent plusieurs fois sont
    presque toujours le refrain, c'est-à-dire le passage que les gens
    reconnaissent — exactement ce qu'il faut mettre dans un edit de 15 secondes.

    Renvoie {indice du segment : poids}, le poids valant 1.0 pour un couplet.
    """
    import re
    from collections import Counter

    def normaliser(texte: str) -> str:
        # On compare le fond, pas la ponctuation ni la casse.
        return re.sub(r"[^\w\s]", "", texte.lower()).strip()

    occurrences = Counter(normaliser(s["text"]) for s in segments)

    poids = {}
    for index, segment in enumerate(segments):
        repetitions = occurrences[normaliser(segment["text"])]
        # 1 occurrence = couplet (1.0), 2 = 2.5, 3 et plus = 4.0.
        # L'écart est volontairement net : une phrase répétée trois fois doit
        # l'emporter sur un couplet plus dense en mots.
        poids[index] = {1: 1.0, 2: 2.5}.get(repetitions, 4.0)
    return poids


def _select_best_window(
    segments: list[dict], language: str, rang: int = 0
) -> dict:
    """Garde au maximum MAX_CLIP_SECONDS de la partie la plus forte de la chanson.

    Couper les 15 premières secondes donnerait le plus souvent une intro
    instrumentale. On cherche donc la fenêtre qui maximise un score combinant :
      - le nombre de mots (un passage bavard vaut mieux qu'un passage vide) ;
      - l'appartenance au refrain (voir _poids_refrain), qui pèse bien plus.
    """
    max_seconds = config.MAX_CLIP_SECONDS

    if segments and segments[-1]["end"] - segments[0]["start"] <= max_seconds:
        window = segments
        alternatives = 1
    else:
        poids = _poids_refrain(segments)

        # On note toutes les fenêtres possibles, puis on garde celle du rang
        # demandé. `rang` permet à l'utilisateur de demander « un autre
        # passage » sans relancer la transcription.
        candidates: list[tuple[float, int, list[dict]]] = []
        for index, first in enumerate(segments):
            limit = first["start"] + max_seconds
            fenetre_indices = [
                i for i in range(index, len(segments))
                if segments[i]["end"] <= limit
            ]
            if not fenetre_indices:
                fenetre_indices = [index]

            score = sum(
                len(segments[i]["words"]) * poids[i] for i in fenetre_indices
            )
            candidates.append((score, index, [segments[i] for i in fenetre_indices]))

        candidates.sort(key=lambda c: (-c[0], c[1]))

        # Deux fenêtres qui se chevauchent presque donnent le même clip : on ne
        # garde que des passages nettement distincts, sinon « autre passage »
        # renverrait quasiment la même chose.
        distinctes: list[tuple[float, int, list[dict]]] = []
        for score, index, fenetre in candidates:
            debut = fenetre[0]["start"]
            if all(abs(debut - d[2][0]["start"]) > max_seconds * 0.5 for d in distinctes):
                distinctes.append((score, index, fenetre))

        alternatives = len(distinctes)
        window = distinctes[rang % alternatives][2] if distinctes else candidates[0][2]

    # On recale tous les temps sur le début de la fenêtre retenue.
    # Un petit coussin avant le premier mot évite de démarrer pile sur la voix.
    offset = max(0.0, window[0]["start"] - 0.4)
    shifted = []
    for segment in window:
        shifted.append(
            {
                "text": segment["text"],
                "start": segment["start"] - offset,
                "end": segment["end"] - offset,
                "confidence": segment["confidence"],
                "words": [
                    {
                        "text": word["text"],
                        "start": word["start"] - offset,
                        "end": word["end"] - offset,
                        "confidence": word["confidence"],
                    }
                    for word in segment["words"]
                ],
            }
        )

    duration = shifted[-1]["end"] + 0.6  # petite respiration à la fin

    return {
        "language": language,
        "start_offset": offset,
        "duration": duration,
        "segments": shifted,
        # Nombre de passages nettement différents disponibles dans ce morceau,
        # et celui qui a été retenu. Permet à l'interface de proposer
        # « essayer un autre passage » à bon escient.
        "passages_disponibles": alternatives,
        "passage_retenu": rang % max(alternatives, 1),
    }
