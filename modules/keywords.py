"""Extraction des mots-clés visuels à partir des paroles.

Deux problèmes à régler, et le second est le vrai :

1. Trouver les mots porteurs de sens dans une phrase (spaCy : noms et adjectifs).

2. Les transformer en quelque chose de FILMABLE. Chercher « amour » ou « liberté »
   sur une banque de vidéos ne donne rien d'utilisable — au mieux des photos de
   couples en studio. Un dictionnaire de correspondance traduit donc chaque
   thème abstrait en scène concrète : « solitude » -> « empty street at night ».

Ce dictionnaire fait aussi office de traduction : Pexels et Pixabay sont indexés
en anglais, donc toutes les requêtes sortent en anglais, même pour des paroles
françaises. Un mot français absent du dictionnaire ne peut pas être traduit :
dans ce cas on bascule sur un visuel générique plutôt que de chercher dans le vide.
"""

import re

from modules import config


class KeywordError(Exception):
    """Erreur d'extraction de mots-clés, message affichable dans l'interface."""


# --- Dictionnaire thème -> scène filmable -----------------------------------
# Clés en minuscules, sans accent traité à part (on normalise avant de chercher).
# Français et anglais dans le même dictionnaire : les paroles mélangent souvent
# les deux, et ça évite d'avoir à savoir dans quelle langue on est.

VISUAL_MAP: dict[str, str] = {
    # Émotions et états
    "amour": "couple silhouette sunset",
    "love": "couple silhouette sunset",
    "coeur": "slow motion waves ocean",
    "heart": "slow motion waves ocean",
    "douleur": "rain on window",
    "pain": "rain on window",
    "peine": "rain on window",
    "tristesse": "rain drops glass dark",
    "sad": "rain drops glass dark",
    "larme": "rain drops glass dark",
    "tears": "rain drops glass dark",
    "solitude": "empty street at night",
    "seul": "person walking alone street",
    "alone": "person walking alone street",
    "lonely": "empty street at night",
    "joie": "friends laughing sunlight",
    "joy": "friends laughing sunlight",
    "bonheur": "sunlight through trees",
    "happy": "friends laughing sunlight",
    "colere": "storm clouds time lapse",
    "anger": "storm clouds time lapse",
    "rage": "fire flames close up",
    "peur": "dark forest fog",
    "fear": "dark forest fog",
    "espoir": "sunrise over horizon",
    "hope": "sunrise over horizon",
    "reve": "clouds time lapse sky",
    "dream": "clouds time lapse sky",
    "liberte": "birds flying sky",
    "freedom": "birds flying sky",
    "libre": "open road drone",
    "free": "birds flying sky",
    "ame": "candle flame dark",
    "soul": "candle flame dark",
    "vie": "city crowd time lapse",
    "life": "city crowd time lapse",
    "mort": "abandoned building fog",
    "death": "abandoned building fog",
    "temps": "clock time lapse",
    "time": "clock time lapse",
    "souvenir": "old photographs vintage",
    "memory": "old photographs vintage",
    "passe": "old film grain vintage",
    "past": "old film grain vintage",
    "avenir": "highway at night driving",
    "future": "highway at night driving",
    "destin": "long road horizon",
    "fate": "long road horizon",
    "silence": "empty room window light",
    "guerre": "storm dark clouds",
    "war": "storm dark clouds",
    "paix": "calm lake morning",
    "peace": "calm lake morning",
    "force": "waves crashing rocks",
    "strength": "waves crashing rocks",
    "argent": "city skyline night lights",
    "money": "city skyline night lights",
    "succes": "city skyline aerial",
    "success": "city skyline aerial",
    "galere": "rainy street city night",
    "haine": "fire flames close up",
    "hate": "fire flames close up",
    "verite": "mirror reflection portrait",
    "truth": "mirror reflection portrait",
    "mensonge": "shattered glass slow motion",
    "lie": "shattered glass slow motion",
    "doute": "fog road morning",
    "doubt": "fog road morning",
    "folie": "neon lights blur night",
    "crazy": "neon lights blur night",
    "fete": "party crowd lights",
    "party": "party crowd lights",
    "danse": "dancing silhouette lights",
    "dance": "dancing silhouette lights",
    "musique": "concert crowd lights",
    "music": "concert crowd lights",
    "chanson": "vinyl record turning",
    "song": "vinyl record turning",
    "voix": "microphone studio dark",
    "voice": "microphone studio dark",
    # Lieux et éléments concrets
    "ville": "city street aerial night",
    "city": "city street aerial night",
    "rue": "street walking people",
    "street": "street walking people",
    "route": "empty road drone shot",
    "road": "empty road drone shot",
    "maison": "house window evening",
    "home": "warm living room light",
    "chambre": "bedroom morning light",
    "room": "empty room window light",
    "mer": "ocean waves aerial",
    "sea": "ocean waves aerial",
    "ocean": "ocean waves aerial",
    "plage": "beach sunset waves",
    "beach": "beach sunset waves",
    "montagne": "mountain range clouds",
    "mountain": "mountain range clouds",
    "foret": "forest trees sunlight",
    "forest": "forest trees sunlight",
    "ciel": "sky clouds time lapse",
    "sky": "sky clouds time lapse",
    "etoile": "starry night sky",
    "star": "starry night sky",
    "lune": "full moon night clouds",
    "moon": "full moon night clouds",
    "soleil": "sun flare golden hour",
    "sun": "sun flare golden hour",
    "pluie": "rain city street",
    "rain": "rain city street",
    "neige": "snow falling forest",
    "snow": "snow falling forest",
    "vent": "wind field grass",
    "wind": "wind field grass",
    "feu": "fire flames close up",
    "fire": "fire flames close up",
    "eau": "water surface slow motion",
    "water": "water surface slow motion",
    "nuit": "night city lights",
    "night": "night city lights",
    "jour": "morning sunrise city",
    "day": "morning sunrise city",
    "matin": "morning light window",
    "morning": "morning light window",
    "soir": "sunset city skyline",
    "evening": "sunset city skyline",
    "hiver": "winter snow landscape",
    "winter": "winter snow landscape",
    "ete": "summer field golden light",
    "summer": "summer field golden light",
    "train": "train passing motion",
    "voiture": "car driving night city",
    "car": "car driving night city",
    "avion": "airplane sky clouds",
    "plane": "airplane sky clouds",
    "telephone": "phone screen hand night",
    "phone": "phone screen hand night",
    "miroir": "mirror reflection portrait",
    "mirror": "mirror reflection portrait",
    "porte": "door opening light",
    "door": "door opening light",
    "fenetre": "window rain city",
    "window": "window rain city",
    "lumiere": "light rays dust",
    "light": "light rays dust",
    "ombre": "shadow silhouette wall",
    "shadow": "shadow silhouette wall",
    "noir": "dark abstract smoke",
    "dark": "dark abstract smoke",
    "route_perdue": "desert road horizon",
    # Personnes
    "ami": "friends walking together",
    "friend": "friends walking together",
    "famille": "family walking beach",
    "family": "family walking beach",
    "mere": "mother child silhouette",
    "mother": "mother child silhouette",
    "pere": "father child walking",
    "father": "father child walking",
    "enfant": "children playing outdoor",
    "child": "children playing outdoor",
    "fille": "young woman portrait window",
    "girl": "young woman portrait window",
    "garcon": "young man portrait street",
    "boy": "young man portrait street",
    "homme": "man walking city street",
    "man": "man walking city street",
    "femme": "woman walking city",
    "woman": "woman walking city",
    "monde": "earth globe space",
    "world": "earth globe space",
    "gens": "crowd people walking",
    "people": "crowd people walking",
    # Verbes fréquents dans les paroles : ils portent souvent l'image quand
    # la phrase n'a aucun nom concret ("let me hear you whisper").
    "marcher": "person walking street slow motion",
    "walk": "person walking street slow motion",
    "courir": "running person motion blur",
    "run": "running person motion blur",
    "danser": "dancing silhouette lights",
    "voler": "birds flying sky",
    "fly": "birds flying sky",
    "tomber": "rain slow motion dark",
    "fall": "rain slow motion dark",
    "brûler": "fire flames close up",
    "burn": "fire flames close up",
    "pleurer": "rain on window",
    "cry": "rain on window",
    "dormir": "bedroom morning light",
    "sleep": "bedroom morning light",
    "conduire": "car driving night city",
    "drive": "car driving night city",
    "partir": "train leaving station",
    "leave": "train leaving station",
    "attendre": "person waiting window",
    "wait": "person waiting window",
    "chuchoter": "close up portrait dark",
    "whisper": "close up portrait dark",
    "briller": "light rays dust",
    "shine": "light rays dust",
    "glow": "light rays dust",
    # Compléments concrets manquants repérés au test
    "autoroute": "highway at night driving",
    "highway": "highway at night driving",
    "terre": "aerial landscape drone",
    "land": "aerial landscape drone",
    "champ": "wheat field wind",
    "field": "wheat field wind",
    "pont": "bridge city fog",
    "bridge": "bridge city fog",
    "ecole": "empty classroom light",
    "school": "empty classroom light",
    "hiver_froid": "frozen lake winter",
    "froid": "frozen lake winter",
    "cold": "frozen lake winter",
    "chaud": "desert heat haze",
    "hot": "desert heat haze",
    "orage": "lightning storm night",
    "storm": "lightning storm night",
    "vague": "ocean waves slow motion",
    "wave": "ocean waves slow motion",
    "fleur": "flowers close up macro",
    "flower": "flowers close up macro",
    "arbre": "trees canopy sunlight",
    "tree": "trees canopy sunlight",
    "oiseau": "birds flying sky",
    "bird": "birds flying sky",
}

# Utilisés quand aucun mot-clé exploitable n'est trouvé.
# On tourne dans la liste pour éviter dix fois le même fond.
FALLBACK_QUERIES = [
    "abstract light background",
    "city night time lapse",
    "clouds moving sky",
    "ocean waves slow motion",
    "neon lights bokeh",
    "smoke abstract dark",
    "forest light rays",
    "rain window night",
]

# --- Morceaux sans paroles --------------------------------------------------
#
# Un instrumental, un beat, une maquette : rien à transcrire, donc aucun
# mot-clé à extraire. Refuser ces morceaux privait l'outil de tout un pan de
# la production musicale.
#
# À la place, on puise dans un répertoire d'images « aspirationnelles » : les
# codes visuels des edits qui tournent — hauteur, vitesse, lumière, matière.
# Ce ne sont pas des illustrations du texte (il n'y en a pas), mais une
# ambiance qui tient sur quinze secondes.
#
# Les familles sont mélangées à la génération pour qu'un clip alterne les
# registres au lieu d'enchaîner cinq plans de voiture.
AMBIANCES_INSTRUMENTALES = {
    "hauteur": [
        "aerial drone mountain sunrise",
        "cliff edge ocean aerial",
        "skyscraper rooftop city aerial",
        "airplane wing above clouds",
        "hot air balloon sunrise valley",
    ],
    "vitesse": [
        "car driving highway night lights",
        "motorcycle riding city night",
        "train window landscape motion",
        "speedboat ocean wake aerial",
        "running through city night",
    ],
    "lumiere": [
        "neon signs rain reflection night",
        "golden hour sun flare field",
        "city lights bokeh out of focus",
        "light rays through fog forest",
        "sunset silhouette horizon",
    ],
    "matiere": [
        "ink drop water slow motion",
        "smoke swirling dark background",
        "sparks flying slow motion dark",
        "water surface ripple macro",
        "fabric flowing slow motion",
    ],
    "solitude": [
        "person walking empty street night",
        "silhouette window city view",
        "lone figure desert road",
        "empty swimming pool night",
        "person on rooftop city skyline",
    ],
    "luxe": [
        "modern architecture minimal interior",
        "marble texture luxury detail",
        "yacht deck ocean sunset",
        "vintage car close up detail",
        "champagne pouring slow motion",
    ],
}


def requetes_instrumentales(nombre: int) -> list[str]:
    """Construit une suite de requêtes visuelles pour un morceau sans paroles.

    On alterne volontairement les familles : cinq plans de voiture à la suite
    ressemblent à une publicité automobile, alors qu'un clip a besoin de
    respirer entre les registres.
    """
    import random

    familles = list(AMBIANCES_INSTRUMENTALES)
    random.shuffle(familles)

    requetes: list[str] = []
    tour = 0
    while len(requetes) < nombre:
        famille = familles[tour % len(familles)]
        choix = AMBIANCES_INSTRUMENTALES[famille]
        requetes.append(choix[(tour // len(familles)) % len(choix)])
        tour += 1

    return requetes[:nombre]

# Mots trop fréquents ou trop creux pour donner une image, en plus des
# mots vides que spaCy connaît déjà (l'argot et les tics de langage oraux).
EXTRA_STOPWORDS = {
    # Tics de langage et interjections
    "ouais", "ouai", "hey", "yeah", "yo", "oh", "ah", "eh", "hein", "bah",
    "uh", "huh", "na", "lala", "ooh", "woah", "hmm", "wesh",
    # Mots creux qui passent pour des noms ou des adjectifs
    "truc", "chose", "machin", "genre", "quoi", "gros", "mec", "putain",
    "fois", "coup", "peu", "rien", "tout", "tous", "chaque", "meme",
    "vrai", "faux", "bon", "mauvais", "petit", "grand", "autre", "tel",
    "thing", "stuff", "gonna", "wanna", "baby", "right", "wrong", "good",
    "bad", "little", "big", "own", "sure", "okay", "kind", "way", "lot",
    "one", "two", "back", "just", "gotta",
}


# --- spaCy ------------------------------------------------------------------

_nlp_cache: dict[str, object] = {}

# Modèles légers : ~15 Mo chacun, suffisants pour repérer noms et adjectifs.
SPACY_MODELS = {"fr": "fr_core_news_sm", "en": "en_core_web_sm"}


def _load_nlp(language: str):
    """Charge le modèle spaCy correspondant à la langue, avec repli sur l'anglais."""
    import spacy

    code = language if language in SPACY_MODELS else "en"
    if code in _nlp_cache:
        return _nlp_cache[code]

    try:
        _nlp_cache[code] = spacy.load(SPACY_MODELS[code])
    except OSError:
        raise KeywordError(
            f"Le modèle spaCy « {SPACY_MODELS[code]} » n'est pas installé.\n"
            f"Installe-le avec : python -m spacy download {SPACY_MODELS[code]}"
        )
    return _nlp_cache[code]


def _strip_accents(text: str) -> str:
    """Retire les accents pour que « été » et « ete » trouvent la même entrée."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _normalise(word: str) -> str:
    return _strip_accents(word.lower().strip())


def theme_general(segments: list[dict], language: str = "fr") -> str | None:
    """Trouve l'ambiance dominante de la chanson, tous couplets confondus.

    Pourquoi c'est nécessaire : chaque segment est traité isolément, donc un
    vers mal transcrit ou sans mot concret tombait sur un visuel générique
    sans rapport avec le reste — d'où des clips qui partaient dans tous les sens.

    En comptant les thèmes sur l'ensemble des paroles, on obtient une ambiance
    de fond qui sert de repli cohérent : mieux vaut une image dans l'esprit de
    la chanson qu'un fond abstrait pris au hasard.
    """
    from collections import Counter

    nlp = _load_nlp(language)
    compteur: Counter[str] = Counter()

    for segment in segments:
        _, requete = _query_for_text(nlp, segment["text"])
        if requete:
            compteur[requete] += 1

    if not compteur:
        return None

    # Un thème qui revient au moins deux fois caractérise la chanson.
    requete, occurrences = compteur.most_common(1)[0]
    return requete if occurrences >= 2 else None


def build_queries(segments: list[dict], language: str = "fr") -> list[dict]:
    """Ajoute une clé "query" à chaque segment : la recherche vidéo à lancer.

    Renvoie une nouvelle liste, les segments d'origine ne sont pas modifiés.
    Chaque segment reçoit aussi "keyword" (le mot des paroles qui a servi),
    utile pour expliquer le résultat dans l'interface.
    """
    nlp = _load_nlp(language)
    fallback_index = 0
    results = []

    # Ambiance dominante de la chanson : sert de repli quand un vers ne donne
    # rien d'exploitable, pour garder une cohérence visuelle d'ensemble.
    ambiance = theme_general(segments, language)

    for segment in segments:
        keyword, query = _query_for_text(nlp, segment["text"])

        if query is None:
            if ambiance:
                # Une variante de l'ambiance, pour rester dans le ton sans
                # servir exactement la même image qu'ailleurs.
                query = ambiance
            else:
                query = FALLBACK_QUERIES[fallback_index % len(FALLBACK_QUERIES)]
                fallback_index += 1
            keyword = keyword or ""

        enriched = dict(segment)
        enriched["keyword"] = keyword
        enriched["query"] = query
        results.append(enriched)

    return results


def _query_for_text(nlp, text: str) -> tuple[str, str | None]:
    """Trouve le meilleur mot-clé d'une phrase et sa requête vidéo.

    Renvoie (mot des paroles, requête) — la requête vaut None si rien
    d'exploitable n'a été trouvé, c'est alors à l'appelant de choisir un repli.
    """
    # On retire les apostrophes de l'oral ("j'suis", "l'amour") qui perturbent
    # l'analyse, en gardant le mot qui suit.
    cleaned = re.sub(r"\b[a-zA-Zàâäéèêëîïôöùûüç]'", " ", text)

    document = nlp(cleaned)

    # On classe les mots en deux paniers. Les noms et adjectifs d'abord : ce
    # sont eux qui décrivent une scène. Les verbes servent de session de
    # rattrapage, car beaucoup de vers n'ont aucun nom concret
    # (« let me hear you whisper » n'a que des verbes).
    nouns: list[tuple[str, str]] = []   # (mot d'origine, forme normalisée)
    verbs: list[tuple[str, str]] = []

    for token in document:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        normalised = _normalise(token.lemma_ or token.text)
        if len(normalised) < 3 or normalised in EXTRA_STOPWORDS:
            continue
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            nouns.append((token.text, normalised))
        elif token.pos_ == "VERB":
            verbs.append((token.text, normalised))

    # Priorité 1 : un nom ou adjectif présent dans le dictionnaire visuel.
    for original, normalised in nouns:
        if normalised in VISUAL_MAP:
            return original, VISUAL_MAP[normalised]

    # Priorité 2 : un verbe présent dans le dictionnaire visuel.
    for original, normalised in verbs:
        if normalised in VISUAL_MAP:
            return original, VISUAL_MAP[normalised]

    # Priorité 3 : le mot brut, mais seulement si les paroles sont en anglais.
    # Un mot français inconnu du dictionnaire ne donnerait rien sur Pexels,
    # qui est indexé en anglais.
    if nouns and nlp.lang == "en":
        original, normalised = nouns[0]
        return original, f"{normalised} cinematic"

    return (nouns[0][0] if nouns else ""), None
