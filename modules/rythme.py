"""Détection des temps forts d'un morceau, pour caler les coupes dessus.

Pourquoi c'est ce qui change tout
---------------------------------
Un plan qui change au hasard donne un diaporama. Un plan qui change **pile sur
le temps** donne un edit : le spectateur ne sait pas pourquoi, mais il le
ressent. C'est la différence la plus nette entre un montage automatique et un
montage fait par quelqu'un qui écoute la musique.

Comment on les trouve, sans bibliothèque lourde
-----------------------------------------------
Pas de `librosa` : il tire `numba`, `scipy` et une centaine de mégaoctets de
dépendances, pour un hébergement déjà à l'étroit. La méthode retenue tient en
trois étapes, avec le seul `numpy` déjà présent :

1. **Enveloppe d'énergie** — on découpe le signal en fenêtres de ~11 ms et on
   mesure l'énergie de chacune. Une frappe de batterie, une basse, un accord
   plaqué : tout cela fait monter l'énergie d'un coup.

2. **Détection des montées** — on ne garde que les augmentations par rapport à
   la fenêtre précédente (*spectral flux* simplifié). Ce qui compte, c'est le
   moment où le son *démarre*, pas celui où il est fort.

3. **Sélection des pics** — un pic doit dépasser la moyenne locale d'une marge
   donnée, et deux pics ne peuvent pas être trop rapprochés, sinon on
   détecterait chaque nuance plutôt que la pulsation.

Le résultat n'est pas un suivi de tempo au sens musical : c'est une liste
d'instants où « il se passe quelque chose ». C'est exactement ce qu'il faut
pour poser une coupe.
"""

import numpy as np

from modules import config

# Fenêtre d'analyse. 512 échantillons à 16 kHz = 32 ms, assez court pour
# distinguer deux frappes rapprochées, assez long pour ne pas suivre le bruit.
TAILLE_FENETRE = 512
FREQUENCE = 16000

# Deux temps forts plus rapprochés que ça viennent de la même frappe.
ECART_MINIMAL_SECONDES = 0.28

# De combien un pic doit dépasser la moyenne locale pour compter.
SEUIL_RELATIF = 1.35


def detecter_temps_forts(audio: np.ndarray, decalage: float = 0.0) -> list[float]:
    """Renvoie les instants (en secondes) où la musique marque un temps fort.

    `audio` : signal mono à 16 kHz, tel que le fournit faster-whisper.
    `decalage` : à ajouter aux instants, si `audio` est un extrait.
    """
    if audio is None or len(audio) < TAILLE_FENETRE * 4:
        return []

    # 1. Enveloppe d'énergie, fenêtre par fenêtre.
    nombre = len(audio) // TAILLE_FENETRE
    fenetres = audio[: nombre * TAILLE_FENETRE].reshape(nombre, TAILLE_FENETRE)
    energie = np.sqrt(np.mean(fenetres.astype(np.float32) ** 2, axis=1))

    if energie.max() <= 0:
        return []

    # 2. On ne garde que les montées : un temps fort, c'est un son qui démarre.
    montees = np.diff(energie, prepend=energie[0])
    montees[montees < 0] = 0

    if montees.max() <= 0:
        return []

    # 3. Comparaison à la moyenne locale (environ une seconde de contexte),
    # pour rester efficace autant sur un couplet calme qu'un refrain saturé.
    largeur = max(int(FREQUENCE / TAILLE_FENETRE), 3)
    noyau = np.ones(largeur) / largeur
    moyenne_locale = np.convolve(montees, noyau, mode="same")

    candidats = montees > (moyenne_locale * SEUIL_RELATIF)

    secondes_par_fenetre = TAILLE_FENETRE / FREQUENCE
    ecart_minimal = int(ECART_MINIMAL_SECONDES / secondes_par_fenetre)

    temps_forts: list[float] = []
    derniere = -ecart_minimal
    for index in np.flatnonzero(candidats):
        if index - derniere < ecart_minimal:
            continue
        derniere = index
        temps_forts.append(round(index * secondes_par_fenetre + decalage, 3))

    return temps_forts


def analyser_ambiance(audio: np.ndarray, temps_forts: list[float]) -> dict:
    """Décrit le caractère d'un morceau à partir du son lui-même.

    Sans paroles, il n'y a aucun mot pour deviner de quoi parle la musique.
    Mais le signal, lui, dit beaucoup. Trois mesures suffisent à distinguer
    un beat de trap d'une nappe mélancolique :

    - **le tempo**, déduit de l'espacement des temps forts. Rapide appelle du
      mouvement, lent appelle de la contemplation.

    - **la brillance** (centroïde spectral) : où se situe le centre de gravité
      des fréquences. Des aigus dominants sonnent clair et ouvert ; des graves
      dominants sonnent lourd et nocturne. C'est la mesure qui sépare le mieux
      une guitare acoustique d'une 808.

    - **l'intensité**, l'énergie moyenne du signal.

    Renvoie un dictionnaire lisible, dont la clé « ambiance » sert à choisir
    les images.
    """
    if audio is None or len(audio) < FREQUENCE:
        return {"ambiance": "calme_lumineux", "tempo": 0, "brillance": 0, "intensite": 0}

    duree = len(audio) / FREQUENCE

    # --- Tempo, d'après l'écart médian entre deux temps forts ---
    tempo = 0.0
    if len(temps_forts) > 2:
        ecarts = np.diff(np.array(temps_forts))
        ecarts = ecarts[(ecarts > 0.15) & (ecarts < 2.0)]
        if len(ecarts):
            tempo = 60.0 / float(np.median(ecarts))
    if not tempo and duree:
        tempo = len(temps_forts) / duree * 60

    # --- Contraste : la pulsation est-elle marquée, ou est-ce du bourdonnement ? ---
    #
    # Une nappe grave et continue produit des variations d'énergie régulières
    # que la détection prend pour des temps forts — et un morceau contemplatif
    # se retrouve classé « rapide ». Un vrai beat, lui, alterne franchement
    # frappes et silences.
    #
    # On mesure donc l'écart-type de l'énergie rapporté à sa moyenne : élevé
    # quand la musique frappe, faible quand elle bourdonne.
    nombre = len(audio) // TAILLE_FENETRE
    fenetres = audio[: nombre * TAILLE_FENETRE].reshape(nombre, TAILLE_FENETRE)
    energie = np.sqrt(np.mean(fenetres.astype(np.float32) ** 2, axis=1))
    moyenne = float(energie.mean())
    contraste = float(energie.std() / moyenne) if moyenne > 0 else 0.0

    # Sans contraste, la notion de tempo n'a pas de sens : on considère le
    # morceau comme lent, ce qu'il est à l'oreille.
    if contraste < 0.35:
        tempo = min(tempo, 80.0)

    # Erreur d'octave, classique en détection de tempo : on compte les
    # doubles-croches au lieu des temps. Au-delà de 190 BPM, la musique
    # populaire n'existe quasiment plus — c'est qu'on a compté deux fois trop
    # vite. On redescend d'une octave, autant de fois que nécessaire.
    while tempo > 190:
        tempo /= 2

    # --- Brillance : centre de gravité du spectre ---
    # On analyse un extrait au milieu du morceau : le début est souvent une
    # intro peu représentative.
    milieu = len(audio) // 2
    tranche = audio[max(0, milieu - FREQUENCE) : milieu + FREQUENCE]
    spectre = np.abs(np.fft.rfft(tranche * np.hanning(len(tranche))))
    frequences = np.fft.rfftfreq(len(tranche), 1 / FREQUENCE)
    total = spectre.sum()
    brillance = float((spectre * frequences).sum() / total) if total else 0.0

    intensite = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    # --- Classement ---
    # 100 BPM sépare ce qui donne envie de bouger de ce qui invite à écouter.
    # 1200 Hz sépare un mix dominé par les graves d'un mix ouvert sur les aigus.
    rapide = tempo >= 100
    clair = brillance >= 1200

    if rapide and clair:
        ambiance = "intense_lumineux"
    elif rapide:
        ambiance = "intense_sombre"
    elif clair:
        ambiance = "calme_lumineux"
    else:
        ambiance = "calme_sombre"

    return {
        "ambiance": ambiance,
        "tempo": round(tempo),
        "brillance": round(brillance),
        "intensite": round(intensite, 4),
        "contraste": round(contraste, 2),
    }


def caler_sur_temps_forts(
    plans: list[dict], temps_forts: list[float], tolerance: float = 0.35
) -> list[dict]:
    """Déplace les bornes des plans vers le temps fort le plus proche.

    On ne déplace que si le temps fort est à portée (`tolerance`) : sinon on
    casserait le rythme du texte pour suivre une frappe sans rapport. Et on
    garde des plans d'une durée raisonnable — un plan de deux images sur un
    roulement de batterie ne se voit pas, il clignote.
    """
    if not temps_forts or len(plans) < 2:
        return plans

    grille = np.array(temps_forts)
    ajustes = [dict(plan) for plan in plans]

    for index in range(1, len(ajustes)):
        frontiere = ajustes[index]["start"]
        proche = float(grille[np.argmin(np.abs(grille - frontiere))])

        if abs(proche - frontiere) > tolerance:
            continue

        # La nouvelle frontière ne doit pas écraser les plans voisins.
        duree_avant = proche - ajustes[index - 1]["start"]
        duree_apres = ajustes[index]["end"] - proche
        if duree_avant < config.MIN_SHOT_SECONDS * 0.7:
            continue
        if duree_apres < config.MIN_SHOT_SECONDS * 0.7:
            continue

        ajustes[index - 1]["end"] = proche
        ajustes[index]["start"] = proche

    return ajustes
