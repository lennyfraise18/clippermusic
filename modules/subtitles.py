"""Génération du fichier de sous-titres karaoké au format .ass.

Pourquoi .ass et pas .srt : le SRT ne sait afficher qu'un bloc de texte fixe.
L'ASS (Advanced SubStation Alpha) gère la balise \\k, qui colore les mots un par
un au moment où ils sont chantés. C'est exactement l'effet karaoké recherché,
et ffmpeg sait l'incruster nativement — aucune bibliothèque supplémentaire.

Comment fonctionne \\k :
  - le texte pas encore chanté s'affiche en SecondaryColour ;
  - chaque {\\kN} passe le mot suivant en PrimaryColour après N centièmes de seconde.
Donc PrimaryColour = couleur de surbrillance, SecondaryColour = couleur de repos.

Format des couleurs ASS : &HAABBGGRR — alpha, bleu, vert, rouge, en hexadécimal,
et alpha 00 signifie « opaque ». Ce n'est pas du RGB, c'est du BGR inversé.
"""

from pathlib import Path

from modules import config

# Blanc au repos, jaune vif sur le mot chanté, contour noir épais pour rester
# lisible sur n'importe quelle vidéo de fond.
COLOR_SUNG = "&H00FFFFFF"       # blanc  (PrimaryColour  = déjà chanté)
COLOR_PENDING = "&H0040D4FF"    # jaune  (SecondaryColour = pas encore chanté)
COLOR_OUTLINE = "&H00000000"    # noir
# Ombre violette plutôt que noire : elle rattache le texte à l'identité de
# l'application et détache mieux les lettres sur un fond sombre.
COLOR_SHADOW = "&H80F755A8"     # violet semi-transparent (BGR + alpha)

# Une ligne trop longue déborde de l'écran vertical.
#
# Le calcul, à refaire si on change SUBTITLE_FONT_SIZE : à 96 px en gras, un
# caractère fait environ 52 px de large. La largeur utile est
# 1080 - 2 x 80 (marges) = 920 px, soit à peine 18 caractères. La limite
# précédente (26) débordait donc de l'écran sur les phrases françaises, qui
# ont des mots plus longs qu'en anglais.
MAX_CHARS_PER_LINE = 18
MAX_WORDS_PER_LINE = 4
MAX_LINE_SECONDS = 4.0


def format_time(seconds: float) -> str:
    """Convertit des secondes en H:MM:SS.cc, le format attendu par l'ASS."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:  # arrondi qui déborde
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def escape_text(text: str) -> str:
    """Neutralise les caractères qui ont un sens spécial en ASS."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
    )


def split_into_lines(words: list[dict]) -> list[list[dict]]:
    """Découpe une suite de mots en lignes affichables une par une.

    Trois raisons de couper : trop de caractères, trop de mots, ou trop de temps
    écoulé (une ligne qui reste 8 secondes à l'écran donne l'impression d'un bug).
    """
    lines: list[list[dict]] = []
    current: list[dict] = []

    for word in words:
        if current:
            characters = sum(len(w["text"]) + 1 for w in current) + len(word["text"])
            elapsed = word["end"] - current[0]["start"]
            too_long = (
                characters > MAX_CHARS_PER_LINE
                or len(current) >= MAX_WORDS_PER_LINE
                or elapsed > MAX_LINE_SECONDS
            )
            if too_long:
                lines.append(current)
                current = []
        current.append(word)

    if current:
        lines.append(current)
    return lines


def _karaoke_text(line: list[dict]) -> str:
    """Construit le texte ASS d'une ligne, avec une balise karaoké par mot.

    Deux choix qui font la différence visuelle :

    - `\\kf` plutôt que `\\k` : la couleur **balaie** le mot de gauche à droite
      au lieu de basculer d'un coup. Le résultat suit la voix au lieu de
      clignoter dessus.

    - un léger zoom à l'apparition de la ligne (`\\t` sur `\\fscx`/`\\fscy`) :
      le texte entre en scène au lieu d'apparaître. C'est ce qui donne le
      côté « lyric video » des edits qui tournent.
    """
    duree_ligne = max(line[-1]["end"] - line[0]["start"], 0.1)
    entree = min(int(duree_ligne * 1000 * 0.18), 260)

    parts = [
        # Fondu court, puis passage de 88 % à 100 % de taille : discret, mais
        # suffisant pour que l'oeil accroche la nouvelle ligne.
        f"{{\\fad(90,110)\\fscx88\\fscy88\\t(0,{entree},\\fscx100\\fscy100)}}"
    ]
    cursor = line[0]["start"]

    for word in line:
        # Silence avant le mot : on l'absorbe dans une balise vide, sinon
        # la surbrillance prend de l'avance sur la voix.
        gap = int(round((word["start"] - cursor) * 100))
        if gap > 0:
            parts.append(f"{{\\kf{gap}}}")

        duration = int(round((word["end"] - word["start"]) * 100))
        parts.append(f"{{\\kf{max(duration, 1)}}}{escape_text(word['text'])} ")
        cursor = word["end"]

    return "".join(parts).rstrip()


def build_ass(segments: list[dict], destination: Path) -> Path:
    """Écrit le fichier .ass complet et renvoie son chemin.

    `segments` : la sortie de transcribe.py (temps déjà recalés sur zéro).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
Title: Clip paroles
ScriptType: v4.00+
PlayResX: {config.VIDEO_WIDTH}
PlayResY: {config.VIDEO_HEIGHT}
; WrapStyle 0 = retour à la ligne automatique. C'est un filet : si un seul mot
; très long dépasse malgré le découpage ci-dessus, libass le renvoie à la ligne
; au lieu de le laisser sortir de l'écran (ce que ferait WrapStyle 2).
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{config.SUBTITLE_FONT},{config.SUBTITLE_FONT_SIZE},{COLOR_SUNG},{COLOR_PENDING},{COLOR_OUTLINE},{COLOR_SHADOW},-1,0,0,0,100,100,0,0,1,7,3,2,80,80,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # On rassemble d'abord TOUTES les lignes, tous segments confondus, avant
    # d'écrire quoi que ce soit : la fin d'une ligne dépend de la suivante.
    lines: list[list[dict]] = []
    for segment in segments:
        lines.extend(split_into_lines(segment["words"]))

    events = []
    for index, line in enumerate(lines):
        start = line[0]["start"]

        # On laisse la ligne un peu après le dernier mot, sinon elle disparaît
        # à l'instant exact où on finit de la lire. Mais jamais au-delà du
        # début de la ligne suivante : deux Dialogue qui se chevauchent sont
        # empilés par libass, ce qui affiche deux lignes en même temps et,
        # pire, dans le désordre.
        end = line[-1]["end"] + 0.35
        if index + 1 < len(lines):
            end = min(end, lines[index + 1][0]["start"] - 0.02)
        end = max(end, line[-1]["end"] + 0.01)

        events.append(
            f"Dialogue: 0,{format_time(start)},{format_time(end)},"
            f"Karaoke,,0,0,0,,{_karaoke_text(line)}"
        )

    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return destination
