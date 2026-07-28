"""Recherche et téléchargement des vidéos de fond (Pexels et Pixabay).

Trois exigences qui font la différence entre un clip regardable et une vidéo
qui sent le robot :

1. Format vertical d'abord. Un plan tourné en 16:9 recadré en 9:16 perd les
   deux tiers de l'image, souvent le sujet avec. On privilégie donc les clips
   déjà verticaux et on ne recadre qu'à défaut.

2. Anti-répétition stricte. Les refrains reviennent, donc les mêmes mots-clés
   reviennent, donc le même plan reviendrait cinq fois. On mémorise ce qui a
   déjà servi dans la vidéo en cours et on ne le ressert jamais.

3. Repli obligatoire. Une recherche qui ne donne rien ne doit jamais arrêter le
   montage : on retombe sur un visuel générique.
"""

import random
from pathlib import Path

import requests

from modules import config, keywords

PEXELS_API = "https://api.pexels.com/videos/search"
PIXABAY_API = "https://pixabay.com/api/videos/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia impose un User-Agent descriptif : un agent générique reçoit un 429.
COMMONS_HEADERS = {
    "User-Agent": (
        "clip-paroles-portfolio/1.0 "
        "(https://huggingface.co/spaces; projet de démonstration portfolio)"
    )
}


class VideoError(Exception):
    """Erreur de récupération vidéo, message affichable dans l'interface."""


def has_any_key() -> bool:
    """Vrai si au moins une VRAIE banque d'illustration est configurée.

    Sans clé, l'application fonctionne quand même grâce à Wikimedia Commons,
    mais le rendu est nettement moins bon : ce n'est pas une banque de plans
    d'illustration.
    """
    return bool(config.PEXELS_API_KEY or config.PIXABAY_API_KEY)


def missing_key_message() -> str:
    return (
        "Aucune clé de banque vidéo configurée : les fonds viendront de "
        "Wikimedia Commons, dont la qualité est inégale et le format presque "
        "toujours paysage.\n"
        "Pour un rendu nettement meilleur, crée une clé (gratuit, immédiat) :\n"
        "  • Pexels  : https://www.pexels.com/api/new/\n"
        "  • Pixabay : https://pixabay.com/api/docs/\n"
        "puis renseigne PEXELS_API_KEY ou PIXABAY_API_KEY."
    )


# --- Recherche sur chaque banque --------------------------------------------


def search_pexels(query: str, per_page: int = 15) -> list[dict]:
    """Cherche des vidéos verticales sur Pexels. Renvoie [] en cas de souci."""
    if not config.PEXELS_API_KEY:
        return []

    try:
        response = requests.get(
            PEXELS_API,
            headers={"Authorization": config.PEXELS_API_KEY},
            params={
                "query": query,
                "per_page": per_page,
                "orientation": "portrait",
                "size": "medium",
            },
            timeout=30,
        )
    except requests.RequestException:
        return []

    if response.status_code == 401:
        raise VideoError("Clé Pexels refusée. Vérifie PEXELS_API_KEY.")
    if response.status_code == 429:
        # Quota atteint : on ne bloque pas, Pixabay prendra le relais.
        return []
    if response.status_code != 200:
        return []

    try:
        payload = response.json()
    except ValueError:
        return []

    candidates = []
    for video in payload.get("videos", []):
        best_file = _pick_pexels_file(video.get("video_files", []))
        if not best_file:
            continue
        candidates.append(
            {
                "source": "pexels",
                "id": f"pexels-{video.get('id')}",
                "url": best_file["link"],
                "width": best_file.get("width") or 0,
                "height": best_file.get("height") or 0,
                "duration": float(video.get("duration") or 0),
                "credit": video.get("user", {}).get("name", "Pexels"),
                "page": video.get("url", ""),
            }
        )
    return candidates


def _pick_pexels_file(video_files: list[dict]) -> dict | None:
    """Choisit le fichier au bon compromis qualité / poids.

    On vise une hauteur proche de 1920 : en dessous l'image sera étirée,
    au-dessus (4K) on télécharge 200 Mo pour rien.
    """
    usable = [
        file
        for file in video_files
        if file.get("link")
        and file.get("file_type") == "video/mp4"
        and (file.get("height") or 0) >= 720
    ]
    if not usable:
        return None
    return min(usable, key=lambda file: abs((file.get("height") or 0) - 1920))


def search_pixabay(query: str, per_page: int = 20) -> list[dict]:
    """Cherche des vidéos sur Pixabay. Renvoie [] en cas de souci."""
    if not config.PIXABAY_API_KEY:
        return []

    try:
        response = requests.get(
            PIXABAY_API,
            params={
                "key": config.PIXABAY_API_KEY,
                "q": query,
                "per_page": per_page,
                "safesearch": "true",
            },
            timeout=30,
        )
    except requests.RequestException:
        return []

    if response.status_code in (400, 401, 403):
        raise VideoError("Clé Pixabay refusée. Vérifie PIXABAY_API_KEY.")
    if response.status_code != 200:
        return []

    try:
        payload = response.json()
    except ValueError:
        return []

    candidates = []
    for hit in payload.get("hits", []):
        files = hit.get("videos", {})
        best_file = files.get("large") or files.get("medium") or files.get("small")
        if not best_file or not best_file.get("url"):
            continue
        candidates.append(
            {
                "source": "pixabay",
                "id": f"pixabay-{hit.get('id')}",
                "url": best_file["url"],
                "width": best_file.get("width") or 0,
                "height": best_file.get("height") or 0,
                "duration": float(hit.get("duration") or 0),
                "credit": hit.get("user", "Pixabay"),
                "page": hit.get("pageURL", ""),
            }
        )
    return candidates


def search_wikimedia(query: str, limit: int = 12) -> list[dict]:
    """Filet de sécurité : cherche des vidéos libres sur Wikimedia Commons.

    Cette source ne demande AUCUNE clé API. Elle sert à deux choses :
      - l'application produit quelque chose dès l'installation, avant même
        d'avoir créé le moindre compte ;
      - si Pexels et Pixabay ne renvoient rien (quota, panne, requête trop
        exotique), le montage continue au lieu d'échouer.

    Ce n'est PAS la source principale : Commons contient des documentaires et
    des captations, pas des plans d'illustration. La qualité est inégale et le
    format quasiment toujours paysage. D'où le score volontairement bas.

    Les fichiers d'origine pèsent souvent 50 à 2000 Mo. On utilise donc les
    versions transcodées que Commons génère automatiquement (~1 à 3 Mo).
    """
    try:
        response = requests.get(
            COMMONS_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f"filetype:video {query}",
                "gsrnamespace": "6",       # espace de noms « Fichier »
                "gsrlimit": str(limit),
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "format": "json",
            },
            headers=COMMONS_HEADERS,
            timeout=30,
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    try:
        pages = response.json().get("query", {}).get("pages", {})
    except ValueError:
        return []

    candidates = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        url = info.get("url")
        if not url:
            continue

        transcode = _commons_transcode_url(url)
        if not transcode:
            continue

        candidates.append(
            {
                "source": "wikimedia",
                "id": f"wikimedia-{page.get('pageid')}",
                "url": transcode,
                "width": info.get("width") or 0,
                "height": info.get("height") or 0,
                # Commons ne renvoie pas la durée ici ; 0 signifie « inconnue »,
                # et probe_video_duration la mesurera après téléchargement.
                "duration": 0.0,
                "credit": "Wikimedia Commons",
                "page": page.get("title", ""),
            }
        )
    return candidates


def _commons_transcode_url(original_url: str) -> str | None:
    """Construit l'URL de la version allégée d'une vidéo Commons.

    Commons range ses transcodes selon un chemin prévisible :
        .../commons/a/ab/Fichier.webm
        .../commons/transcoded/a/ab/Fichier.webm/Fichier.webm.480p.vp9.webm

    On ne vérifie pas que le fichier existe : une requête HEAD par candidat
    coûterait plus cher que l'échec occasionnel du téléchargement, que
    find_and_download() sait déjà rattraper en passant au candidat suivant.
    """
    if "/commons/" not in original_url:
        return None
    base, _, filename = original_url.rpartition("/")
    dossier = base.replace("/commons/", "/commons/transcoded/", 1)
    return f"{dossier}/{filename}/{filename}.480p.vp9.webm"


def _score(candidate: dict) -> float:
    """Note un candidat : le vertical d'abord, puis une durée confortable.

    Note haute = meilleur candidat.
    """
    width = candidate["width"] or 1
    height = candidate["height"] or 1
    ratio = height / width

    if ratio >= 1.6:
        score = 100.0        # déjà en 9:16 ou proche : idéal
    elif ratio >= 1.0:
        score = 60.0         # carré ou portrait doux : recadrage acceptable
    else:
        score = 20.0         # paysage : on perd les bords, dernier choix

    # Un plan trop court oblige à le boucler, ce qui se voit.
    if candidate["duration"] >= config.MAX_SHOT_SECONDS:
        score += 10
    elif candidate["duration"] >= config.MIN_SHOT_SECONDS:
        score += 5

    # Wikimedia n'est qu'un filet : ses vidéos passent toujours après celles
    # des vraies banques d'illustration, même quand leur format est meilleur.
    if candidate["source"] == "wikimedia":
        score -= 200

    # Décoder de la 4K demande beaucoup de mémoire, pour un résultat qui sera
    # de toute façon réduit en 1080x1920. Sur un conteneur limité, c'est ce
    # décodage qui fait tuer ffmpeg : on écarte donc les très grosses sources.
    memoire = config.memoire_disponible_mo()
    if memoire is not None and memoire < 1500:
        pixels = (candidate["width"] or 0) * (candidate["height"] or 0)
        if pixels > 2_500_000:      # au-delà d'environ 1080x1920
            score -= 50
        if pixels > 6_000_000:      # 4K et au-delà
            score -= 150

    return score


# --- Sélection avec mémoire des clips déjà utilisés --------------------------


class ClipFinder:
    """Trouve un clip par requête, sans jamais réutiliser le même dans une vidéo.

    Un objet ClipFinder = une vidéo en cours de fabrication.
    """

    def __init__(self, download_dir: Path):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.used_ids: set[str] = set()
        # Cache des recherches : les refrains relancent la même requête,
        # inutile de rappeler l'API à chaque fois.
        self._search_cache: dict[str, list[dict]] = {}
        self.credits: list[str] = []

    def _search(self, query: str) -> list[dict]:
        if query in self._search_cache:
            return self._search_cache[query]

        found = search_pexels(query) + search_pixabay(query)

        # Wikimedia n'est interrogé que si les vraies banques n'ont rien donné :
        # inutile de lui faire une requête réseau quand Pexels a répondu.
        if not found:
            found = search_wikimedia(query)

        found.sort(key=_score, reverse=True)
        self._search_cache[query] = found
        return found

    def find(self, query: str) -> dict | None:
        """Renvoie un candidat non encore utilisé, ou None si vraiment rien.

        Ordre d'essai : la requête demandée, puis des replis génériques.
        """
        attempts = [query] + list(keywords.FALLBACK_QUERIES)

        for attempt in attempts:
            for candidate in self._search(attempt):
                if candidate["id"] in self.used_ids:
                    continue
                self.used_ids.add(candidate["id"])
                candidate["query_used"] = attempt
                return candidate

        return None

    def download(self, candidate: dict) -> Path | None:
        """Télécharge un candidat. Renvoie None si le téléchargement échoue."""
        destination = self.download_dir / f"{candidate['id']}.mp4"
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        # Wikimedia refuse les User-Agent génériques (429).
        headers = COMMONS_HEADERS if candidate["source"] == "wikimedia" else {}

        try:
            response = requests.get(
                candidate["url"], timeout=120, stream=True, headers=headers
            )
            response.raise_for_status()
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    file.write(chunk)
        except (requests.RequestException, OSError):
            destination.unlink(missing_ok=True)
            return None

        if destination.stat().st_size < 1024:
            destination.unlink(missing_ok=True)
            return None

        credit = f"{candidate['credit']} ({candidate['source']})"
        if credit not in self.credits:
            self.credits.append(credit)
        return destination

    def find_and_download(self, query: str) -> tuple[Path, dict] | None:
        """Cherche puis télécharge, en réessayant si un téléchargement échoue."""
        for _ in range(4):
            candidate = self.find(query)
            if candidate is None:
                return None
            path = self.download(candidate)
            if path is not None:
                return path, candidate
        return None


def probe_video_duration(path: Path) -> float:
    """Durée réelle d'un fichier vidéo téléchargé (les API mentent parfois)."""
    import json
    import subprocess

    try:
        command = [
            config.ffprobe_path(), "-v", "error",
            "-print_format", "json", "-show_format",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, timeout=60)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        # Durée inconnue : l'appelant traitera le clip comme « assez long ».
        # ffprobe absent est signalé plus tôt, à la validation de l'audio.
        return 0.0


def pick_start_offset(clip_duration: float, needed: float) -> float:
    """Choisit où démarrer dans un clip pour éviter de toujours prendre le début.

    Les premières images d'un stock shot sont souvent les moins intéressantes
    (fondu, mise au point). Si le clip est plus long que nécessaire, on pioche
    un point de départ au hasard dans la partie utilisable.
    """
    marge = clip_duration - needed
    if marge <= 0.5:
        return 0.0
    return round(random.uniform(0, min(marge, clip_duration * 0.5)), 2)
