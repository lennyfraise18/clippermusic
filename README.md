---
title: Générateur de clips paroles
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎬 Générateur automatique de clips paroles

Transforme une chanson en **vidéo verticale 1080×1920** avec les paroles en
**karaoké synchronisé mot par mot**, illustrée automatiquement par des extraits
vidéo libres de droit.

Le pipeline : transcription alignée (Whisper) → extraction des thèmes visuels
(spaCy) → recherche d'illustrations (Pexels / Pixabay) → montage (ffmpeg).

---

## Ce que ce projet cherche à démontrer

C'est un projet **portfolio**, pas un service commercial. Il a deux objectifs.

**Un objectif technique :** assembler une chaîne de traitement audio/vidéo
complète — reconnaissance vocale avec alignement temporel, traitement du langage
naturel, appels d'API, montage vidéo automatisé — et la rendre utilisable par
quelqu'un qui clique sur un lien.

**Un objectif métier, qui est le vrai sujet :** une application qui colle des
paroles sur une chanson touche directement au droit d'auteur musical. La plupart
des outils de ce type l'ignorent. Celui-ci en fait une fonctionnalité, avec deux
modes explicites (voir plus bas).

---

## 🎯 La fonctionnalité qui compte : le mode « légal par construction »

L'application propose deux modes, et la différence entre les deux n'est pas
technique mais juridique.

### Mode Creative Commons (par défaut)

Le morceau vient de l'API **Jamendo**, sous licence Creative Commons. La vidéo
produite est **réellement diffusable** sur les réseaux sociaux, à condition de
créditer l'artiste — le crédit, avec la licence exacte et le lien vers le
morceau, est affiché avec le résultat, prêt à recopier.

**Le détail qui compte : toutes les licences Creative Commons ne conviennent
pas.** Les licences **ND** (*No Derivatives*) — `by-nd` et `by-nc-nd` —
autorisent le partage du morceau tel quel mais **interdisent d'en tirer une
œuvre dérivée**. Or c'est exactement ce que produit cette application. Environ
**un tiers** des résultats de l'API Jamendo est dans ce cas, et l'API ne sait pas
les exclure elle-même (`ccnd=false` renvoie zéro résultat). Le filtrage est donc
fait côté application, dans `audio.allows_derivatives()`, et vérifié par
`scripts/test_jamendo.py`. Sans ce filtre, le mode « légal par construction » ne
tiendrait pas sa promesse.

### Mode démo (fichier personnel)

L'utilisateur envoie son propre MP3. Le montage est identique, mais **un
avertissement doit être lu et accepté avant de pouvoir lancer la génération**.
Cet avertissement explique deux risques distincts, souvent confondus :

- **le blocage automatique.** Content ID (YouTube) et les systèmes équivalents
  d'Instagram et TikTok reconnaissent un enregistrement en quelques secondes,
  même ralenti, coupé, ou recouvert de voix. Résultat : retrait, coupure du son,
  ou monétisation au profit de l'ayant droit ;
- **le risque juridique.** Deux droits distincts s'appliquent : celui de la
  **composition** (auteur, compositeur, éditeur) et celui de l'**enregistrement**
  (le master, généralement détenu par le label). Utiliser un extrait sans licence
  porte atteinte aux deux, et la courte durée n'est pas une exception en droit
  français.

Les vidéos de fond viennent de **Pexels** et **Pixabay**, dont les licences
autorisent explicitement l'intégration dans une œuvre dérivée.

---

## 🔑 Comptes à créer (tous gratuits, environ 10 minutes)

| Service | À quoi ça sert | Lien direct | Variable |
|---|---|---|---|
| **Pexels** | Vidéos de fond (source principale) | https://www.pexels.com/api/new/ | `PEXELS_API_KEY` |
| **Pixabay** | Vidéos de fond (secours) | https://pixabay.com/api/docs/ | `PIXABAY_API_KEY` |
| **Jamendo** | Musique Creative Commons | https://devportal.jamendo.com/admin/applications | `JAMENDO_CLIENT_ID` |
| **Hugging Face** | Hébergement du Space | https://huggingface.co/join | — |

**L'application fonctionne sans aucune clé.** Quand ni Pexels ni Pixabay ne
répondent (clé absente, quota atteint, requête trop exotique), les fonds sont
repris sur **Wikimedia Commons**, qui ne demande aucune inscription. C'est un
filet, pas un équivalent : Commons héberge des captations et des documentaires,
pas des plans d'illustration, et presque tout y est en format paysage.

Autrement dit : tu peux tester tout de suite, mais **crée au moins une clé
Pexels ou Pixabay avant de montrer le projet en entretien** — la différence de
rendu est très visible.

Sans clé Jamendo, seul le mode démo fonctionne.

Détails utiles :

- **Pexels** — crée un compte, puis demande une clé sur la page ci-dessus. Elle
  est délivrée immédiatement. Limite gratuite : 200 requêtes/heure, largement
  suffisant (une vidéo consomme entre 5 et 20 requêtes).
- **Pixabay** — la clé apparaît directement sur la page de documentation une fois
  connecté. Limite : 100 requêtes/minute.
- **Jamendo** — crée une application dans le portail développeur ; c'est le
  **Client ID** qu'il faut copier, pas le Client Secret.

---

## 🚀 Installation en local (Windows)

Toutes les commandes sont à copier-coller telles quelles, depuis un terminal
ouvert dans le dossier du projet.

### 1. Installer ffmpeg

```bash
winget install Gyan.FFmpeg
```

**Ferme puis rouvre ton terminal** après cette commande, sinon `ffmpeg` reste
introuvable. Pour vérifier :

```bash
ffmpeg -version
```

### 2. Créer l'environnement Python

Python **3.12** est requis (3.13 et 3.14 ne sont pas encore compatibles avec
PyTorch).

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe -m spacy download fr_core_news_sm
```

```bash
.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

### 3. Renseigner les clés API

Copie `.env.example` en `.env` :

```bash
copy .env.example .env
```

Puis ouvre `.env` dans un éditeur de texte et colle tes clés après les `=`.

### 4. Vérifier que tout est en place

```bash
.venv\Scripts\python.exe scripts\check_env.py
```

Ce script contrôle ffmpeg, l'encodage, **les polices**, le fichier de test et les
clés API. Tant qu'il affiche une ligne `[FAIL]`, inutile d'aller plus loin.

### 5. Lancer l'application

```bash
.venv\Scripts\python.exe app.py
```

Puis ouvre **http://localhost:7860** dans un navigateur.

---

## 🧪 Tests

Chaque brique se teste séparément — c'est ce qui permet de savoir *quelle* étape
casse quand quelque chose casse.

```bash
.venv\Scripts\python.exe scripts\check_env.py
```
Environnement : ffmpeg, encodeur, polices, clés API.

```bash
.venv\Scripts\python.exe scripts\fetch_test_audio.py
```
Télécharge le morceau de test (domaine public, Wikimedia Commons).

```bash
.venv\Scripts\python.exe scripts\test_transcribe.py
```
Transcription alignée mot par mot sur le fichier de test.

```bash
.venv\Scripts\python.exe scripts\test_keywords.py
```
Extraction de mots-clés, y compris sur de l'argot et des mots abstraits.

```bash
.venv\Scripts\python.exe scripts\test_videos.py
```
Banques vidéo : recherche, anti-répétition, repli, téléchargement.

```bash
.venv\Scripts\python.exe scripts\test_jamendo.py
```
Mode Creative Commons : recherche, **filtrage des licences No Derivatives**,
téléchargement. **Nécessite le Client ID Jamendo.**

```bash
.venv\Scripts\python.exe scripts\test_montage.py --complet
```
Montage isolé + comparaison chronométrée des deux approches.

```bash
.venv\Scripts\python.exe scripts\test_cas_limites.py
```
Fichier corrompu, instrumental, trop court, sans audio, nettoyage après erreur.

```bash
.venv\Scripts\python.exe scripts\test_pipeline.py
```
Chaîne complète, de l'audio à la vidéo. **Nécessite une clé.**

---

## 🎥 Montage : pourquoi une seule passe

Deux approches étaient possibles, et le choix a été tranché par la mesure, pas
par principe (`scripts/test_montage.py --complet`).

**Approche A — deux passes.** Normaliser chaque clip en 1080×1920 séparément,
concaténer avec le concat demuxer en `-c copy` (aucun réencodage), puis une passe
finale pour incruster les sous-titres.

**Approche B — une seule passe.** Un `filter_complex` unique qui fait
scale + crop + concat + incrustation des sous-titres.

Mesure sur le cas réel (18 plans, 86 s de vidéo, encodeur `h264_qsv`) :

| Approche | Temps | Taille |
|---|---|---|
| **B — une passe** | **8,8 s** | 43,6 Mo |
| A — deux passes | 15,3 s | 40,4 Mo |

**L'approche B est retenue** : 42 % plus rapide, et plus simple à maintenir.

La raison est celle qui était pressentie : le `-c copy` de l'approche A ne fait
économiser que la concaténation, mais la passe finale d'incrustation réencode
**toute** la vidéo. Chaque image est donc encodée deux fois au total. Le gain
apparent du `-c copy` est annulé par le travail supplémentaire.

L'approche A reste dans le code (`editor.render_two_pass`) : elle parallélise la
normalisation des clips et pourrait reprendre l'avantage sur une machine avec
beaucoup de cœurs et un encodeur logiciel lent.

---

## ⚙️ Choix techniques et pièges rencontrés

**`whisper-timestamped` plutôt que `openai-whisper`.** Le karaoké a besoin de
savoir quand chaque *mot* commence et finit. Whisper de base ne fournit que des
timestamps par phrase.

**Pas d'API Genius pour les paroles.** L'API officielle de Genius ne renvoie que
des métadonnées, pas le texte des paroles. Les récupérer supposerait de scraper
leur site, ce que leurs conditions d'utilisation interdisent. Whisper suffit, et
il a l'avantage de fournir l'alignement temporel — que Genius ne donnerait pas.

**Pas de téléchargement YouTube.** YouTube bloque agressivement les IP de
datacenter. Depuis un Space Hugging Face, `yt-dlp` échouerait la plupart du temps.
Une fonctionnalité qui marche une fois sur cinq en démo est pire que pas de
fonctionnalité du tout.

**Le dictionnaire thème → visuel est indispensable.** Chercher « amour » ou
« liberté » sur une banque de vidéos ne donne rien d'exploitable. `modules/keywords.py`
traduit chaque thème abstrait en scène concrète : « solitude » → *empty street at
night*, « douleur » → *rain on window*. Ce dictionnaire sert aussi de traduction,
Pexels et Pixabay étant indexés en anglais.

**Les polices dans le conteneur — le piège n°1.** Si la police nommée dans le
fichier `.ass` n'existe pas dans l'image Docker, ffmpeg incruste des carrés vides,
ou rien du tout, **sans jamais renvoyer d'erreur**. Le `Dockerfile` installe
explicitement `fonts-dejavu-core`, et `scripts/check_env.py` vérifie le rendu en
comptant les pixels de texte réellement dessinés.

**PyTorch en version CPU — le piège n°2.** Sur Linux, `pip install torch` prend
par défaut le wheel CUDA, soit environ 2,5 Go de bibliothèques GPU inutiles sur un
Space gratuit. Le `Dockerfile` force l'index CPU.

**Les hallucinations de Whisper.** Sur du silence ou de l'instrumental, Whisper
invente des paroles. Trois filtres sont appliqués : confiance minimale, détection
des répétitions en boucle (« la la la la »), et rejet des segments absurdement
longs pour leur nombre de mots. Si rien de valable ne subsiste, l'application
affiche « aucune parole détectée » plutôt que d'inventer des sous-titres.

**L'anti-répétition des clips.** Les refrains reviennent, donc les mêmes mots-clés
reviennent. Sans mémoire des clips déjà servis, le même plan reviendrait cinq
fois et la vidéo aurait visiblement l'air générée par un bot. `ClipFinder` mémorise
tout ce qui a servi dans la vidéo en cours.

**Un filet de sécurité sans clé API.** Une démo qui affiche « configure d'abord
trois clés » avant de montrer quoi que ce soit est une démo ratée. Quand Pexels
et Pixabay ne répondent pas — clé absente, quota atteint, requête trop exotique —
l'application se rabat sur **Wikimedia Commons**, qui ne demande aucune
inscription. Les fichiers d'origine y pèsent souvent 50 à 2000 Mo, donc on
utilise les versions transcodées que Commons génère automatiquement (1 à 3 Mo).
Cette source est volontairement notée très bas dans le classement des candidats :
elle ne sert que lorsque rien d'autre n'est disponible.

**Chansons longues.** Au-delà de 90 secondes, l'application ne coupe pas
bêtement le début (souvent une intro instrumentale) : elle cherche la fenêtre de
90 secondes **la plus dense en paroles**.

**Nettoyage garanti.** Les fichiers intermédiaires sont supprimés dans un bloc
`finally`, donc y compris quand une étape échoue. Sans ça, le disque d'un Space
gratuit sature en quelques essais. Le morceau Jamendo téléchargé est supprimé de
la même façon, une fois la vidéo montée.

**Le débit vidéo est plafonné.** À qualité constante, l'encodeur produisait
environ 10 Mbps, soit 140 Mo pour moins de deux minutes — au-delà de ce
qu'Instagram accepte pour un Reel, et pénible à téléverser. Un plafond à 7 Mbps
(`config.MAX_BITRATE`) ramène le fichier à une taille publiable sans perte
visible en 1080×1920.

**Les lignes de sous-titres sont courtes pour une raison mesurable.** À 96 px en
gras, un caractère fait environ 52 px ; la largeur utile étant de 920 px, une
ligne ne peut pas dépasser ~18 caractères. Une limite plus généreuse débordait de
l'écran sur les paroles françaises, dont les mots sont plus longs qu'en anglais.
Le fichier `.ass` utilise en plus `WrapStyle: 0` comme filet, pour qu'un mot
exceptionnellement long passe à la ligne au lieu de sortir du cadre.

---

## 🚢 Déploiement sur Hugging Face Spaces

### 1. Créer le Space

Va sur **https://huggingface.co/new-space** et renseigne :

- **Space name** : `clip-paroles` (ou ce que tu veux)
- **License** : MIT
- **Space SDK** : **Docker** → *Blank*
- **Hardware** : CPU basic (gratuit)
- **Visibility** : Public

### 2. Envoyer le code

Remplace `TON-PSEUDO` par ton nom d'utilisateur Hugging Face :

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "Générateur de clips paroles"
```

```bash
git remote add space https://huggingface.co/spaces/TON-PSEUDO/clip-paroles
```

```bash
git push space main
```

Git demandera ton nom d'utilisateur Hugging Face et, comme mot de passe, un
**token d'accès** à créer ici : https://huggingface.co/settings/tokens
(type *Write*).

### 3. Renseigner les clés API sur le Space

Dans ton Space : **Settings** → **Variables and secrets** → **New secret**.

Ajoute-les en **Secret** (pas en *Variable*, sinon elles seraient publiques) :

- `PEXELS_API_KEY`
- `PIXABAY_API_KEY`
- `JAMENDO_CLIENT_ID`

Le Space redémarre automatiquement.

### 4. Attendre la construction

Le premier build prend **15 à 25 minutes** : l'image installe PyTorch, spaCy et
pré-télécharge le modèle Whisper (~460 Mo). L'onglet *Logs* montre l'avancement.

---

## ⏱️ À savoir pour une démo en entretien

**Le Space gratuit se met en veille après 48 h d'inactivité** et met plusieurs
dizaines de secondes à redémarrer au premier appel. **Ouvre le lien cinq minutes
avant ton entretien** pour le réveiller.

**Compte une à trois minutes de traitement** pour une chanson, sur le CPU d'un
Space gratuit. L'interface affiche l'étape en cours pendant ce temps.

**Le modèle Whisper est pré-téléchargé dans l'image Docker**, donc la première
génération après un réveil ne perd pas de temps à le récupérer.

---

## 📁 Structure du projet

```
app.py                      Interface Gradio et branchements
modules/
  config.py                 Clés API, constantes, détection de l'encodeur
  audio.py                  Validation des fichiers, Jamendo, filtre licences ND
  transcribe.py             Whisper aligné mot par mot, filtres anti-hallucination
  keywords.py               spaCy + dictionnaire thème → visuel
  videos.py                 Pexels / Pixabay / Wikimedia, anti-répétition, replis
  subtitles.py              Génération du .ass karaoké
  editor.py                 Découpage en plans et montage ffmpeg
  pipeline.py               Orchestration et nettoyage garanti
scripts/                    Un script de test par brique
assets/test_song.mp3        Morceau de test (domaine public)
Dockerfile                  Image du Space
```

---

## 📊 Repères de performance

Mesuré en local (Intel avec Quick Sync, modèle Whisper `small`) :

| Étape | Durée |
|---|---|
| Transcription (153 s d'audio) | ~70 s |
| Recherche et téléchargement des 18 clips | 20 à 60 s |
| Montage (18 plans, 86 s de vidéo) | ~9 s |
| **Total, de l'upload à la vidéo** | **~115 s** |

Sur le CPU d'un Space gratuit, sans accélération matérielle, compter environ
deux à trois fois plus.

---

## ✅ État de validation

Ce qui a été vérifié, et comment :

| Contrôle | État | Vérifié par |
|---|---|---|
| Le MP3 de test produit une vidéo verticale complète avec paroles synchronisées | ✅ | `test_pipeline.py` + contrôle visuel des images extraites |
| Les sous-titres s'affichent avec la bonne police, pas des carrés vides | ✅ | `check_env.py` compte les pixels de texte réellement dessinés |
| Une chanson en argot ou avec des anglicismes ne casse pas l'extraction | ✅ | `test_keywords.py` (15 cas difficiles) |
| Un fichier instrumental renvoie un message clair | ✅ | `test_cas_limites.py` cas 5 |
| Une clé API absente ou invalide renvoie un message clair | ✅ | `test_videos.py`, `test_jamendo.py` |
| Aucun clip vidéo n'est utilisé deux fois dans la même sortie | ✅ | `test_pipeline.py` compare les identifiants de clips (18 plans → 18 clips distincts) |
| Les fichiers temporaires sont nettoyés, y compris après une erreur | ✅ | `test_cas_limites.py` cas 6 |
| Le mode Jamendo fonctionne comme alternative complète | ✅ | `test_jamendo.py` + pipeline complet sur un morceau CC |
| Les licences No Derivatives sont écartées | ✅ | `test_jamendo.py` |
| Le Space est déployé et accessible via un lien public | ⬜ | à faire, voir *Déploiement* |
| Le README liste les clés à créer avec liens et commandes | ✅ | ce fichier |

---

## 📄 Licences et crédits

- Code : MIT
- Musique en mode Creative Commons : [Jamendo](https://www.jamendo.com), licence
  affichée avec chaque morceau
- Vidéos de fond : [Pexels](https://www.pexels.com/license/),
  [Pixabay](https://pixabay.com/service/license-summary/) et, en dernier
  recours, [Wikimedia Commons](https://commons.wikimedia.org) (licences libres,
  crédit affiché avec le résultat)
- Fichier de test : *Let Me Call You Sweetheart* (1911), domaine public, via
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Let_Me_Call_You_Sweetheart_(1911).ogg)

Les vidéos produites en **mode démo** utilisent un enregistrement protégé et ne
doivent pas être diffusées publiquement.
