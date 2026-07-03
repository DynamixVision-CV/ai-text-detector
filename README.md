# Scriptoire — Détecteur de texte IA

Application web complète pour estimer, paragraphe par paragraphe, la probabilité
qu'un document (mémoire, thèse, rapport) ait été généré par une IA. Combine :

- **Perplexité** via un petit modèle de langage local (`distilgpt2` par défaut)
- **Burstiness** (variation du rythme des phrases)
- **Répétition lexicale** (n-grammes répétés)

Le tout combiné en un score 0–100 par segment, affiché sur le document surligné,
avec un score global. Conçu pour tenir des documents de ~100 pages via un
découpage en segments d'environ 150 mots chacun.

⚠️ **Ce n'est pas un outil de preuve.** Comme tous les détecteurs de ce type
(Turnitin, GPTZero, Compilatio…), il produit une estimation statistique avec
un taux de faux positifs non négligeable. À utiliser comme point de départ
pour une conversation avec l'étudiant·e, jamais comme seule base d'une
sanction. Voir `backend/app/scoring.py` pour les seuils de calibration.

## Architecture

```
ai-text-detector/
├── backend/
│   ├── app/
│   │   ├── main.py          # API FastAPI (upload, jobs, résultats)
│   │   ├── extraction.py    # PDF / DOCX / TXT -> texte brut
│   │   ├── chunking.py      # découpage en segments analysables
│   │   ├── heuristics.py    # burstiness, répétition (sans IA)
│   │   ├── perplexity.py    # scoring via LM local (transformers)
│   │   ├── scoring.py       # combinaison des scores + verdict
│   │   └── models.py        # schémas Pydantic
│   ├── static/               # frontend (HTML/CSS/JS vanilla, même conteneur)
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

Le frontend est servi directement par FastAPI (`StaticFiles`) — un seul
conteneur à déployer, pas de build step séparé.

## Lancer en local

```bash
git clone <ton-repo>
cd ai-text-detector
docker compose up --build
```

Puis ouvre `http://localhost:8000`.

### Sans Docker

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Le premier appel à `/api/analyze` télécharge automatiquement les poids de
`distilgpt2` (~350 Mo) depuis Hugging Face — il faut une connexion internet
au premier lancement. Si le modèle ne peut pas se charger, l'app continue de
fonctionner en mode dégradé (heuristiques seules, sans perplexité).

## Déploiement

**Render / Railway / Fly.io** : ces plateformes détectent le `Dockerfile`
automatiquement — connecte simplement le repo GitHub et déploie. Prévois au
moins 1–2 Go de RAM (le modèle de langage tient en mémoire).

**GitHub Actions → GHCR** : le workflow fourni (`.github/workflows/ci.yml`)
lance les tests à chaque push. Tu peux étendre ce fichier avec un job de
build+push d'image Docker vers GitHub Container Registry si tu veux
déployer automatiquement.

## Changer le modèle de perplexité

```bash
export PERPLEXITY_MODEL=gpt2  # modèle plus gros, plus précis, plus lent
```

Tout modèle causal compatible `transformers.AutoModelForCausalLM` fonctionne.

## Limites connues

- Le score est calibré à la main (`PPL_LOW`/`PPL_HIGH` dans `scoring.py`), pas
  appris sur un corpus étiqueté — à ajuster si tu observes trop de faux
  positifs/négatifs sur tes propres documents.
- Les PDF scannés (images) ne contiennent pas de texte extractible ; il
  faudrait ajouter une étape OCR (ex. `pytesseract`) pour les gérer.
- Le job store est en mémoire : redémarrer le serveur efface les jobs en
  cours. Pour un usage multi-instance, remplace par Redis ou une base.

## Licence

MIT — voir `LICENSE`.
