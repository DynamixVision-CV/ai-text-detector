# Scriptoire — Détecteur de texte IA (100% navigateur)

Application web statique qui estime, paragraphe par paragraphe, la probabilité
qu'un document (mémoire, thèse, rapport) ait été généré par une IA.

**Aucun serveur nécessaire.** Tout tourne dans le navigateur de la personne
qui utilise le site :
- Extraction du texte (PDF via `pdf.js`, DOCX via `mammoth.js`)
- Calcul de perplexité via un petit modèle de langage (`distilgpt2`) chargé
  et exécuté directement dans le navigateur (`transformers.js`)
- Heuristiques de rythme (burstiness) et de répétition lexicale

⚠️ **Ce n'est pas un outil de preuve**, comme tous les détecteurs de ce type
(Turnitin, GPTZero, Compilatio…). C'est une estimation statistique avec un
taux de faux positifs non négligeable — à utiliser comme point de départ
pour une conversation, jamais comme seule base d'une sanction.

## Déployer sur GitHub Pages (3 étapes)

1. **Crée un nouveau repository sur GitHub** (public — GitHub Pages gratuit
   nécessite un repo public, sauf si tu as GitHub Pro/Enterprise).

2. **Mets les 3 fichiers de ce dossier à la racine du repo** (`index.html`,
   `style.css`, `app.js`) — glisse-les directement dans l'interface GitHub
   ("Add file" → "Upload files"), ou via `git` :
   ```bash
   git init
   git add index.html style.css app.js README.md
   git commit -m "Scriptoire — détecteur de texte IA"
   git remote add origin https://github.com/TON-USERNAME/TON-REPO.git
   git branch -M main
   git push -u origin main
   ```

3. **Active GitHub Pages** : dans le repo, va dans **Settings** → **Pages**
   (menu de gauche) → sous "Build and deployment", "Source" : choisis
   **"Deploy from a branch"** → Branch : **`main`**, dossier **`/ (root)`** →
   **Save**.

GitHub te donne un lien du type `https://ton-username.github.io/ton-repo/`
en 1 à 2 minutes. C'est tout — pas de build, pas de configuration serveur.

## Tester en local avant de déployer (optionnel)

Comme le site utilise des modules JS (`type="module"`), il faut le servir
via un petit serveur local plutôt que d'ouvrir `index.html` directement
(certains navigateurs bloquent les modules en `file://`) :

```bash
python3 -m http.server 8000
# puis ouvre http://localhost:8000
```

## Notes importantes

- **Premier chargement** : le modèle (~85 Mo) se télécharge depuis un CDN au
  premier passage sur le site, puis reste en cache dans le navigateur — les
  visites suivantes sont quasi instantanées.
- **Vitesse** : un document de ~100 pages prend de quelques secondes à
  1-2 minutes à analyser selon l'appareil (tout tourne sur le processeur de
  l'utilisateur, pas sur un serveur).
- **PDF scannés** (images sans texte) ne fonctionnent pas sans OCR — non
  inclus dans cette version.
- **Vie privée** : comme rien n'est envoyé à un serveur, les documents des
  étudiant·e·s ne quittent jamais leur ordinateur. C'est un vrai avantage
  par rapport à la version avec backend.
- Les seuils de calibration (`PPL_LOW`, `PPL_HIGH`, etc. dans `app.js`) sont
  réglés à la main — ajuste-les si tu observes trop de faux positifs/négatifs.

## Licence

MIT
