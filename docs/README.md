# Card Enhancer Suite — project site

`index.html` is the project landing page. It is self-contained except for the
artwork in `assets/`, which is binary and therefore fetched separately:

```bash
bash docs/fetch-assets.sh
git add docs/assets && git commit -m "Add generated site artwork" && git push
```

Then enable GitHub Pages: **Settings → Pages → Deploy from a branch → main → /docs**.
The site will be served at `https://nietzsche-ubermensch.github.io/card-enhancer-suite/`.
