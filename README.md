# cs180-showcase

Berkeley CS 180 · Computational Photography — project showcase.

The site is the **Light Table** concept: prints laid out on a darkroom light table.
It lives at the repo root and GitHub Pages is configured to publish from `main` `/`.

## Structure

```
index.html               the Light Table — prints on a darkroom light table
project.html             project template (P1 example)
assets/                  css · js · placeholder images
README.md                this file
```

## Working on the site

Edit the files here and open `index.html`, or run a local server from the repo root:

```
python3 -m http.server
```

## Deploy (GitHub Pages)

The site publishes from the repository root, so:

1. Push the repo to GitHub.
2. Repo → **Settings → Pages**.
3. **Source:** *Deploy from a branch* → **Branch:** `main` → **Folder:** `/` (root) → Save.
4. The site builds at `https://<your-username>.github.io/<repo>/`.

The `.nojekyll` file at the root keeps GitHub from Jekyll-processing the site.

## Making a project real

Drop your photos into `assets/images/` (keeping the filenames `p0.svg` … `final.svg`), then
duplicate `project.html` per project and link the cards in `index.html`. Colors, fonts,
and sizes are CSS variables at the top of `assets/css/main.css`.
