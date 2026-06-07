# Install this portal into your ATLC repo

From the root of your existing ATLC repo:

```bash
mkdir -p docs/training_portal
cp -R /path/to/ATLC_UdemyStyle_Training_Portal/docs/training_portal/* docs/training_portal/
open docs/training_portal/index.html

git add docs/training_portal
git commit -m "docs: add Udemy-style ATLC training portal"
git push
```

GitHub Pages setup:

```text
Settings -> Pages
Source: Deploy from a branch
Branch: main or training-portal
Folder: /docs
```

Open:

```text
https://<username>.github.io/<repo-name>/training_portal/
```
