# Deploying the SFR-Net project page

The website is a dependency-free static site. GitHub Pages can serve it directly from this `docs` directory.

1. Commit and push the `docs/` directory to the repository's default branch (normally `main`).
2. Open the repository on GitHub and go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the `main` branch and the `/docs` folder, then click **Save**.
5. After GitHub finishes the first deployment, open `https://chuyuzhong.github.io/SFR-Net/`.

If the repository is hosted under a different GitHub account, replace `chuyuzhong` in the URL with that account name. All assets use relative paths, so no source edits are required for standard project-site deployment.
