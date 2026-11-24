# Deploying Sphinx documentation to GitHub Pages

## Goal
Host source code along with documentation sources on a public Git repository.

Each time changes are pushed to the repository, a GitHub Action triggers to rebuild the documentation, which is then pushed to a separate branch called ‘gh-pages’.

This setup ensures that the documentation is always up-to-date with the latest changes in the source code.


# Documentation Example Project Setup

This guide walks you through creating a GitHub repository for a documentation project using Sphinx, and deploying it using GitHub Pages and GitHub Actions.

## Step 1: Create a New Repository from a Template

1. Go to the [**documentation-example**](https://github.com/new?template_name=documentation-example&template_owner=coderefinery) project template on GitHub.
2. Create a copy in your namespace and name it `documentation-example`.
2. Create a copy in your namespace and name it `documentation-example`.
3. Ensure to uncheck "Include all branches".
4. Click on **Create a repository**.

## Step 2: Structure of the Repository

- After creating your repository, it will contain a `docs/` directory for Sphinx documentation.
- Your project's source code should be placed under the `src/` directory.

## Step 3: Set Up GitHub Actions

1. Navigate to the `.github/workflows/` directory in your repository.
2. Create a new file named `documentation.yml`.
3. Insert the following configuration:

    ```yaml
    name: documentation

    on: [push, pull_request, workflow_dispatch]

    permissions:
      contents: write

    jobs:
      docs:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
          - uses: actions/setup-python@v3
          - name: Install dependencies
            run: |
              pip install sphinx sphinx_rtd_theme myst_parser
          - name: Sphinx build
            run: |
              sphinx-build docs _build
          - name: Deploy to GitHub Pages
            uses: peaceiris/actions-gh-pages@v3
            if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
            with:
              publish_branch: gh-pages
              github_token: ${{ secrets.GITHUB_TOKEN }}
              publish_dir: _build/
              force_orphan: true
    ```

## Step 4: Enable GitHub Pages

1. In your GitHub repository, go to **Settings** -> **Pages**.
2. Under the "Source" section, select "Deploy from a branch".
3. In the "Branch" section, choose `gh-pages` and `/root` from the dropdown menus.
4. Click save.

## Step 5: Verify the Deployment

- Check that your site is live at `https://<USER>.github.io/documentation-example/` (replace `<USER>` with your GitHub username).

## Step 6: Continuous Documentation Updates

- Make changes to your documentation in the `docs/` directory.
- Commit and push these changes to verify that the GitHub Pages site refreshes automatically.
