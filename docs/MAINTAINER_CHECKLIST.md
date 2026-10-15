# Publishing and maintaining the repository

Upload the contents of the prepared release folder, including `.github` and
`.streamlit`, to the root of the repository. Use the repository name
`Machine-Learning-Based-Geospatial-Super-Resolution-System`.

After uploading:

1. Keep the repository public if it is intended for public contributions.
2. Enable Issues in repository settings. Bug and feature forms appear automatically
   when `.github/ISSUE_TEMPLATE` is present on the default branch.
3. Check the Actions tab and resolve failures. Included CI configuration is not
   evidence of a successful GitHub run until that run actually finishes.
4. Enable private vulnerability reporting under the repository's security settings.
5. Where available, create a branch ruleset requiring a pull request and passing
   test checks before merging. Choose check names after the first CI run.
6. Review incoming changes and dependency-update pull requests before merging.
   Do not grant contributor write access merely to let someone submit a pull request.
7. Label suitable small issues `good first issue` or `help wanted` after creating
   those labels. Keep reproduction steps and acceptance criteria concrete.

People can fork a public repository and propose changes through pull requests.
The owner retains control over what is merged. The MIT license permits use and
modification; satellite-data and third-party model licenses remain separate.

The included files do not change remote repository settings automatically. Check
GitHub's current documentation if your account's settings layout differs:
[issue and pull request templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates).
