# spine_annotator

A python library including pretrained models in order to annotate vertebrae and IVDs (intevertebral discs) on small field-of-view MRIs. This was created as part of my [master's thesis](https://brutenis.net/master-thesis) which was submitted on 9 November 2023.

![](media/showcase.png)


## Installation

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```
