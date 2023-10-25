<a href="https://github.com/alexandrainst/tts_text"><img src="https://github.com/alexandrainst/tts_text/raw/main/gfx/alexandra_logo.png" width="239" height="175" align="right" /></a>
# tts_text

Code for collection/generation of text for tts data collection

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/tts_text/tts_text.html)
[![License](https://img.shields.io/github/license/alexandrainst/tts_text)](https://github.com/alexandrainst/tts_text/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/tts_text)](https://github.com/alexandrainst/tts_text/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-47%25-orange.svg)](https://github.com/alexandrainst/tts_text/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/tts_text/blob/main/CODE_OF_CONDUCT.md)


Developers:

- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)
- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Quick Start

The quickest way to build the dataset is using Docker. With Docker installed, simply
write `make docker` and the final dataset will be built in the `data/processed`
directory, with the individual datasets in `data/raw`.


## Development Setup

To install the project for further development, run the following steps:

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a
   virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

With the project installed, you can build the dataset by running:

```
python src/scripts/build_tts_dataset.py
```


## Project structure
```
.
├── .devcontainer
│   └── devcontainer.json
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── .ruff_cache
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   └── hydra
│       └── job_logging
│           └── custom.yaml
├── data
│   ├── final
│   │   └── .gitkeep
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       └── .gitkeep
├── docs
│   └── .gitkeep
├── gfx
│   ├── .gitkeep
│   └── alexandra_logo.png
├── makefile
├── models
│   └── .gitkeep
├── notebooks
│   └── .gitkeep
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── build_tts_dataset.py
│   │   └── fix_dot_env_file.py
│   └── tts_text
│       ├── __init__.py
│       ├── __pycache__
│       ├── bus_stops_and_stations.py
│       ├── dates.py
│       ├── times.py
│       └── utils.py
└── tests
    ├── __init__.py
    ├── __pycache__
    └── test_dummy.py
```
