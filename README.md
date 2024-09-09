foundational-nature-ai
==============================

Explore the current LLMs as a prototype for a Foundational Nature AI capable of informing questions about biodiversity and conservation relevant for real-world decisions.

--------

## Setup

### The environment

To run the notebooks you need to create an environment with the dependencies. There are two options:

#### Docker

If you have [docker](https://docs.docker.com/engine/install/) in your system,
you run a jupyter lab server with:

``` bash
docker compose up --build
```

And if you want to get into the container, use a terminal in jupyter lab,
vscode remote development or run this command:

```shell
docker exec -it foundational_nature_ai_notebooks /bin/bash
```

#### Conda environment

Create the environment with:

``` bash
mamba env create -n foundational_nature_ai -f environment.yml
```

This will create an environment called foundational-nature-ai with a common set of dependencies.

### API Keys
You will need to generate appropriate API keys for the LLMs that you want to use.
Create a .env file and save this in the root directory, e.g.
OPENAI_API_KEY={YOUR-KEY}


### Data

Download the data, eval and training folders from here:
https://drive.google.com/drive/folders/1idjUhOEqePqDYp-l-jUFq2jl7v-8yCm5?usp=sharing

and place in the root of the project


### Model and evalaution setup
Parameters for defining which model to run the evaluation/app with are defined in the params dictionary constructed in setup.py


### Streamlit app

To run the steamlit app execute:
```bash
streamlit run src/app.py
```

This will run a prototype app and launch it in a web browser

