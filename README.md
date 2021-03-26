# Installation

It is recommended to create a virtual environmnent to run the code of this repository. This will avoid package conflicts with other python projects.

```
python -m venv path_for_the_venv
source path_for_the_venv
pip install --upgrade wheel
```

or if you use anaconda, you can create the environnement as follows:

```
conda create --name myenv python=3.8
```

Once the environnement is created, activate the environment (through `source path_for_the_venv` or through `conda activate myenv`), clone the repository:

git clone https://github.com/jokteur/epfl-master-project

Install the package with the following command

```
cd epfl-master-project
pip install -e .
```

this will install the package from the folder. If changes are made (new commits), then the package will be automatically updated.

# Use the interactive widget

Once the package is installed, before launching the app, make sure that your environment is activated. Then simply type :

```
markov-emdedding-widget
```

this will open a window in your browser, and you can start to interact with the model.
