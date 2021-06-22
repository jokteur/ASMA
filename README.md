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

For the Rust accelerated code, please install Rust-Lang on your system: https://www.rust-lang.org/. Then, install the Rust setup tool with:
```
pip install -r requirements-dev.txt
```

Once this is done, install the package with the following command

```
cd epfl-master-project
pip install -e .
```

this will install the package from the folder. If changes are made (new commits), then the package will be automatically updated.
