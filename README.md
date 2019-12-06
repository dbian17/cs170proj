# project-fa19
CS 170 Fall 2019 Project "Drive the TAs Home"
David, Cindy, Ivy

Requirements
------
Our project requires several free packages to help solve our routing problem.
Use package manager pip to install scikit learn, networkx, NumPy and ORTools. Without these packages the program will fail to run.

NOTE: If using python3, please pip3 install instead of pip install if running into trouble installing these packages.

```bash
pip install -U scikit-learn
```
```bash
pip install numpy
```
```bash
pip install networkx
```
```bash
python -m pip install --upgrade --user ortools
```
Now you are ready!

Usage
-----
After you've installed the above packages, our code is ready to be run!

We've created a script called "run.py" that runs solve_all on all inputs in specified folder. Thus, to run our solver/code, simply follow the below instructions:

1. Place inputs in a folder named "inputs"
2. Open terminal and enter python3 run.py

All outputs should be put into a folder named "output."
