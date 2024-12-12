# Final Project of Intelligent Optimization Algorithms and Its Applications

This is a repository containing my codes for the final project of the course *Intelligent Optimization Algorithms and Its Applications* in 2024 fall, Department of Automation, Tsinghua University.

## Environment setup

You can create a `conda` environment if you want:

```bash
conda create -n optim
```

or just use the default `Python` environment. Then install the dependencies:

```bash
pip install numpy matplotlib pyyaml tqdm
```

## Run

```bash
cd /path/to/InteOptimFinalProj
# Example command to test SA on TSP problems
python run_exp.py --prob TSP --alg SA
# Example command to test GA on function optimization problems
python run_exp.py --prob func --alg GA
```