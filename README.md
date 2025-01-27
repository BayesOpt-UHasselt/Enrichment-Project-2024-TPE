# Enrichment-Project-2024-TPE
This repository contains the files necessary to run the TPE-based Bayesian Optimization approach on the adhesive bonding simulator. The code was developed within the Enrichment Project 2024, supported by the FAIR program.

# Constrained Tree-structured Parzen Estimator (cTPE)
We use cTPE to optimise selected parameters of the adhesive bonding simulator. More information on the optimization method itself can be found in the [source repository](https://github.com/nabenabe0928/constrained-tpe) or in the manuscript:

```
@article{watanabe2023ctpe,
  title={{c-TPE}: Tree-structured {P}arzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization},
  author={S. Watanabe and F. Hutter},
  journal={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```

# cTPE installation

The information below originates directly from the cTPE author:

> [!IMPORTANT]
> ...
> This repository is now not maintained anymore, so people might experience hard time in the installation.
> ...

Therefore, when installing cTPE, you may encounter potential difficulties such as version conflicts of the dependencies. 
However, this does not imply that the provided code is flawed. Troubleshooting based on web resources and downgrading selected packages should alleviate these issues.  

# Repository structure

The matlab scripts of the adhesive bonding model are stored in [adhesive_bonding_simulator](adhesive_bonding_simulator/) directory.

[ctpe_bonding_model_utils.py](./ctpe_bonding_model_utils.py) contains several utility functions to: create a configuration space (an object determining the optimisation space), call matlab codes, run the analysis $N$ times and save the results.   

Afterwards, the [ctpe_x_bonding_model.py](./ctpe_x_bonding_model.py) script runs the optimisation for each material separately, and writes result CSV files to the [results](/results) folder. 

[explore_outcome_space.py](./explore_outcome_space.py)  randomly draws points from the parameter space to explore the outcomes space. This is useful to learn more about the range of values generated by the simulator any to compare these outcomes with the outcomes found via cTPE optimisation.    

[graphs.R](./graphs.R)  includes three types of useful result visualisations.

