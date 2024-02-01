# RBF-FDax: Radial Basis Function - Finite Differences in Jax

## Description
This repository contains the implementation of Radial Basis Function Finite Differences (RBF-FD), written in Jax. RBF-FD is a numerical method used for creating linear operators. It is particularly well-suited for problems where flexibility in node placement and high accuracy are required.

## Installation
To install the package, clone this repository to your local machine and run the following command from the base directory of the repo:

```bash
pip install -e .
```


## Usage 

Here is an example of generating the Dx and Dy operators using rbffdax, and applying it to the sinc function in 2D. 

```python
import numpy as np
import matplotlib.pyplot as plt
from rbffd import *

N = 25
rng = np.random.default_rng(0)
var = 0.00001

# Set up the nodes and epsilon
X_grid = np.linspace(-1, 1, N)
Y_grid = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X_grid, Y_grid)
X += rng.normal(0, var, (N, N))
Y += rng.normal(0, var, (N, N))
D = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
F = np.sinc(np.sqrt(np.sum(D**2, axis=-1)))

plt.scatter(D[:, 0], D[:, 1])
plt.show()

rbf = lambda x, y: phs_rbf(x, y, 5)
Dx = build_operator(D, operator=dx, rbf=rbf, stencil_size=9, pdeg=1)
Dy = build_operator(D, operator=dy, rbf=rbf, stencil_size=9, pdeg=1)

fig, axs = plt.subplots(3, 1, figsize=(3, 9))
axs[0].pcolormesh(X, Y, F.reshape(N, N), shading='auto')
axs[1].pcolormesh(X, Y, (Dx@F).reshape(N, N), shading='auto')
axs[2].pcolormesh(X, Y, (Dy@F).reshape(N, N), shading='auto')

```

This code should output something like these two plots:

![Scatter plot of points](https://github.com/kvndhrty/RBF-FDax/assets/35179093/9a6c8b41-72fe-443c-868e-65a2518f7636)

![Image of function and its derivatives](https://github.com/kvndhrty/RBF-FDax/assets/35179093/8efc3f64-c1f0-4827-a3ee-92a01edae071)

## Future Updates
I made this package over the course of the last couple weeks, and its development will continue for at least the near term.
