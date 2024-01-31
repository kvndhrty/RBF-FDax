import jax.numpy as jnp
import jax
from jax import grad, jacrev
from functools import partial

import numpy as np
import jax.scipy.linalg as jla

from scipy.spatial import KDTree

from rbffd.polyterms import polyterms_jax




# Define the radial basis function

@jax.jit
def gaussian_rbf(x, c, epsilon):
    """ Gaussian Radial Basis Function """
    return jnp.exp(-(epsilon * jnp.sqrt(jnp.sum((x - c)**2, axis=-1)))**2)

@jax.jit
def phs_rbf(x, c, m):
    """ Polynomial Radial Basis Function """
    return jnp.sqrt(jnp.sum((x - c)**2, axis=-1))**m


@partial(jax.jit, static_argnames=['rbf', 'rbf_arg'])
def setup_rbf_matrix(X, rbf_arg, rbf=gaussian_rbf):
    """ Sets up the system matrix for the RBF-FD method """

    X = jnp.expand_dims(X, axis=0)

    A = rbf(X, jnp.swapaxes(X, 0 , 1), rbf_arg)

    return A

@partial(jax.jit, static_argnames=['rbf', 'rbf_arg', 'pdeg'])
def laplacian_stencil(X, y, rbf_arg, rbf=gaussian_rbf, pdeg=2):
    """ Finds the Laplacian operator stencil using RBF-FD """

    N = len(X)

    # create the matrix defining basis over the nodes
    A = setup_rbf_matrix(X, rbf_arg, rbf=rbf)

    # create a matrix for polynomial exactness
    P = polyterms_jax(X, pdeg)

    #saddle point filler
    null = jnp.zeros((P.shape[1], P.shape[1]))  

    #saddle point system
    C = jnp.block([[A, P], [P.T, null]])


    # rhs that defines the action of the operator over the nodes
    rhs_A = jnp.zeros(N)

    laplace_phi = jacrev(grad(rbf, argnums=0))

    for i in range(N):
        rhs_A = rhs_A.at[i].set(jnp.trace(laplace_phi(X[i], y+1e-4 , rbf_arg))) # 1e-9 is a hack to avoid 0/0


    # rhs that defines the action of the operator over the polynomials
    my_polyterms = lambda points : polyterms_jax(points, pdeg)

    laplace_poly = jacrev(jacrev(my_polyterms, argnums=0))

    hessian = jnp.squeeze(laplace_poly(jnp.expand_dims(y,axis=0)+1e-4))

    if len(hessian.shape) == 1:
        rhs_P = hessian
    else:
        offset = len(hessian.shape) - 2
        rhs_P = jnp.trace(hessian, axis1=0+offset, axis2=1+offset)

    #block the rhs
    rhs_C = jnp.block([rhs_A, rhs_P])

    #solve
    w = jla.solve(C, rhs_C)[:N]

    return w

def vmap_laplacian_system(X, rbf_arg, rbf=gaussian_rbf, tree=None, stencil_size=5):
    # THIS DOESN'T WORK WITH THE TREE INSIDE THE JITTED FUNCTION
    """ This solves for the laplace weights for the rbf nodes / laplacian operator"""

    my_laplace_operator = partial(laplacian_stencil, rbf_arg=rbf_arg, rbf=rbf, tree=tree, stencil_size=stencil_size)

    L = jax.vmap(my_laplace_operator, in_axes=(None, 0), out_axes=0)(X, X)

    return L


def laplacian_operator(X, stencil_dict, rbf_arg, rbf=gaussian_rbf, pdeg=1):
    """ This solves for the laplace weights for the rbf nodes / laplacian operator"""

    #relative stencil size (sten. size/num. poly. terms) 2.5 is the golden number

    N = len(X)

    L = np.zeros((N, N))

    # find the stencil for each node
    for i in range(N):
        L[i, stencil_dict[i]] = laplacian_stencil(X[stencil_dict[i]], X[i], rbf_arg=rbf_arg, rbf=rbf, pdeg=pdeg)

    return L


def build_operator(X, rbf_arg=3, rbf=phs_rbf, stencil_size=9, pdeg=1):
    """ Builds the operator for the RBF-FD method """

    # build the tree
    kdtree = KDTree(X)

    stencil_dict = {i: kdtree.query(X[i], k=stencil_size)[1] for i in range(len(X))}


    # find the laplacian operator
    L = laplacian_operator(X, stencil_dict, rbf_arg, rbf, pdeg)

    return L





if __name__ == "__main__":

    N = 10

    # Set up the nodes and epsilon
    X_grid = np.linspace(0, 1, N)
    Y_grid = np.linspace(0, 1, N)

    X, Y = np.meshgrid(X_grid, Y_grid)

    D = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

    L = build_operator(D, rbf_arg=3, rbf=phs_rbf, stencil_size=9, pdeg=1)

    if L is not None:
        print("Operator built successfully")