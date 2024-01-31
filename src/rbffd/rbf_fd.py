import jax.numpy as jnp
import jax
from jax import grad, jacrev
from functools import partial

import numpy as np
import jax.scipy.linalg as jla

from scipy.spatial import KDTree

from rbffd.polyterms import polyterms_jax




# Define the radial basis function

# Example operators
def laplacian(f, argnums=0):
    """ Laplacian operator """

    hessian = jacrev(grad(f, argnums=argnums))

    return lambda *args : jnp.trace(jnp.squeeze(hessian(*args)))

def gradient(f, argnums=0):
    """ Gradient operator """
    return grad(f, argnums=argnums)

@jax.jit
def gaussian_rbf(x, c, epsilon=2.0):
    """ Gaussian Radial Basis Function """
    return jnp.exp(-(epsilon * jnp.sqrt(jnp.sum((x - c)**2, axis=-1)))**2)

@jax.jit
def phs_rbf(x, c, m=3):
    """ Polynomial Radial Basis Function """
    return jnp.sqrt(jnp.sum((x - c)**2, axis=-1))**m


@partial(jax.jit, static_argnames=['rbf'])
def rbf_matrix(X, rbf=phs_rbf):
    """ Sets up the system matrix for the RBF-FD method """

    X = jnp.expand_dims(X, axis=0)

    A = rbf(X, jnp.swapaxes(X, 0 , 1))

    return A

@partial(jax.jit, static_argnames=['operator', 'pdeg', 'rbf'])
def make_stencil(X, y, operator, pdeg=1, rbf=phs_rbf):
    """ Finds the operator stencil using RBF-FD """

    N = len(X)

    # create the matrix defining basis over the nodes
    A = rbf_matrix(X, rbf=rbf)

    # create a matrix for polynomial exactness
    P = polyterms_jax(X, pdeg)

    #saddle point filler
    null = jnp.zeros((P.shape[1], P.shape[1]))  

    #saddle point system
    C = jnp.block([[A, P], [P.T, null]])


    # rhs that defines the action of the operator over the nodes
    rhs_A = jnp.zeros(N)

    operator_phi = operator(rbf)

    rhs_A = jnp.array([ operator_phi(y+1e-6, X[i]) for i in range(N) ]) # 1e-6 is a hack to avoid 0/0

    # rhs that defines the action of the operator over the polynomials
    my_polyterms = lambda i : (lambda y : polyterms_jax(y, pdeg)[0,i])

    rhs_P = jnp.array([ operator(my_polyterms(i))(jnp.expand_dims(y,axis=0)+1e-6) for i in range(P.shape[1]) ] ) # 1e-6 is a hack to avoid 0/0

    #block the rhs
    rhs_C = jnp.block([rhs_A, rhs_P])

    #solve
    w = jla.solve(C, rhs_C)[:N]

    return w

def build_operator(X, operator, rbf=phs_rbf, stencil_size=9, pdeg=1):
    """ Builds the operator for the RBF-FD method """

    # build the tree
    kdtree = KDTree(X)

    stencil_dict = {i: kdtree.query(X[i], k=stencil_size)[1] for i in range(len(X))}

    #relative stencil size (sten. size/num. poly. terms) 2.5 is the golden number

    N = len(X)

    L = np.zeros((N, N))

    # find the stencil for each node
    for i in range(N):
        L[i, stencil_dict[i]] = make_stencil(X[stencil_dict[i]], X[i], operator = laplacian, pdeg=pdeg, rbf=rbf)

    return L





if __name__ == "__main__":

    N = 10

    # Set up the nodes and epsilon
    X_grid = np.linspace(0, 1, N)
    Y_grid = np.linspace(0, 1, N)

    X, Y = np.meshgrid(X_grid, Y_grid)

    D = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

    test_rbf = lambda x, c : phs_rbf(x, c, m=5.0)

    try: 
        G = build_operator(D, operator=gradient, rbf=test_rbf, stencil_size=9, pdeg=1)
    except:
        print("Gradient operator failed to build")

    if G is not None:
        print("Gradient operator built successfully")

    try: 
        L = build_operator(D, operator=laplacian, rbf=test_rbf, stencil_size=9, pdeg=1)
    except:
        print("Laplacian operator failed to build")

    if L is not None:
        print("Laplacian operator built successfully")