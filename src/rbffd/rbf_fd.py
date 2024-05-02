import jax.numpy as jnp
import jax
from jax import grad, jacrev
from functools import partial

import numpy as np
import jax.scipy.linalg as jla

from scipy.spatial import KDTree

from rbffd.polyterms import polyterms_jax

#special functions to help with multi-dimensional arrays vs scalars
def c_trace(A):

    if A.ndim == 2:
        return jnp.trace(A)
    elif A.ndim == 0:
        return A
    else:
        raise ValueError("Input must be a 2D or 0D array")
    
def c_squeeze(A):

    if A.ndim == 2:
        return A.reshape(-1,)
    elif A.ndim == 1:
        return A
    elif A.ndim == 0:
        return A.reshape(1)
    else:
        raise ValueError("Input must be a 2D or 1D or 0D array")


# Example operators
def laplacian(f, argnums=0):
    """ Laplacian operator """

    hessian = jacrev(grad(f, argnums=argnums))

    return lambda *args : c_trace(jnp.squeeze(hessian(*args)))

def dx(f, argnums=0):
    """ Derivative with respect to x """

    full_grad = grad(f, argnums=argnums)

    return lambda *args : c_squeeze(full_grad(*args))[0]

def dy(f, argnums=0):
    """ Derivative with respect to y """

    full_grad = grad(f, argnums=argnums)

    return lambda *args : jnp.squeeze(full_grad(*args))[1]

def dz(f, argnums=0):
    """ Derivative with respect to z """

    full_grad = grad(f, argnums=argnums)

    return lambda *args : jnp.squeeze(full_grad(*args))[2]

def divergence(f, argnums=0):
    """ Divergence operator """

    full_grad = grad(f, argnums=argnums)

    return lambda *args : jnp.sum(full_grad(*args))


# Define the radial basis functions
@jax.jit
def gaussian_rbf(x, c, epsilon=2.0):
    """ Gaussian Radial Basis Function """
    z = jnp.sum((x - c)**2, axis=-1)

    r = jnp.sqrt(jnp.where(z < 1e-10, 0.0, z))
    return jnp.exp(-(epsilon * r)**2)

@jax.jit
def circular_gaussian_rbf(x, c, epsilon=2.0):
    """ Gaussian Radial Basis Function """
    z = jnp.sum((jnp.exp(2*jnp.pi*1j*x) - jnp.exp(2*jnp.pi*1j*c))**2, axis=-1)

    r = jnp.real(jnp.sqrt(jnp.where(z < 1e-10, 0.0, z)))
    return jnp.exp(-(epsilon * r)**2)

@jax.jit
def phs_rbf(x, c, m=3.0):
    """ Polynomial Radial Basis Function """

    z = jnp.sum((x - c)**2, axis=-1)

    r = jnp.sqrt(jnp.where(z < 1e-10, 0.0, z))

    return r**m


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
    #rhs_A = jnp.zeros(N)

    operator_phi = operator(rbf)

    rhs_A = jnp.array([ operator_phi(y+1e-6, X[i]) for i in range(N) ]) # 1e-6 is a hack to avoid 0/0

    # rhs that defines the action of the operator over the polynomials
    my_polyterms = lambda i : (lambda y : jnp.squeeze(polyterms_jax(y, pdeg))[i])

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
        L[i, stencil_dict[i]] = make_stencil(X[stencil_dict[i]], X[i], operator = operator, pdeg=pdeg, rbf=rbf)

    return L



if __name__ == "__main__":

    N = 10

    # Set up the nodes and epsilon
    X_grid = np.linspace(0, 1, N)
    Y_grid = np.linspace(0, 1, N)

    X, Y = np.meshgrid(X_grid, Y_grid)

    D = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

    test_rbf = lambda x, c : phs_rbf(x, c, m=5.0)


    try: DX = build_operator(D, operator=dx, rbf=test_rbf, stencil_size=9, pdeg=1)
    except: DX = None

    if DX is not None:
        print("dx operator built successfully")

    try: DY = build_operator(D, operator=dy, rbf=test_rbf, stencil_size=9, pdeg=1)
    except: DY = None

    if DY is not None:
        print("dy operator built successfully")

    try: DZ = build_operator(D, operator=dz, rbf=test_rbf, stencil_size=9, pdeg=1)
    except: DZ = None

    if DZ is not None:
        print("dz operator built successfully")

    try: divergence = build_operator(D, operator=divergence, rbf=test_rbf, stencil_size=9, pdeg=1)
    except: DX = None

    if divergence is not None:
        print("divergence operator built successfully")

    try: L = build_operator(D, operator=laplacian, rbf=test_rbf, stencil_size=9, pdeg=1)
    except: L = None

    if L is not None:
        print("Laplacian operator built successfully")