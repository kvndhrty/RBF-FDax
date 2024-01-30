
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial



def polyterms(pdeg, *args):
# create the polynomials of degree pdeg for the args variables, args may be len 1, 2, or 3 (x, y, z)

    if len(args) == 1:
        p = np.zeros((pdeg+1, len(args[0])))
        for i in range(pdeg+1):
            p[i] = args[0]**i

    elif len(args) == 2:
        p = np.zeros((int((pdeg+1)*(pdeg+2)/2), len(args[0])))
        idx = 0 
        for i in range(pdeg+1):
            for j in range(pdeg+1-i):
                p[idx] = args[0]**i * args[1]**j
                idx += 1

    elif len(args) == 3:
        p = np.zeros((int((pdeg+1)*(pdeg+2)*(pdeg+3)/6), len(args[0])))
        idx = 0 
        for i in range(pdeg+1):
            for j in range(pdeg+1-i):
                for k in range(pdeg+1-i-j):
                    p[idx] = args[0]**i * args[1]**j * args[2]**k
                    idx += 1

    return p.T

@partial(jax.jit, static_argnames=['n', 'k'])
def jax_binom(n : int ,k : int):
    #recursive function to compute the binomial coefficient
    if k > n:
        raise ValueError("k must be less than or equal to n")
    elif k == 0:
        return 1
    elif k > n/2:
        return jax_binom(n,n-k)
    return jnp.int32(n * jax_binom(n-1,k-1) / k)


#@partial(jax.jit, static_argnames=['points', 'mult_term'])
def polyterms_jax(pdeg, points, mult_term=1):
    # a recursive jax function that builds the polynomials of degree pdeg evaluated at points (which may be of any size)

    npoints, ndims = points.shape
    index = 0
    poly = jnp.zeros((npoints, jax_binom(pdeg+ndims, pdeg)))

    max_range = pdeg + 1

    for i in range(max_range):
        if ndims > 1:
            terms = jax_binom(pdeg+ndims-1-i, pdeg-i)
            poly = poly.at[:,index:index+terms].set(polyterms_jax(pdeg-i, points[:,0:-1], mult_term * points[:,-1]**i))
        else:
            terms = 1
            poly = poly.at[:,index].set(points[:,-1]**i * mult_term)

        index += terms

    return poly




if __name__ == "__main__":

    # test that the outputs of polyterms and polyterms_jax are the same

    pdeg = 2
    points = np.random.rand(5,3)
    poly = polyterms(pdeg, points[:,0], points[:,1], points[:,2])

    poly_jax = polyterms_jax(pdeg, points)

    if np.allclose(poly, poly_jax):
        print("polyterms and polyterms_jax are the same")