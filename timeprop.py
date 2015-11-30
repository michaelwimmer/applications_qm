"""Simple time propagation for the Schroedinger equation"""
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.sparse as sp


def propagate(ham, psi, delta_t, tridiag=False):
    """Propagage the wave function `psi` by time `delta_t` using the hamiltonian
    `ham`. Uses the Crank-Nicolson method.
    """

    if sp.isspmatrix_dia(ham):
        # Optimization for banded matrices
        denom = ham.copy()
        denom.data *= 0.5j * delta_t
        denom.data[denom.offsets==0, :] += 1.0
        numer = sp.eye(ham.shape[0], format='csr') - 0.5j * ham.tocsr() * delta_t
        denom, l, u = _make_banded_matrix(denom.todia())
        temp = la.solve_banded((l, u), denom, psi)
        return numer.dot(temp)
    if sp.isspmatrix(ham):
        denom = sp.eye(ham.shape[0]) + 0.5j * ham * delta_t
        numer = sp.eye(ham.shape[0]) - 0.5j * ham * delta_t
        temp = sla.spsolve(denom, psi)
        return numer.dot(temp)
    else:
        denom = np.eye(ham.shape[0]) + 0.5j * ham * delta_t
        numer = np.eye(ham.shape[0]) - 0.5j * ham * delta_t
        temp = la.solve(denom, psi)
        return np.dot(numer, temp)


def _make_banded_matrix(ham):
    """Turn a sparse dia_matrix into a banded matrix
       format understood by scipy.linalg.solve_banded.
    """

    N = ham.shape[0]
    u = ham.offsets.max()
    l = ham.offsets.min()
    banded = np.zeros(shape=(1+u-l, N), dtype=complex)
    for k, offset in enumerate(ham.offsets):
        banded[u - offset, :] = ham.data[k, :]
        
    return banded, -l, u
