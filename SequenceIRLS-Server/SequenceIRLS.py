import numpy as np
import math
import scipy as sp
from typing import List, Dict, Tuple

import numpy as np
import scipy as sp
from scipy.linalg import hankel
import math

import asyncio

def ceil(R):
    return R


def hankel(s):
    return sp.linalg.hankel(s[:len(s) // 2 + 1], s[len(s) // 2:])


def P_omega(Y):
    omega = []
    for i, x in enumerate(Y):
        if not math.isnan(x):
            omega.append(i)
    return (np.eye(len(Y))[omega])


def weight_tilda(U, S, V, E, X, type_mean='harmonic'):
    # U, left singular vectors, matrix of size d_1 x rk
    # S, Sigma Values
    # V, right singular vectors, matrix of size d_2 x rk,
    # E is the smoothing parameter
    # X is the matrix we want to change using the weight operator

    S_ = np.maximum(S, E)
    [d1, rk] = np.shape(U)
    d2 = np.shape(V)[0]

    # print(S)
    # print(S_)
    R = min(len(U), len(V))

    H = np.zeros([rk, rk])
    D = np.zeros([rk, rk])

    # U = U[:,:R]
    # V = V[:,:R]
    """
    print(U.shape)
    print(S.shape)
    print(V.shape)
    print(X.shape)
    print(H.shape)
    print(D.shape)
    #"""

    ##################################
    # Building H
    # We assume R to be the size of the sigmas due to the size of the sequence being relatively small, we also set r ~ 2/4
    # meaing R must be larger then 2/4 and with sequences this means we will often have weird things...
    for i in range(rk):
        for j in range(rk):
            if type_mean == 'harmonic':
                H[i, j] = 2 * ((S[i] ** 2 + S[j] ** 2 + 2 * (E ** 2)) ** -1)
            elif type_mean == 'geometric':
                H[i, j] = (S_[i] ** -1) * (S_[j] ** -1)

    ##################################
    # Building D
    for i in range(rk):
        if type_mean == 'harmonic':
            D[i, i] = 2 * ((S[i] ** 2 + 2 * (E ** 2)) ** -1)
        elif type_mean == 'geometric':
            D[i, i] = (S_[i] ** -1) * (E ** -1)  # 2 * (( S[i]**2 + 2*(E**2))**-1)

    # print(U.shape,H.shape,U.T.shape,X.shape,V.shape,V.T.shape)
    L = U @ (H * (U.T @ X @ V)) @ V.T

    # M is the ...
    M = U @ D.T @ U.T @ X @ (np.eye(d2) - V @ V.T)

    # N is the ...
    # print(R,U.shape,U.T.shape,X.shape,V.shape,D.T.shape,V.T.shape)
    N = (np.eye(d1) - U @ U.T) @ X @ V @ D.T @ V.T

    # O is the ...
    O = (E ** -2) * ((np.eye(d1) - U @ U.T) @ X @ (np.eye(d2) - V @ V.T))

    return (L + M + N + O)


def hm_irls(data_vector, rank_estimate, max_iter=100, tol=1e-8, type_mean='harmonic'):
    """Runs an iteratively reweighted least squares algorithms for the recovery of incomplete linear recurrence sequences from partial data.
    Leverages low-rank property of underlying Hankel matrix.

    Parameters
    ----------
    data_vector :  numpy.array / list
        Incomplete sequence of total length n. Zero values correspond to missing data at respective index (potential todo: change to float("nan"))
    rank_estimate : int
        Target rank of Hankel matrix. Corresponds to order of linear recurrence relation.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter that governs the stopping criterion. Stopping criterion: Relative change of l2-norm smaller than tol.
    Returns
    -------
    x : numpy.array
        Reconstructed sequence.
    stats : dictionary
        Contains algorithmic statistics.
    """
    x = data_vector
    P = P_omega(x)
    x = [0 if (isinstance(num, float) or isinstance(num, int)) and math.isnan(num) else num for num in x]

    # The unpadded 0 vector of data
    y = x @ P.T

    r = rank_estimate  # true_rank_parameter

    n = len(x)
    m = len(y)
    # R = rank_estimate  : I got rid of this parameter as it is not necessary to formulate the IRLS algorithm

    U, S, Vt = np.linalg.svd(hankel(x))

    smoothing = [0] * (max_iter)
    smoothing[0] = S[0]

    weights = [0] * (max_iter)
    weights[0] = smoothing[0] * np.identity(n)
    ranks = []
    stats = dict()

    for f in range(1, max_iter):
        x_o = x.copy()
        # print(f)

        W = weights[f - 1]

        if np.linalg.det(W) != 0:
            W_i = np.linalg.inv(W)

        # We can simplify the following line to make it more computationally efficent by solving directly for z
        # x = W_i @ P.T @ (np.linalg.inv(P@W_i@P.T) @ y )
        # P @ W_i @ P.T * z = Y    => linalg.solve(P@W_i@P.T,y)

        # recalculate the data vector using a minimisation method & constraint (4
        x = W_i @ P.T @ (np.linalg.solve(P @ W_i @ P.T, y))

        # SVD of the Padded hankel data vector (5)
        U, S, Vt = np.linalg.svd(hankel(x))

        smoothing[f] = min(smoothing[f - 1], S[r])
        r_c = np.sum(S > smoothing[f])
        ranks.append(r_c)

        # Calculate each element in the weight matrix with
        # W_ij = e_i.T W @ e_j
        # => vec(e_i.hankel) @ vec(weight_tilda(e_j.hankel))
        W = np.zeros([n, n])

        for i in range(0, n):
            e_i = [0] * n
            e_i[i] = 1
            for j in range(0, n):
                e_j = [0] * n
                e_j[j] = 1
                W[i, j] = hankel(e_i).flatten() @ weight_tilda(U[:, :r_c], S, Vt[:r_c, :].T, smoothing[f], hankel(e_j),
                                                               type_mean=type_mean).flatten()
        weights[f] = W
        #print(np.round(x, 8))
        # Check stopping criterion: relativ l2-error < tol
        rel_chg = np.linalg.norm(x_o - x) / np.linalg.norm(x_o)
        #print(rel_chg)
        if f > 1 and rel_chg < tol:
            break
    """    
    for w in weights:
        print("\n\n\n")
        print(np.round(w,0))
    #for s in smoothing:
        #print("\n\n\n")
        #print(s)
    """
    stats['ranks'] = ranks
    stats['smoothing'] = smoothing[0:f]
    stats['k'] = f
    print(S)
    return x, S, rel_chg

def complete_sequence(data_vector,rank_estimate,max_iter=100,tol=1e-8,type_mean='harmonic')->List[float]:
    """
    Returns only missing values in order.
    """
    data_vector = [float('nan') if x is None else x for x in data_vector]
    print(f"data_vector: {data_vector}")
    results, S, rel_chg = hm_irls(data_vector,rank_estimate,max_iter,tol,type_mean)
    data_vector = np.array(data_vector, dtype=float)

    # Now np.isnan works
     # Create mask for missing entries (NaN)
    missing_mask = np.isnan(data_vector)
    #print(type(results))
    #print(missing_mask)
    # Return only the results corresponding to missing entries
    #print(results[missing_mask].tolist())
    print(np.round(results[missing_mask],0).tolist())
    rounded = np.round(results[missing_mask],0)
    pred_x = rounded.tolist()
    print(pred_x)
    print(S.tolist())
    print(rel_chg)
    return pred_x, S.tolist(), rel_chg

async def hm_irls_iterative(data_vector,rank_estimate,max_iter=100,tol=1e-8,type_mean='harmonic',frequency=1)->List[float]:
    """Runs an iteratively reweighted least squares algorithms for the recovery of incomplete linear recurrence sequences from partial data.
    Leverages low-rank property of underlying Hankel matrix.

    Parameters
    ----------
    data_vector :  numpy.array / list
        Incomplete sequence of total length n. Zero values correspond to missing data at respective index (potential todo: change to float("nan"))
    rank_estimate : int
        Target rank of Hankel matrix. Corresponds to order of linear recurrence relation.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter that governs the stopping criterion. Stopping criterion: Relative change of l2-norm smaller than tol.
    Returns
    -------
    x : numpy.array
        Reconstructed sequence.
    stats : dictionary
        Contains algorithmic statistics.
    """

    x = [float('nan') if x is None else x for x in data_vector]
    x = np.array(x, dtype=float)

    missing_mask = np.isnan(x)

    P = P_omega(x)
    x = [0 if (isinstance(num, float) or isinstance(num, int)) and math.isnan(num) else num for num in x] 




    # The unpadded 0 vector of data
    y = x @ P.T

    r = rank_estimate  # true_rank_parameter

    n = len(x)
    m = len(y)
    # R = rank_estimate  : I got rid of this parameter as it is not necessary to formulate the IRLS algorithm

    U, S, Vt = np.linalg.svd(sp.linalg.hankel(x))

    smoothing = [0] * (max_iter)
    smoothing[0] = S[0]

    weights = [0] * (max_iter)
    weights[0] = smoothing[0] * np.identity(n)
    ranks = []
    stats = dict()

    idx=0
    for f in range(1, max_iter):
        x_o = x.copy()
        # print(f)

        W = weights[f - 1]

        if np.linalg.det(W) != 0:
            W_i = np.linalg.inv(W)

        # We can simplify the following line to make it more computationally efficent by solving directly for z
        # x = W_i @ P.T @ (np.linalg.inv(P@W_i@P.T) @ y )
        # P @ W_i @ P.T * z = Y    => linalg.solve(P@W_i@P.T,y)

        # recalculate the data vector using a minimisation method & constraint (4
        x = W_i @ P.T @ (np.linalg.solve(P @ W_i @ P.T, y))

        # SVD of the Padded hankel data vector (5)
        U, S, Vt = np.linalg.svd(hankel(x))

        smoothing[f] = min(smoothing[f - 1], S[r])
        r_c = np.sum(S > smoothing[f])
        ranks.append(r_c)

        # Calculate each element in the weight matrix with
        # W_ij = e_i.T W @ e_j
        # => vec(e_i.hankel) @ vec(weight_tilda(e_j.hankel))
        W = np.zeros([n, n])

        for i in range(0, n):
            e_i = [0] * n
            e_i[i] = 1
            for j in range(0, n):
                e_j = [0] * n
                e_j[j] = 1
                W[i, j] = hankel(e_i).flatten() @ weight_tilda(U[:, :r_c], S, Vt[:r_c, :].T, smoothing[f], hankel(e_j),
                                                               type_mean=type_mean).flatten()
        weights[f] = W

        # Check stopping criterion: relative l2-error < tol
        norm_xo = np.linalg.norm(x_o)
        rel_chg = np.linalg.norm(x_o - x) / norm_xo if norm_xo > 1e-12 else 0.0

        # Always yield this iteration's data (before possibly breaking)
        yield S, x[missing_mask].tolist(), rel_chg, W.tolist()
        await asyncio.sleep(0)  # Allow other tasks to run
        idx += 1

        # Convergence check — only after at least 2 iterations
        if f > 1 and rel_chg < tol:
            break

    stats['ranks'] = ranks
    stats['smoothing'] = smoothing[0:f]
    stats['k'] = f