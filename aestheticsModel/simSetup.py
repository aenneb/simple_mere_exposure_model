# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:56:05 2020
Last update on Fri Dec 18 2020

@author: abrielmann

collection of functions that will help us to set up simulations of our
model of aesthetic value that replicate existing empirical findings
"""
import os
import sys
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix

# import costum functions
os.chdir('..')
home_dir = os.getcwd()
sys.path.append(home_dir)
from aestheticsModel import helpers


def KL(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def KL_distributions(mu1, cov1, mu2, cov2):
    """
    derivation: http://stanford.edu/~jduchi/projects/general_notes.pdf

    Parameters
    ----------
    mu1 : array_like
        means of the 'true' feature dsitribution.
    cov1 : ndarray
        covariance matrix of the 'true' feature distribution
    mu2 : array_like
        means of the system state.
    cov2 : ndarray
        covariance matrix of the system state

    Returns
    -------
    KL.

    """
    n = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    mu_diff =np.array(mu2).astype(float) - np.array(mu1).astype(float)
    KL = (0.5 *
          np.log((np.linalg.det(cov2) / np.linalg.det(cov1)))
          - n
          + np.trace(np.dot(cov2_inv, cov1))
          + np.dot(np.dot(mu_diff.T, cov2_inv), mu_diff))

    return KL

def simulate_practice_trials_compact(mu, cov, alpha, stim_mu, n_stims, stim_dur):
    """
    update agent's system state after exposure to n_stims stimuli from the
    stimulus distribution with means stim_mu for stim_dur each
    given an agent's system state as defined by mu and cov
    Returns updated mu
    This function abbreviates the simulation process by updating mu only once
    summing up estimated changes

    Parameters
    ----------
    mu : array_like
        list or 1d-array of length n_features; means of agent's system state
        before practice trials
    cov : ndarrary
        array of size len(mu) x len(mu)
       covariance of agent's system state
    alpha : float
         magnitude of shift in mu towards currently 'presented' feature vector.
    stim_mu : ndarray
        means of the stimulus distribution.
    n_stims : int
        number of stimuli the agent is exposed to.
    stim_dur: int
        number of time steps per stimulus exposure.

    Returns
    -------
    mu_new : array, float
        1d-array of length n_features; means of agent's system state
        after practice trials.

    """

    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # update mu
    s_minus_mu = stim_mu - mu_new
    sum_change = s_minus_mu * (1-(1-alpha) ** (stim_dur * n_stims))
    mu_new += sum_change


    return mu_new


def calc_A_known_init(mu, cov, mu_init, cov_init, alpha,
                                   stim,
                                   w_sensory=1, w_learn=1, bias=0,
                                   return_mu=False, return_r_t=False):
    """
    Calculates A for a given system state and stimulus given known p_true
    as specified by mu_init and cov_init

    Parameters
    ----------
    mu : array_like
        list or 1d-array of length n_features; means of agent's system state
        before practice trials
    cov : ndarrary
        array of size len(mu) x len(mu)
       covariance of agent's system state
    mu_init : array_like
        list or 1d-array of length n_features; means of p_true
        before practice trials
    cov_init : ndarrary
        array of size len(mu) x len(mu)
        covariance of p_true
    alpha : float
         magnitude of shift in mu towards currently 'presented' feature vector.
    stim : ndarray
        feature value combinations for the stimulus
    w_sensory : float, optional
        realtive weight of delta-V for calculating A(t). The default is 1.
    w_learn : float, optional
        relative weight of r(t) for calculating A(t). The default is 1.
    bias : float, optional
        constant to be added to predicted A(t). The default is 0.
    return_mu : bool, optional
        whether to return updated value of mu. The default is False.

    Returns
    -------
    A_t : float
        predicted A(t) for stim
    mu_new : array, float, optional
        list or 1d-array of length n_features;
        means of agent's system state after exposure
    r_t : float, optional
        predicted r(t) for stim
    """

    n_features = len(mu)
    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # update mu because at this moment, agent is exposed to stim
    mu_new += ((stim - mu_new) * alpha)

    # get r
    r_t = np.exp(helpers.logLik_from_mahalanobis(stim, mu_new, cov))

    # get V(t)
    V_t = n_features - KL_distributions(mu_init, cov_init, mu_new, cov)

    # expected mu
    mu_exp = mu_new + ((stim - mu_new) * alpha)

    # get estimate for V(t+1)
    V_t_exp = n_features - KL_distributions(mu_init, cov_init, mu_exp, cov)

    # get delta-V
    delta_V = V_t_exp - V_t

    A_t = bias + (w_sensory * r_t + w_learn * delta_V)

    if return_mu:
        if return_r_t:
            return A_t, mu_new, r_t
        else:
            return A_t, mu_new
    elif return_r_t:
        return A_t, r_t
    else:
        return A_t
