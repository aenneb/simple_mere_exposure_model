#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:52:18 2020

@author: aennebrielmann

collection of helper functions
"""
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest # for esnuring cov

def mahalanobis(stim, mu_x, cov):
    """
    Calculates the mahalanobis distance between stim and mu_x

    Parameters
    ----------
    stim : (N,) array_like
        feature vector of the stimulus.
    mu_x : (N,) array_like
        vector describing the system state (same shape as stim).
    cov : ndarray
        covariance matrix.

    Returns
    -------
    mahalanobis : double
        the mahalanobis distance between stim and mu_x given cov

    """
    stim = np.array(stim)
    mu_x = np.array(mu_x)
    inv_cov = np.linalg.inv(cov)
    s_minus_mu = stim - mu_x
    mahalanobis = np.sqrt(np.dot(np.dot(s_minus_mu.T, inv_cov), s_minus_mu))

    return mahalanobis

def logLik_from_mahalanobis(stim, mu_x, cov, k=None):
    """calculate the log likelihood of the current image given the presumed
    'system state' mu_x and covariance matrix cov based on the mahalanobis
    distance between image feature vector and vector representing the system
    state
    """
    if k is None:
        k = 0

    stim = np.array(stim)
    mu_x = np.array(mu_x)
    inv_covmat = np.linalg.inv(cov)
    s_minus_mu = stim - mu_x
    log_p = k - np.dot(np.dot(s_minus_mu.T, inv_covmat), s_minus_mu) / 2

    return log_p
