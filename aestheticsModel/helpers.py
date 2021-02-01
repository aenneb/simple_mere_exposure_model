#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:52:18 2020

@author: aennebrielmann

collection of helper functions
"""
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest # for esnuring cov

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def normalize_each(a):
    """
    Calculates the mahalanobis distance between stim and mu_x

    Parameters
    ----------
    a : nd-array
        array with n rows to be normalized

    Returns
    -------
    a_norm : nd-array
        modified version of the input array a where each row is normalized

    """
    a_norm = np.zeros(a.shape)
    for ii in range(a.shape[0]):
        norm = np.linalg.norm(a[ii])
        if norm == 0:
            a_norm[ii] = a[ii]
        else:
            a_norm[ii] = a[ii]/norm
    return a_norm

def triu_to_cov(cov_uptri, n_features):
    """Convert a flat vector containing the upper triangle (including diagonal)
    back into a symmetric matrix, here the covariance matrix
    """
    cov = np.zeros((n_features, n_features))
    cov[np.triu_indices(cov.shape[0], k = 0)] = cov_uptri
    cov = cov + cov.T - np.diag(np.diag(cov))

    return cov

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

# another function that loops through all images
def loop_through_images(features, mu_x, cov):
    """calculate the logLikelihood of all images with the function
    logLik_from_mahalanobis() and return all logLiks in one vector
    """
    n_images = features.shape[0]
    # n_features = features.shape[1]
    logLik = []

    for im in range(n_images):
        stim = features[im]
        tmp = logLik_from_mahalanobis(stim, mu_x, cov)
        logLik.append(tmp)

    return logLik

def cov_constraint(x):
    """constrain covariance matrix to be positive definite"""

    try:
        sum_ev = np.sum(np.linalg.eigvals(triu_to_cov(x[14:], 14)))
        check = sum_ev - 14
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            check = 99
        else:
            raise

    return check

# additional constraint for mu_x
def normality_constraint(x):
    """constrain mu_x to be a normalized vector"""
    x = x[:14]
    check = np.sum(x) - 1

    return check

def cost_fn_fit_to_ratings(parameters, features, ratings):
    """
    re-shapes the flat parameters vector into its interpretable components and
    passes them on to the likelihood function. Calculates the overall cost as
    the negative sum of all log-likelihoods and scales them appropriately for
    opitmization.

    Parameters
    ----------
    parameters : array of float
        flattened version of all parameters to be estimated.
    features : m x n array of float
        contains the m feature values of n images for n features.
    ratings  : array of float
        contains the ratings to which the likelihoods shall relate to

    Returns
    -------
    cost : float
        overall cost to be minimized by optimizer.
    """

    n_features = features.shape[1]
    mu_x = parameters[:n_features]
    cov_uptri = parameters[n_features:]
    cov = triu_to_cov(cov_uptri, n_features)
    # giving up on getting constraints to handle covariance matrix being one
    cov = cov_nearest(cov)

    logLik_list = loop_through_images(features, mu_x, cov)
    corr = np.corrcoef(ratings, logLik_list)[0][1]
    cost = - corr

    # print(cost) # to see whether the optimizer is actually making progress
    return cost
