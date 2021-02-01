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

def generate_random_state(n_features, seed):
    """
    generate random state (of environment or a system),
    given seed, with n_features

    Parameters
    ----------
    n_features : int
        number of feature dimensions.
    seed : int
        seed for setting random states (for reproducibility).

    Returns
    -------
    init_mu : ndarray
        array of length n_features; means of the state.
    init_cov : ndarray
        array of size n_features x n_features; covariance of the state.
    """
    init_mu = np.random.rand(n_features).astype(float)
    init_cov = make_spd_matrix(n_features, random_state=seed)

    return init_mu, init_cov

def init_agent(n_features, n_init_images, init_mu, init_cov, alpha,
               init_im_dur, seed=np.random.randint(0, 1e4)):
    """
    Initialize the system state of an agent with a system state representing
    n_features based on n_init_images from a distribution definied by init_mu
    and init_cov, assuming a shift of the system state proportional to
    alpha per time step, where each of n_init_images is 'presented' to the
    agent for a duration of init_im_dur.

    Parameters
    ----------
    n_features : int
        number of feature dimensions.
    n_init_images : int
        number of random sample feauture vectors (aka 'images') to be drawn
        for initial exposre.
    init_mu : list
        list  of length n_features
        means for the distribution from which feature vectors for
        initial exposure will be drawn.
    init_cov : ndarray
        array of size n_features x n_features
        Covariance matrix for the distribution from which feature vectors for
        initial exposure will be drawn.
    alpha : float
        magnitude of shift in mu towards currently 'presented' feature vector
    init_im_dur : int
        number of arbitrary time steps each initialization feature vector is
        'presented' for
    seed : int
        seed for setting random states (for reproducibility).

    Returns
    -------
    mu : array_like
        list or 1d-array of length n_features;
        means of agent's system state after initial exposure
    cov : ndarray
       n_features x n_features array
       covariance of agent's system state after initial exposure.
    images : ndarray
        n_features x n_init_images array;
        all feature value combinations that were used to initialize the agent
    """
    # set seed for randomizations
    np.random.seed(seed)
    # set an initial, random system state
    mu_start = np.random.rand(n_features).astype(float)

    # generate a 'natural environment' of images
    images = multivariate_normal.rvs(init_mu, init_cov,
                                     n_init_images)

    # and get covariance matrix
    cov = EmpiricalCovariance().fit(images)
    cov = cov.covariance_

    mu = mu_start.copy()

    done = False
    t = 0
    trial = 0

    while not done:
        this_image = np.array(images[trial])

        # only continue with this stimulus if stim_dur is not reached
        if t < init_im_dur:
            # update mu
            mu += ((this_image - mu) * alpha)

            # next time step
            t += 1

        # next trial if time's up
        else:
            trial += 1
            t = 0

        # we are done once the agent switches while the last image is on
        if trial == len(images):
            done = True

    return mu, cov, images

def simulate_one_step_known_init(mu, cov, mu_init, cov_init, alpha,
                                   stim,
                                   w_sensory=1, w_learn=1, bias=0):
    """
    simulates one time step while sensory input is given by stim
    where the agent with system state defined by mu and cov.
    This function returns predicted r(t), A(t), and updated mu.
    mu_init and cov_init define true distribution based on which
    KL is computed as basis for delta-V.

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
    init_images : ndarray
        all feature value combinations that were used to initialize the agent.
    stim : array_like
        list or 1d-array of length n_features; feature vector of the stimulus.
    w_sensory : float, optional
        realtive weight of delta-V for calculating A(t). The default is 1.
    w_learn : float, optional
        relative weight of r(t) for calculating A(t). The default is 1.
    bias : float, optional
        constant to be added to predicted A(t). The default is 0.

    Returns
    -------
    mu_new : array_like
        list or 1d-array of length n_features;
        means of agent's system state after exposure
    r_t : float
        predicted r(t) value after exposure
    A_t : float
       predicted A(t) value after exposure
    """
    n_features = len(mu)
    mu = np.array(mu).astype(float)
    stim = np.array(stim)
    
    # get r
    r_t = np.exp(helpers.logLik_from_mahalanobis(stim, mu, cov))

    # get V(t)
    V_t = n_features - KL_distributions(mu_init, cov_init, mu, cov)

    # expected mu
    mu_new = mu + ((stim - mu) * alpha)

    # get estimate for V(t+1)
    V_t_exp = n_features - KL_distributions(mu_init, cov_init, mu_new, cov)

    # get delta-V
    delta_V = V_t_exp - V_t

    A_t = bias + (w_sensory * r_t + w_learn * delta_V)

    return mu_new, r_t, A_t



def simulate_practice_trials(mu, cov, alpha, stims, stim_dur):
    """
    simulate a run through n (number of rows in stims) trials given an agent's
    system state as defined by mu and cov and upsdated proportionally to alpha
    as it is exposed to stims for stim_dur time steps.
    Returns updated mu, in contrast to simulate_experiment(), which returns
    predicted r(t) and A(t) per trial.

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
    init_images : ndarray
        all feature value combinations that were used to initialize the agent.
    stims : ndarray
        feature value combinations for the stimuli to be used for practice
        trial simulations.
    stim_dur: list
        number of time steps per trial.

    Returns
    -------
    mu_new : array_like
        list or 1d-array of length n_features; means of agent's system state
        after practice trials.

    """
    done = False
    t = 0
    trial = 0
    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)
    
    while not done:
        this_image = np.array(stims[trial])

        # only continue with this stimulus if stim_dur is not reached
        if t < stim_dur[trial]:
            # update mu
            mu_new += ((this_image - mu_new) * alpha)

            # next time step
            t += 1

        # next trial if time is up
        else:
            trial += 1
            t = 0

        # we are done once the agent switches while the last image is on
        if trial == len(stims):
            done = True

    return mu_new


def simulate_experiment(mu, cov, alpha, init_images, stims, stim_dur,
                        w_sensory=1, w_learn=1, bias=0, return_mu=False):
    """
    simulates an 'experiment' consisting of n (number of rows in stims) trials
    where the agent with system state defined by mu and cov and based on
    previous exposure to init_images. The agent is exposed to each stimulus
    representation in stims for stim_dur, siulating one trial.
    This function returns predicted r(t) and A(t) at the end of each trial.
    Init_images is taken as basis for estimating true distribution against
    which KL is computed as basis for delta-V.
    Use simulate_experiment_known_init() if underlying means and variance of
    true distribution is (supposed to be) known.

    Parameters
    ----------
    u : array_like
        list or 1d-array of length n_features; means of agent's system state
        before practice trials
    cov : ndarrary
        array of size len(mu) x len(mu)
       covariance of agent's system state
    alpha : float
         magnitude of shift in mu towards currently 'presented' feature vector.
    init_images : ndarray
        all feature value combinations that were used to initialize the agent.
    stims : ndarray
        feature value combinations for the stimuli to be used for practice
        trial simulations.
    stim_dur: list
        number of time steps per trial.
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
    r_t_list : list
        a list of lenght len(stims) giving the predicted r(t) value at the end
        of each simulated trial.
    A_list : list
        a list of lenght len(stims) giving the predicted A(t) value at the end
        of each simulated trial..
    mu_new : array_like, optional
        list or 1d-array of length n_features;
        means of agent's system state after exposure

    """
    done = False
    t = 0
    trial = 0
    n_features = len(mu)
    true_prob = multivariate_normal.pdf(init_images, mu, cov)
    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # set up some empty lists for variables to be tracked
    r_t_list = []
    A_list = []

    while not done:
        this_image = np.array(stims[trial])

        # only continue with this stimulus if stim_dur is not reached
        if t < stim_dur[trial]:
            # update mu
            mu_new += ((this_image - mu_new) * alpha)

            # next time step
            t += 1

        # if time is up, compute output, next trial
        else:
            # get r
            r_t = np.exp(helpers.logLik_from_mahalanobis(this_image, mu_new, cov))

            # get V(t)
            V_t = n_features - KL(true_prob, multivariate_normal.pdf(init_images,
                                                                     mu_new, cov))

            # expected mu
            mu_exp = mu_new + ((this_image - mu_new) * alpha)

            # get estimate for V(t+1)
            V_t_exp = n_features - KL(true_prob,
                                      multivariate_normal.pdf(init_images,
                                                              mu_exp, cov))

            # get delta-V
            delta_V = V_t_exp - V_t

            A_list.append(bias + (w_sensory * r_t + w_learn * delta_V))
            r_t_list.append(r_t)

            trial += 1
            t = 0

        # we are done once the agent switches while the last image is on
        if trial == len(stims):
            done = True

    if return_mu:
        return r_t_list, A_list, mu_new
    else:
        return r_t_list, A_list

def simulate_experiment_known_init(mu, cov, mu_init, cov_init, alpha,
                                   stims, stim_dur,
                                   w_sensory=1, w_learn=1, bias=0,
                                   return_mu=False):
    """
    simulates an 'experiment' consisting of n (number of rows in stims) trials
    where the agent with system state defined by mu and cov.
    The agent is exposed to each stimulus representation in stims for stim_dur,
    siulating one trial.
    This function returns predicted r(t) and A(t) at the end of each trial.
    mu_init and cov_init define true distribution against
    which KL is computed as basis for delta-V.
    Use simulate_experiment() if true distribution is supposed to be estimated
    given a series of stimulus representations instead.

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
    stims : ndarray
        feature value combinations for the stimuli to be used for practice
        trial simulations.
    stim_dur: list
        number of time steps per trial.
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
    r_t_list : list
        a list of lenght len(stims) giving the predicted r(t) value at the end
        of each simulated trial.
    A_list : list
        a list of lenght len(stims) giving the predicted A(t) value at the end
        of each simulated trial..
    mu_new : array_like, optional
        list or 1d-array of length n_features;
        means of agent's system state after exposure
    """

    done = False
    t = 0
    trial = 0
    n_features = len(mu)
    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # set up some empty lists for variables to be tracked
    r_t_list = []
    A_list = []

    while not done:
        this_image = np.array(stims[trial])

        # only continue with this stimulus if stim_dur is not reached
        if t < stim_dur[trial]:
            # update mu
            mu_new += ((this_image - mu_new) * alpha)

            # next time step
            t += 1

        # compute output and next trial
        else:
            # get r
            r_t = np.exp(helpers.logLik_from_mahalanobis(this_image, mu_new, cov))

            # get V(t)
            V_t = n_features - KL_distributions(mu_init, cov_init, mu_new, cov)

            # expected mu
            mu_exp = mu_new + ((this_image - mu_new) * alpha)

            # get estimate for V(t+1)
            V_t_exp = n_features - KL_distributions(mu_init, cov_init, mu_exp, cov)

            # get delta-V
            delta_V = V_t_exp - V_t

            A_list.append(bias + (w_sensory * r_t + w_learn * delta_V))
            r_t_list.append(r_t)

            trial += 1
            t = 0

        # we are done once the agent switches while the last image is on
        if trial == len(stims):
            done = True

    if return_mu:
        return r_t_list, A_list, mu_new
    else:
        return r_t_list, A_list


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
