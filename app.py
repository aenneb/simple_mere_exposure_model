#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:53:09 2021

@author: aennebrielmann

initial round of trying to develop an interactive html page for exploring the 
model parameter space
"""

import os
import sys
import numpy as np
import pandas as pd

# import costum functions
home_dir = os.getcwd()
sys.path.append(home_dir)
from aestheticsModel import simSetup

# %% set up dash and plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='Simple mere exposure model (Brielmann & Dayan)'
# %% define the functions we need to update the data for the graph
def transform_alpha(alpha):
    return 10**(-alpha)

def update_mus(mu_0, var_X, mu_stim_1, mu_stim_2, alpha, stim_dur):
    mu_stim = np.array([mu_stim_1, mu_stim_2])
    cov = [[var_X, 0],[0, var_X]]
    new_mu = simSetup.simulate_practice_trials_compact(mu_0, cov, alpha, 
                                                       mu_stim, n_stims=1, 
                                                       stim_dur=stim_dur)
    return new_mu

def calc_predictions(mu_0, var_X, mu_true_1, mu_true_2, var_true,
           mu_stim_1, mu_stim_2, alpha, w_V):
    mu_true = np.array([mu_true_1, mu_true_2])
    mu_stim = np.array([mu_stim_1, mu_stim_2])
    cov = [[var_X, 0],[0, var_X]]
    cov_true = [[var_true, 0],[0, var_true]]
    A_t, r_t = simSetup.calc_A_known_init(mu_0, cov, mu_true, cov_true, alpha,
                                   mu_stim,
                                   w_sensory=1, w_learn=w_V, bias=0,
                                   return_mu=False, return_r_t=True)
    return A_t, r_t

def get_time_series(mu_0_1, mu_0_2, var_X,
                    mu_true_1, mu_true_2, var_true,
                    mu_stim_1, mu_stim_2,
                    alpha, w_V, stim_dur, n_reps):
    new_mu = np.array([mu_0_1, mu_0_2])
    A_t_list = []
    r_t_list = []
    delta_V_list = []
    for rep in range(n_reps):
        new_mu = update_mus(new_mu, var_X, mu_stim_1, mu_stim_2,
                            alpha, stim_dur)
        A_t, r_t = calc_predictions(new_mu, var_X, 
                                            mu_true_1, mu_true_2, var_true, 
                                            mu_stim_1, mu_stim_2, alpha, w_V)
        A_t_list.append(A_t)
        r_t_list.append(r_t)
        if w_V > 0:
            delta_V_list.append((A_t - r_t)/w_V)
        else:
            delta_V_list.append(A_t - r_t)
        
    return A_t_list, r_t_list, delta_V_list

# further, for creating contour plots
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def create_contour_data(mu, Sigma, N=100, xmin=-20, xmax=20, ymin=-20, ymax=20):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    return X[0,:], Y[:,0], Z

# %% define the overall html layout
app.layout = html.Div([
     # for every possible input, we need to define another dcc element whose id
    # and value are used by @app.callback
    html.Div([
        # for the system state
        html.Div([
            html.P('system state mean 1'),
            dcc.Slider(
                id='slider-mu_0_1',
                min=-10,
                max=10,
                value= 0, # np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('system state mean 2'),
            dcc.Slider(
                id='slider-mu_0_2',
                min=-10,
                max=10,
                value=0, #np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('system state variance'),
            dcc.Slider(
                id='slider-var_X',
                min=.1,
                max=10,
                value=1.04,
                # marks={str(mu): str(mu) for mu in np.arange(.01,10,.1)},
                step=0.1
            ),
        ], className = 'three columns'),
        
        # for expected true distribution
        html.Div([
            html.P('expected true mean 1'),
            dcc.Slider(
                id='slider-mu_true_1',
                min=-10,
                max=10,
                value=4.81, #np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('expected true mean 2'),
            dcc.Slider(
                id='slider-mu_true_2',
                min=-10,
                max=10,
                value=16.7, #np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('true distribution variance'),
            dcc.Slider(
                id='slider-var_true',
                min=.1,
                max=10,
                value=0.47,
                # marks={str(mu): str(mu) for mu in np.arange(.01,10,.1)},
                step=0.1
            ),
        ], className = 'three columns'),
            
        # for stimulus
        html.Div([
            html.P('stimulus mean 1'),
            dcc.Slider(
                id='slider-mu_stim_1',
                min=-10,
                max=10,
                value=-0.17, #np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('stimulus mean 2'),
            dcc.Slider(
                id='slider-mu_stim_2',
                min=-10,
                max=10,
                value=1.06, #np.round(np.random.rand(),1),
                # marks={str(mu): str(mu) for mu in np.arange(-1,1,.1)},
                step=0.1
            ),
            html.P('stimulus duration'),
            dcc.Slider(
                id='slider-stim_dur',
                min=1,
                max=100,
                value=33,
                step=1
            ),
        ], className = 'three columns'),
        
        # for rest of model
        html.Div([
            html.P('Learning rate'),
            dcc.Slider(
                id='slider-alpha',
                min=0,
                max=10,
                value=-np.log10(0.0005),
                step=.25
            ),
            html.P('weight of Delta-V'),
            dcc.Slider(
                id='slider-w_V',
                min=0,
                max=100,
                value=32,
                step=1
            ),
            html.P('max # repetitions'),
            dcc.Slider(
                id='slider-n_reps',
                min=1,
                max=100,
                value=50,
                step=1
            ),
        ], className = 'three columns'),
        
    ], className = 'row'),
    
    # all graphs
    html.Div([
        # dcc.Graph si placeholder for the figure we re-draw upon every cahnge using
        # the @app.callback
        html.Div([
            dcc.Graph(id='line_graph'),
            html.Br(),
        ], className='six columns'),
        html.Div([
            dcc.Graph(id='contour_graph'),
            html.Br(),
        ], className='six columns'),
    ], className='row'),
    
    # give an overview of all settings directly below the graph
    html.Div([
        html.Div([], className='one column'),
        html.Div([
            html.P(id='message'),
            html.Br(),
        ], className='ten columns'),
        html.Div([], className='one column'),
    ], className='row'),
])

# %% callback functions
@app.callback(
    Output('line_graph', 'figure'),
    Input('slider-mu_0_1', 'value'),
    Input('slider-mu_0_2', 'value'),
    Input('slider-var_X', 'value'),
    Input('slider-mu_true_1', 'value'),
    Input('slider-mu_true_2', 'value'),
    Input('slider-var_true', 'value'),
    Input('slider-mu_stim_1', 'value'),
    Input('slider-mu_stim_2', 'value'),
    Input('slider-alpha', 'value'),
    Input('slider-w_V', 'value'),
    Input('slider-stim_dur', 'value'),
    Input('slider-n_reps', 'value'))

def update_line_graphs(mu_0_1, mu_0_2, var_X, mu_true_1, mu_true_2, var_true,
                  mu_stim_1, mu_stim_2, alpha_input, w_V, stim_dur, n_reps):
    
    # fetch the predictions
    alpha = transform_alpha(alpha_input)
    A_t_list, r_t_list, delta_V_list = get_time_series(mu_0_1, mu_0_2, var_X,
                                                  mu_true_1, mu_true_2, var_true,
                                                  mu_stim_1, mu_stim_2,
                                                  alpha, w_V, stim_dur, n_reps)
    df = pd.DataFrame({"A(t)": A_t_list,
                       'r(t)': r_t_list,
                       'Delta-V(t)': delta_V_list,
                       "Repetition number": np.arange(1,n_reps+1)})
    
    fig = make_subplots(rows=1, cols=2,
                    shared_yaxes=False,
                    x_title='Repetition #', y_title='Predictions',
                    vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df['Repetition number'], y=df["A(t)"], 
                             mode='lines', name='A(t)'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Repetition number'], y=df["r(t)"],
                          mode='lines', name='r(t)'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=df['Repetition number'], y=df["Delta-V(t)"],
                          mode='lines', name='Delta-V(t)'),
                  row=1, col=2)
    fig.update_layout({"yaxis1": dict(autorange = True)})
    fig.update_layout({"yaxis2": dict(range = [0,1])})
    return fig

@app.callback(
    Output('contour_graph', 'figure'),
    Input('slider-mu_0_1', 'value'),
    Input('slider-mu_0_2', 'value'),
    Input('slider-var_X', 'value'),
    Input('slider-mu_true_1', 'value'),
    Input('slider-mu_true_2', 'value'),
    Input('slider-var_true', 'value'),
    Input('slider-mu_stim_1', 'value'),
    Input('slider-mu_stim_2', 'value'))

def update_contour_graphs(mu_0_1, mu_0_2, var_X, mu_true_1, mu_true_2, var_true,
                  mu_stim_1, mu_stim_2):
    mu_state = np.array([mu_0_1, mu_0_2])
    sigma_state = np.array([[var_X, 0], [0, var_X]])
    x, y, z_state = create_contour_data(mu_state, sigma_state)
    mu_true = np.array([mu_true_1, mu_true_2])
    sigma_true = np.array([[var_true, 0], [0, var_true]])
    _, y, z_true = create_contour_data(mu_true, sigma_true)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("System state", "Expected true distribution"),
                        shared_yaxes=True,
                        vertical_spacing=0.02)
    fig.add_trace(go.Contour(x=x, y=y, z=z_state, contours_coloring='lines',
                                               line_width=5, showscale = False,
                                               colorscale='Blues',), 1,1)
    fig.add_trace(go.Scatter(x=[mu_stim_1], y = [mu_stim_2], mode='markers',
                    name='Stimulus', marker=dict(size=10), fillcolor='orange'), 1,2)
    
    fig.add_trace(go.Contour(x=x, y=y, z=z_true, contours_coloring='lines',
                                               line_width=5, showscale = False,
                                               colorscale='Blues',), 1,2)
    fig.add_trace(go.Scatter(x=[mu_stim_1], y = [mu_stim_2], mode='markers',
                    name='Stimulus', marker=dict(size=10), fillcolor='orange'), 1,1)
    
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1,)
    
    return fig
    
@app.callback(
    Output('message', 'children'),
    Input('slider-mu_0_1', 'value'),
    Input('slider-mu_0_2', 'value'),
    Input('slider-mu_true_1', 'value'),
    Input('slider-mu_true_2', 'value'),
    Input('slider-mu_stim_1', 'value'),
    Input('slider-mu_stim_2', 'value'),
    Input('slider-var_X', 'value'),
    Input('slider-var_true', 'value'),
    Input('slider-alpha', 'value'),
    Input('slider-w_V', 'value'),
    Input('slider-stim_dur', 'value'))

def update_message(mu_0_1, mu_0_2, mu_true_1, mu_true_2, mu_stim_1, mu_stim_2,
                   var_X, var_true, alpha_input, w_V, stim_dur):
    alpha = transform_alpha(alpha_input)
    return (f'This simulation is based on a system state with means {mu_0_1}' + 
            f' and {mu_0_2}, and a variance of {var_X}.' +'\n' +
            f'The stimulus is set to {mu_stim_1} and {mu_stim_2}' +
            f' and presented for {stim_dur} time steps.' +'\n' +
            f'The true distribution has means {mu_true_1} and {mu_true_2}.' +
            f' and a variance of {var_true}.' +'\n' +
            'The learning rate is set to' + '{:.2e}'.format(alpha) + 
            ' and the weight of Delta-V' + f' to {w_V}.')

############ Execute the app
if __name__ == '__main__':
    app.run_server()
