import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce 
from itertools import combinations

def get_slice(i, n_lags):
    return slice(n_lags - (i+1), -(i+1))

def make_lag_matrix(X, n_lags):
    lag_idxs = reversed(range(n_lags))
    return np.concatenate([X[get_slice(i, n_lags)] for i in lag_idxs], axis=-1)

def make_lag_df(data, n_lags):
    df = reduce(lambda left, right: pd.concat([left, right], axis=1), [pd.DataFrame(data).shift(i) for i in range(1, n_lags+1)]).dropna()
    return df

def compute_mu(X, beta, n_eqs):    
    mu = pt.concatenate([X @ beta[i].ravel()[:, None] for i in range(n_eqs)], axis=-1)

    return mu

@np.vectorize
def str_add(x, y):
    return f'{x}{y}'

def make_coords_and_indices(n, T, region_names):
    if region_names is None:
        ALPHA = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        regions = ALPHA[:n]
    else:
        regions = region_names
    
    cov_mat_names = str_add(np.array(regions), np.array(regions)[:, None]).T
    
    pairs = list(combinations(regions, 2)) + list(combinations(regions[::-1], 2))[::-1]
    region_dict = {k:v for v, k in enumerate(regions)}
    pair_idx = np.array([region_dict[x[0]] for x in pairs])

    pairs = [''.join(x) for x in pairs]

    tril_idx = np.tril_indices(len(regions), k=-1)
    triu_idx = np.triu_indices(len(regions), k=1)

    n_regions = len(regions)
    triang_size = (n_regions * (n_regions - 1)) // 2

    coords = {
        'region_1':regions,
        'region_2':regions,
        'region_pairs':pairs,
        'position':['diag', 'offdiag'],
        'time':np.arange(T),
        'triang_names':cov_mat_names[np.tril_indices(len(regions))].ravel()
    }
    
    return coords, pair_idx, tril_idx, triu_idx, triang_size

def make_model(data, prior_params, observation_noise=False, estimate_x0=False, A_transform=None, 
               use_t_likelihood=False, 
               use_t_shocks=False,
               use_closed_form_cov = True,
               nu=3):
    
    T, n = data.shape
    region_names = getattr(data, 'columns', None)
    coords, pair_idx, tril_idx, triu_idx, triang_size = make_coords_and_indices(n, T, region_names)
    
    with pm.Model(coords=coords) as model:
        df_wide = make_lag_matrix(data, 1)    
        wide_data = pm.ConstantData('data', df_wide)
        
        if estimate_x0:
            x0 = pm.Normal('x0', 
                           mu=prior_params['x0']['mu'],
                           sigma=prior_params['x0']['sigma'],
                           dims=['region_1'])
        else:
            x0 = pt.zeros((1, n))
        
        wide_data = pt.concatenate([pt.atleast_2d(x0), wide_data], axis=0)
        n_nans = int(np.isnan(df_wide).ravel().sum())
        
        if n_nans > 0: 
            nan_idx = np.where(np.isnan(df_wide))
            missing_x = pm.Normal('missing_data', size = n_nans)
            wide_data = pt.set_subtensor(wide_data[nan_idx], missing_x)
            
        A_mu = pm.Normal('A_mu', 
                         mu=prior_params['A_mu']['mu'], 
                         sigma=prior_params['A_mu']['sigma'], 
                         dims=['position'])
        A_sigma = pm.Gamma('A_sigma', 
                           alpha=prior_params['A_sigma']['alpha'], 
                           beta=prior_params['A_sigma']['beta'], 
                           dims=['position'])
        
        A_offset_diag = pm.Normal('A_offsets_diag', dims=['region_1'])
        A_offset_offdiag = pm.Normal('A_offsets_offdiag', dims=['region_pairs'])

        A_diag = A_mu[0] +  A_sigma[0] * A_offset_diag
        A_offdiag = A_mu[1] + A_sigma[1] * A_offset_offdiag

        A = pt.diag(A_diag)
        A = pt.set_subtensor(A[triu_idx], A_offdiag[:triang_size])
        A = pt.set_subtensor(A[tril_idx], A_offdiag[triang_size:])
        
        if A_transform is not None:
            A = A_transform(A)
        
        A = pm.Deterministic('A', A, dims=['region_1', 'region_2']) 
        
        intercept = pm.Normal('intercept', 
                              mu=prior_params['intercept']['mu'], 
                              sigma=prior_params['intercept']['sigma'],
                              dims=['region_1'])
        
        var_state_mu = pm.Deterministic('mu', 
                                        compute_mu(beta=pt.expand_dims(A, 1), 
                                                   n_eqs=n,
                                                   X=wide_data),
                                        dims=['time', 'region_1'])
        
        if use_closed_form_cov:
            shock_sigma = pm.Gamma('shock_sigma', 
                                   alpha=prior_params['shock_sigma']['alpha'], 
                                   beta=prior_params['shock_sigma']['beta'], 
                                   dims=['region_1'])

            cov = A @ pt.diag(shock_sigma) @ A.T
            chol = pm.Deterministic('chol_cov', pt.linalg.cholesky(cov))
            chol_triangle = pm.Deterministic('chol_cov_triang', chol[np.tril_indices(n)], dims=['triang_names'])
        else:
            shock_sigma = pm.Gamma.dist(alpha=prior_params['shock_sigma']['alpha'], 
                                        beta=prior_params['shock_sigma']['beta'])
            chol, _, _ = pm.LKJCholeskyCov('chol_cov', n=n, eta=10, sd_dist=shock_sigma, dims=['triang_names'])
            chol_triangle = pm.Deterministic('chol_cov_triang', chol[np.tril_indices(n)], dims=['triang_names'])
        
        if observation_noise:
            obs_sigma = pm.Gamma('obs_sigma', alpha=prior_params['obs_sigma']['alpha'], beta=prior_params['obs_sigma']['alpha'], dims=['region_1'])
            
            if use_t_shocks:
                shock_offsets = pm.StudentT('shock_offsets', sigma=prior_params['shock_offsets']['sigma'], nu=nu, dims=['time', 'region_1'])
            else:
                shock_offsets = pm.Normal('shock_offsets', sigma=prior_params['shock_offsets']['sigma'], dims=['time', 'region_1'])
            shocks = (chol @ shock_offsets.T).T
        
            mu = pm.Deterministic('mu_plus_shocks', intercept + var_state_mu + shocks, dims=['time', 'region_1'])
            
            if use_t_likelihood:
                obs = pm.StudentT('obs', mu=mu, sigma=obs_sigma, nu=nu, observed=data, dims=['time', 'region_1'])
            else:
                obs = pm.Normal('obs', mu=mu, sigma=obs_sigma, observed=data, dims=['time', 'region_1'])
        
        else:
            if use_t_shocks:
                obs = pm.MvStudentT('obs', mu = intercept + var_state_mu, chol=chol, observed=data, dims=['time', 'region_1'])
            else:
                obs = pm.MvNormal('obs', mu=intercept + var_state_mu, chol=chol, observed=data, dims=['time', 'region_1'])
    
    return model

def prepare_gridspec_figure(n_cols: int, n_plots: int):
    """
     Prepare a figure with a grid of subplots. Centers the last row of plots if the number of plots is not square.

     Parameters
     ----------
     n_cols : int
         The number of columns in the grid.
     n_plots : int
         The number of subplots in the grid.

     Returns
     -------
     GridSpec
         A matplotlib GridSpec object representing the layout of the grid.
    list of tuple(slice, slice)
         A list of tuples of slices representing the indices of the grid cells to be used for each subplot.
    """

    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + 1

    gs = plt.GridSpec(2 * n_rows, 2 * n_cols)
    plot_locs = []

    for i in range(n_rows - int(has_remainder)):
        for j in range(n_cols):
            plot_locs.append((slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)))

    if has_remainder:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = int(n_cols - remainder)
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs