import pymc as pm
import aesara.tensor as at
import numpy as np
import pandas as pd
import os
import arviz as az


def make_top_vocabulary_matrix(word_counts, count_matrix, vocabulary):
    top_vocab = [x[0] for x in word_counts.most_common(100)]
    top_idx = np.where([x in top_vocab for x in vocabulary])[0]
    X_top = count_matrix[:, top_idx]
    nonzero_idx = X_top.sum(axis=1).A1 != 0
    X_top = X_top[nonzero_idx, :]

    return X_top, top_vocab, nonzero_idx


def create_fourier_features(t, n, p=365.25):
    x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


def prepare_model_inputs(df, word_counts, count_matrix, vocabulary, n_components):
    X_top, top_vocab, nonzero_idx = make_top_vocabulary_matrix(word_counts, count_matrix, vocabulary)

    normed_date = df.date.dt.normalize()

    X_time = (pd.DataFrame(X_top.todense(), columns=top_vocab)
              .assign(time=normed_date.loc[nonzero_idx].values)
              .groupby('time').sum()
              .melt(ignore_index=False, var_name='word')
              .reset_index())

    time_idx, times = pd.factorize(X_time.time)
    word_idx, words = pd.factorize(X_time.word)

    time_range = (times.max() - times.min()).days
    time = ((times - times.min()).days / time_range).values

    components = [f'{f}_{i + 1}' for f in ['cos', 'sin'] for i in range(n_components)]
    X_fourier = create_fourier_features(time, n_components, p=time.max())

    coords = {'words': words, 'time': times, 'components': components}

    return X_time, X_fourier, time, coords, word_idx, time_idx


def create_pymc_model(X_time, X_fourier, time, coords, word_idx, time_idx):
    with pm.Model(coords=coords) as freq_model:
        mu_intercept = pm.Normal('mu_intercept', sigma=0.1)
        sigma_intercept = pm.Gamma('sigma_intercept', alpha=2, beta=10)
        offset_intercept = pm.Normal('offset_intercept', dims='words')
        intercept = pm.Deterministic('intercept', mu_intercept + sigma_intercept * offset_intercept, dims='words')

        mu_trend = pm.Normal('mu_trend', sigma=0.1)
        sigma_trend = pm.Gamma('sigma_trend', alpha=2, beta=10)
        offset_trend = pm.Normal('offset_trend', dims='words')
        trend = pm.Deterministic('trend', mu_trend + sigma_trend * offset_trend, dims='words')

        mu_seasonal = pm.Normal('mu_seasonal', sigma=0.1)
        sigma_seasonal = pm.Gamma('sigma_seasonal', alpha=2, beta=10)
        offset_seasonal = pm.Normal('offset_seasonal', dims=['words', 'components'])
        seasonal_beta = pm.Deterministic('seasonal_beta', mu_seasonal + sigma_seasonal * offset_seasonal,
                                         dims=['words', 'components'])

        latent_mu = intercept[word_idx] + trend[word_idx] * time[time_idx] + (
                seasonal_beta[word_idx] * X_fourier[time_idx]).sum(axis=-1)
        mu = pm.Deterministic('mu', at.exp(latent_mu))

        loglike = pm.Poisson('obs', mu=mu, observed=X_time.value)

    return freq_model


def bayesian_decomposition(df, word_counts, count_matrix, vocabulary, n_components,
                           out_path='../data/models',
                           save_posterior=True):
    model_inputs = prepare_model_inputs(df, word_counts, count_matrix, vocabulary, n_components)
    model = create_pymc_model(*model_inputs)

    with model:
        idata = pm.sample(init='jitter+adapt_diag_grad')
        idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    if save_posterior:
        fname = os.path.join(out_path, 'posterior.json')
        idata.to_json(fname)

    return idata


def load_posterior(model_path):
    fname = os.path.join(model_path, 'posterior.netcdf')
    idata = az.from_netcdf(fname)

    return idata
