from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import os
import cloudpickle as pickle

def build_pipeline(count_vec, n_topics):
    tfidf = TfidfTransformer()
    lda = LatentDirichletAllocation(n_components=n_topics)

    lda_pipeline = Pipeline([('count_vectorize', count_vec),
                         ('tfidf', tfidf),
                         ('lda', lda)])

    return lda_pipeline


def LDA(corpus, count_vec, n_topics,
        fit_model = True,
        save_model = True,
        out_fpath='../data/models/',
        out_fname='lda.p'):

    out_file = os.path.join(out_fpath, out_fname)
    if fit_model:
        pipeline = build_pipeline(count_vec, n_topics)
        pipeline.fit(corpus)

    else:
        with open(out_file, 'rb') as file:
            pipeline = pickle.load(file)

    if save_model and fit_model:
        with open(out_file, 'wb') as file:
            pickle.dump(pipeline, file)

    return pipeline
