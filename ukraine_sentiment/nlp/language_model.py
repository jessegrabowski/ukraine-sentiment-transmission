from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode_expect_ascii
from string import punctuation
from collections import Counter

from wordcloud import WordCloud

import cloudpickle as pickle

import re
import os

STOP_WORDS = stopwords.words('english')
PUNCT_PATTERN = re.compile('[' + re.escape(punctuation) + ']')
STEMMER = WordNetLemmatizer().lemmatize


def tokenize(s):
    remove_non_latin = ''.join([c for c in s if ord(c) < 150]).strip()
    decoded = unidecode_expect_ascii(remove_non_latin)
    no_punct = re.sub(PUNCT_PATTERN, '', decoded)
    no_punct = re.sub(' +', ' ', no_punct)
    tokens = wordpunct_tokenize(no_punct)
    return [STEMMER(x.lower()) for x in tokens if x.lower() not in STOP_WORDS and len(x) > 0]


def create_corpus_from_processed_dataframe(df):
    corpus = df.event_description.str.strip().copy()
    corpus = corpus.apply(tokenize)

    return corpus


def count_words(corpus):
    word_counts = Counter()
    word_counts.update([w for l in corpus.to_list() for w in l])

    return word_counts


def make_wordcloud_from_corpus(corpus, out_path):
    cloud = WordCloud(collocations=False).generate(' '.join(x for l in corpus.to_list() for x in l))

    fig, ax = plt.subplots(dpi=144)
    ax.imshow(cloud, interpolation='bilinear')
    ax.axis('off')

    plt.savefig(out_path)


def fit_count_vectorizer(corpus, count_vec_kwargs,
                         pickle_model=True,
                         out_path='../data/models',
                         file_name='count_vec_model.p',
                         fit_model=True):
    out_fname = os.path.join(out_path, file_name)

    if fit_model:
        count_vec = CountVectorizer(**count_vec_kwargs)
    else:
        with open(out_fname, 'rb') as file:
            count_vec = pickle.load(file)

    if pickle_model and fit_model:
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        with open(out_fname, 'wb') as file:
            pickle.dump(count_vec, file)

    count_matrix = count_vec.fit_transform(corpus)
    vocabulary = count_vec.get_feature_names_out()

    return count_vec, count_matrix, vocabulary
