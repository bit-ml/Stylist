import os
import sys
import time
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
from threadpoolctl import threadpool_limits


def get_random_features(samples, method_out_path, seed):
    random.seed(seed)
    indexes = np.arange(0, samples.shape[1])
    random.shuffle(indexes)

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_pca_loadings_features(samples, method_out_path):
    n_features = samples.shape[1]
    with threadpool_limits(limits=1):
        pca = PCA(n_components=n_features,
                  svd_solver='randomized', random_state=115)
        pca.fit(samples)
    loadings = pca.components_.T
    loadings = loadings ** 2
    loadings = loadings / np.sum(loadings, 0)
    loadings = loadings * np.sqrt(pca.explained_variance_)[None, :]
    loadings_mean = loadings.mean(1)
    indexes = np.argsort(loadings_mean)
    indexes = indexes[::-1].copy()

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_infogain_features(samples, env_labels, method_out_path):
    unique_env_labels = list(set(env_labels))
    env_labels_indexes = [unique_env_labels.index(
        label) for label in env_labels]
    importances = mutual_info_classif(samples, env_labels_indexes)
    indexes = np.argsort(importances)

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_mad_features(samples, method_out_path):
    mad_scores = np.sum(
        np.abs(samples - np.mean(samples, axis=0)), axis=0) / samples.shape[0]
    indexes = np.argsort(mad_scores)
    indexes = indexes[::-1]

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_variance_features(samples, method_out_path):
    var_scores = np.var(samples, axis=0)
    indexes = np.argsort(var_scores)
    indexes = indexes[::-1]

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_fisherscore_features(samples, env_labels, method_out_path):
    unique_env_labels = list(set(env_labels))
    env_labels_indexes = [unique_env_labels.index(
        label) for label in env_labels]
    indexes = list(np.arange(0, samples.shape[0]))
    import random
    random.seed(115)
    n_samples = min(10000, samples.shape[0])
    indexes = random.sample(indexes, n_samples)
    samples = samples[np.array(indexes), :]
    env_labels_indexes = [env_labels_indexes[i] for i in indexes]
    ranks = fisher_score.fisher_score(samples, env_labels_indexes)

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, ranks)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def get_dispersion_features(samples, method_out_path):
    samples = samples - np.min(samples, axis=0) + 1
    am = np.mean(samples, axis=0)
    gm = np.mean(np.log(samples), axis=0)
    gm = np.exp(gm)
    disp_ratio = am / gm
    indexes = np.argsort(disp_ratio)
    indexes = indexes[::-1]

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()
