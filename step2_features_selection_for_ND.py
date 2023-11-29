import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import faiss
from sklearn.metrics import roc_auc_score
from configs import metadata_main_path, embeddings_main_path, id_ood_envs
from configs import DATASETS, FEATURES_TYPES, RANKING_METHODS, ND_METHODS

from read_data import get_samples


def apply_knn_aux(k,
                  train_samples,
                  test_samples_ood, test_labels_ood, ood_envs):
    """Auxiliary function for applying kNN

    Parameters
    ----------
    k: int
        Number of neighbors
    train_samples: numpy array
        Features matrix for training samples
    test_samples_ood: numpy array
        Features matrix for test samples - OOD
    test_labels_ood: list
        List of labels for test samples - OOD
    ood_envs: list
        List of OOD environments
    """

    def knn_rule(D): return np.mean(D, axis=1)

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(train_samples.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(train_samples)

    df = pd.DataFrame(columns=['ROC-AUC', 'split', 'environment_name'])

    all_ood_roc_aucs = []
    for idx, ood_env in enumerate(ood_envs):
        D, _ = gpu_index.search(test_samples_ood[idx], k)
        scores_in = knn_rule(D)
        test_ood_env_roc_auc = roc_auc_score(
            test_labels_ood[idx], scores_in) * 100

        row = [test_ood_env_roc_auc, 'OOD', ood_env]
        df.loc[len(df)] = row
        # print(row)
        # sys.stdout.flush()

        all_ood_roc_aucs.append(test_ood_env_roc_auc)

    gpu_index.reset()
    del index
    del gpu_index

    return df


def apply_knn(train_samples,
              test_samples_ood, test_labels_ood, ood_envs):
    """Apply kNN

    Parameters
    ----------
    train_samples: numpy array
        Features matrix for training samples
    test_samples_ood: numpy array
        Features matrix for test samples - OOD
    test_labels_ood: list
        List of labels for test samples - OOD
    ood_envs: list
        List of OOD environments
    """
    return apply_knn_aux(k=30,
                         train_samples=train_samples,
                         test_samples_ood=test_samples_ood, test_labels_ood=test_labels_ood, ood_envs=ood_envs)


def apply_features_selection_based_on_ranking(samples, n_features, ordered_indexes):
    """Apply features selection based on ranking

    Parameters
    ----------
    samples: numpy array
        Features matrix for all samples
    n_features: int
        Number of selected features
    ordered_indexes: numpy array
        Array of indexes of features sorted by ranking
    """
    ordered_indexes_ = ordered_indexes[0:n_features]
    local_samples = np.ascontiguousarray(samples[:, ordered_indexes_])
    return local_samples


def evaluate_ND_ranking_sel(
        ordered_indexes, features_percents,
        train_samples,
        test_samples_ood, test_labels_ood, ood_envs, nd_method_fct):
    """Evaluate novelty detection performance with different percents of features

    Parameters
    ----------
    ordered_indexes: numpy array
        Array of indexes of features sorted by ranking
    features_percents: list
        List of percents of features to be used
    train_samples: numpy array
        Features matrix for training samples
    test_samples_ood: list of numpy array
        List of features matrix for each OOD environment
    test_labels_ood: list
        List of lists of labels for each OOD environment
    ood_envs: list
        List of OOD environments
    nd_method_fct: function
        Function for novelty detection algorithm
    """

    dfs = []
    for percent in features_percents:

        n_features = int(train_samples.shape[1]*percent//100)

        local_train_samples = apply_features_selection_based_on_ranking(
            samples=train_samples, n_features=n_features, ordered_indexes=ordered_indexes)
        local_test_samples_ood = [apply_features_selection_based_on_ranking(
            samples=samples, n_features=n_features, ordered_indexes=ordered_indexes) for samples in test_samples_ood]

        df_ = nd_method_fct(
            train_samples=local_train_samples,
            test_samples_ood=local_test_samples_ood, test_labels_ood=test_labels_ood, ood_envs=ood_envs)

        df_['features_percent'] = percent

        dfs.append(df_)
    df = pd.concat(dfs, ignore_index=True)
    return df


nd_methods_dict = {
    'kNN': apply_knn
}


def run(dataset_name, features_type, feature_ranking_method,
        nd_method, step1_results_path, main_out_path):
    """Run Step 2 - Features Selection for Robust Novelty Detection

    Parameters
    ----------
    dataset_name: str
        Name of the dataset (one of DATASETS from configs.py)
    features_type: str
        Type of features (one of FEATURES_TYPES from configs.py)
    feature_ranking_method: str
        Feature ranking method (one of RANKING_METHODS from configs.py)
    nd_method: str
        Novelty detection method (one of ND_METHODS from configs.py)
    step1_results_path: str
        Path to the output directory of step1_feature_ranking.py
    main_out_path: str
        Path to the output directory
    """

    features_percents = [100, 95, 90, 85, 80, 75, 70, 65,
                         60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

    results_out_path = os.path.join(
        main_out_path,  '%s_features_%s' % (dataset_name, features_type))
    os.makedirs(results_out_path, exist_ok=True)

    id_envs = id_ood_envs[dataset_name]['id_envs']
    ood_envs = id_ood_envs[dataset_name]['ood_envs']

    embeddings_current_path = os.path.join(
        embeddings_main_path[dataset_name], 'embeddings_%s.pt' % features_type)

    with open(embeddings_current_path, "rb") as fd:
        all_features = torch.load(fd)
    with open(metadata_main_path[dataset_name], "rb") as fd:
        all_metadata = pkl.load(fd)

    scaler_path = os.path.join(
        step1_results_path, '%s_features_%s' % (dataset_name, features_type), 'scaler.pkl')
    with open(scaler_path, "rb") as fd:
        scaler = pkl.load(fd)

    train_samples, _, _ = get_samples(dataset_name=dataset_name,
                                      all_metadata=all_metadata, all_features=all_features,
                                      add_normal_samples=True, add_anomaly_samples=False,
                                      selected_splits=['ID'], selected_envs=id_envs)
    train_samples = scaler.transform(train_samples)

    test_samples_ood = []
    test_labels_ood = []
    for ood_env in ood_envs:
        samples, labels, _ = get_samples(dataset_name=dataset_name,
                                         all_metadata=all_metadata, all_features=all_features,
                                         add_normal_samples=True, add_anomaly_samples=True,
                                         selected_splits=['OOD'], selected_envs=[ood_env])
        samples = scaler.transform(samples)

        test_samples_ood.append(samples)
        test_labels_ood.append(labels)

    sel_method_path = os.path.join(
        step1_results_path, '%s_features_%s' % (dataset_name, features_type), feature_ranking_method)
    nd_method_fct = nd_methods_dict[nd_method]

    ordered_indexes = np.load(os.path.join(
        sel_method_path, 'ranking.npy'))

    df = evaluate_ND_ranking_sel(ordered_indexes=ordered_indexes,
                                 features_percents=features_percents,
                                 train_samples=train_samples,
                                 test_samples_ood=test_samples_ood, test_labels_ood=test_labels_ood, ood_envs=ood_envs, nd_method_fct=nd_method_fct)
    df['ranking_method'] = feature_ranking_method
    df['dataset_name'] = dataset_name
    df['features_type'] = features_type
    df['nd_method'] = nd_method
    df.to_csv(os.path.join(
        results_out_path, 'results__%s_%s.csv' % (feature_ranking_method, nd_method)))

    print('------------------------------------------------')
    df_100 = df[df['features_percent'] == 100]
    avg_roc_auc_all_features = df_100['ROC-AUC'].mean()
    print('Average ROC-AUC (over OOD environments) with all features: %4.2f' %
          avg_roc_auc_all_features)

    best_percent = 0
    best_roc_auc = 0
    for percent in features_percents:
        df_ = df[df['features_percent'] == percent]
        avg_roc_auc = df_['ROC-AUC'].mean()
        print('%d%% features -- %4.2f' % (percent, avg_roc_auc))
        if avg_roc_auc > best_roc_auc:
            best_percent = percent
            best_roc_auc = avg_roc_auc
    print('Best average ROC-AUC (over OOD environments) %4.2f (%d%% features)' %
          (best_roc_auc, best_percent))
    print('------------------------------------------------')

# Usage example:
#   python step2_features_selection_for_ND.py COCOShift95 resnet18 Stylist kNN ./results/ranking_methods ./results/nd_methods
#   python step2_features_selection_for_ND.py DomainNet resnet18 Stylist kNN ./results/ranking_methods ./results/nd_methods
#   python step2_features_selection_for_ND.py FMoW resnet18 Stylist kNN ./results/ranking_methods ./results/nd_methods


if __name__ == '__main__':
    # name of the dataset (one of DATASETS from configs.py)
    dataset_name = sys.argv[1]
    # type of features (one of FEATURES_TYPES from configs.py)
    features_type = sys.argv[2]
    # feature ranking method (one of RANKING_METHODS from configs.py)
    feature_ranking_method = sys.argv[3]
    # novelty detection method (one of ND_METHODS from configs.py)
    nd_method = sys.argv[4]
    # path to the output directory of step1_feature_ranking.py
    step1_results_path = sys.argv[5]
    # path to the output directory
    main_out_path = sys.argv[6]

    assert dataset_name in DATASETS, "invalid dataset name"
    assert features_type in FEATURES_TYPES, "invalid features type"
    assert feature_ranking_method in RANKING_METHODS, "invalid feature ranking method"
    assert nd_method in ND_METHODS, "invalid novelty detection method"

    run(dataset_name, features_type, feature_ranking_method,
        nd_method, step1_results_path, main_out_path)
