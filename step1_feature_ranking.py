import os
import sys
import torch
import numpy as np
import pickle as pkl
import random
from sklearn.preprocessing import StandardScaler
from configs import id_ood_envs, embeddings_main_path, metadata_main_path
from stylist_utils import get_stylist_wass_distance, get_stylist_kl_distance
from stylist_utils import get_stylist_mean_order, get_stylist_medianranking_order, get_stylist_weightedranking_order
from baselines import get_random_features, get_pca_loadings_features, get_infogain_features, get_mad_features, get_variance_features, get_fisherscore_features, get_dispersion_features
from read_data import get_samples

from configs import DATASETS, FEATURES_TYPES, RANKING_METHODS
from configs import env_aware_baselines
from configs import STANDARDIZE_BEFORE_RANKING

BASELINE_RANKING_METHODS_FUNCTIONS = {
    'PCA_loadings': get_pca_loadings_features,
    'InfoGain': get_infogain_features,
    'FisherScore': get_fisherscore_features,
    'MAD': get_mad_features,
    'Dispersion': get_dispersion_features,
    'Variance': get_variance_features
}


def get_Stylist_features(samples, env_labels, method_out_path, distance_name, ranking_method):
    """Get Stylist feature ranking 

    Parameters
    ----------
    samples : numpy array
        Features matrix for all samples 
    env_labels : list
        List of environment labels for all samples
    method_out_path : str
        Path to the output directory
    distance_name : str
        Name of the distance metric (one of ['Wass', 'KL'])
    ranking_method : str
        Name of the ranking method (one of ['mean', 'medianranking', 'weightedranking']) 
    """
    unique_env_labels = list(set(env_labels))
    per_env_samples = {}
    for env in unique_env_labels:
        env_samples = [samples[i]
                       for i, env_ in enumerate(env_labels) if env_ == env]
        per_env_samples[env] = np.array(env_samples)

    distances_dict = {
        'Wass': get_stylist_wass_distance,
        'KL': get_stylist_kl_distance
    }
    distances = distances_dict[distance_name](per_env_samples)

    ranking_dict = {
        'mean': get_stylist_mean_order,
        'medianranking': get_stylist_medianranking_order,
        'weightedranking': get_stylist_weightedranking_order
    }
    indexes = ranking_dict[ranking_method](distances)

    out_path = os.path.join(method_out_path, 'ranking.npy')
    np.save(out_path, indexes)
    print('Feature ranking saved at: ', out_path)
    sys.stdout.flush()


def run(dataset_name, features_type, feature_ranking_method, main_out_path):
    """Run Step 1 - Feature Ranking

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (one of DATASETS from configs.py)
    features_type : str
        Type of features (one of FEATURES_TYPES from configs.py)
    feature_ranking_method : str
        Feature ranking method (one of RANKING_METHODS from configs.py)
    main_out_path : str
        Path to the output directory
    """

    ranking_out_path = os.path.join(
        main_out_path, '%s_features_%s' % (dataset_name, features_type))
    os.makedirs(ranking_out_path, exist_ok=True)

    id_envs = id_ood_envs[dataset_name]['id_envs']

    embeddings_current_path = os.path.join(
        embeddings_main_path[dataset_name], 'embeddings_%s.pt' % features_type)

    with open(embeddings_current_path, "rb") as fd:
        all_features = torch.load(fd)
    with open(metadata_main_path[dataset_name], "rb") as fd:
        all_metadata = pkl.load(fd)

    samples, _, env_labels = get_samples(dataset_name=dataset_name,
                                         all_metadata=all_metadata, all_features=all_features,
                                         add_normal_samples=True, add_anomaly_samples=False,
                                         selected_splits=['ID'], selected_envs=id_envs)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(samples)
    standardize = STANDARDIZE_BEFORE_RANKING[feature_ranking_method]
    if standardize == 1:
        samples_std = scaler.transform(samples)
    else:
        samples_std = samples

    scaler_out_path = os.path.join(ranking_out_path,
                                   'scaler.pkl')
    with open(scaler_out_path, "wb") as fd:
        pkl.dump(scaler, fd)

    method_out_path = os.path.join(
        ranking_out_path, feature_ranking_method)
    os.makedirs(method_out_path, exist_ok=True)

    if feature_ranking_method == 'random':
        # 115 / 42 / 10 / 0 / 15 / 300
        random_seed = 115
        get_random_features(samples=samples_std, method_out_path=method_out_path,
                            seed=random_seed)
    elif feature_ranking_method == 'Stylist':
        # Wass / KL
        distance_name = 'Wass'
        # mean / medianranking / weightedranking
        ranking_method = 'mean'
        get_Stylist_features(samples=samples_std, env_labels=env_labels,
                             method_out_path=method_out_path, distance_name=distance_name, ranking_method=ranking_method)
    elif feature_ranking_method in env_aware_baselines:
        BASELINE_RANKING_METHODS_FUNCTIONS[feature_ranking_method](
            samples_std, env_labels, method_out_path)
    else:
        BASELINE_RANKING_METHODS_FUNCTIONS[feature_ranking_method](
            samples_std, method_out_path)

# Usage examples:
#   python step1_feature_ranking.py COCOShift95 resnet18 Stylist ./results/ranking_methods
#   python step1_feature_ranking.py DomainNet resnet18 Stylist ./results/ranking_methods
#   python step1_feature_ranking.py FMoW resnet18 Stylist ./results/ranking_methods


if __name__ == '__main__':
    # name of the dataset (one of DATASETS from configs.py)
    dataset_name = sys.argv[1]
    # type of features (one of FEATURES_TYPES from configs.py)
    features_type = sys.argv[2]
    # feature ranking method (one of RANKING_METHODS from configs.py)
    feature_ranking_method = sys.argv[3]
    # path to the output directory
    main_out_path = sys.argv[4]

    assert dataset_name in DATASETS, "invalid dataset name"
    assert features_type in FEATURES_TYPES, "invalid features type"
    assert feature_ranking_method in RANKING_METHODS, "invalid feature ranking method"

    run(dataset_name, features_type, feature_ranking_method, main_out_path)
