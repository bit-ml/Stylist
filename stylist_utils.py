import numpy as np
from scipy.stats import wasserstein_distance, entropy


def wasserstein(data1, data2):
    per_feat_dist = np.zeros(data1.shape[1])
    for feat_id in range(data1.shape[1]):
        per_feat_dist[feat_id] = wasserstein_distance(
            data1[:, feat_id], data2[:, feat_id]
        )
    return per_feat_dist


def get_stylist_wass_distance(per_env_samples):
    w_inter_dist = []
    envs = list(per_env_samples.keys())
    for env1 in range(len(envs)):
        env1_data = per_env_samples[envs[env1]]

        for env2 in range(env1 + 1, len(envs)):
            env2_data = per_env_samples[envs[env2]]
            dist12 = wasserstein(env1_data, env2_data)

            w_inter_dist.append(dist12)
    w_inter_dist = np.array(w_inter_dist)
    return w_inter_dist


def kl(data1, data2):
    per_feat_dist = np.zeros(data1.shape[1])
    for feat_id in range(data1.shape[1]):
        feat_data1 = data1[:, feat_id]
        feat_data2 = data2[:, feat_id]
        min_val = min(feat_data1.min(), feat_data2.min())
        max_val = max(feat_data1.max(), feat_data2.max())
        bins = np.linspace(min_val, max_val, num=100)
        data1_hist, _ = np.histogram(feat_data1, bins=bins, density=True)
        data2_hist, _ = np.histogram(feat_data2, bins=bins, density=True)
        data1_hist += 1e-10
        data2_hist += 1e-10
        per_feat_dist[feat_id] = entropy(
            data1_hist, data2_hist) + entropy(data2_hist, data1_hist)

    return per_feat_dist


def get_stylist_kl_distance(per_env_samples):

    kl_inter_dist = []
    envs = list(per_env_samples.keys())
    for env1 in range(len(envs)):
        env1_data = per_env_samples[envs[env1]]

        for env2 in range(env1 + 1, len(envs)):
            env2_data = per_env_samples[envs[env2]]
            dist12 = kl(env1_data, env2_data)

            kl_inter_dist.append(dist12)
    kl_inter_dist = np.array(kl_inter_dist)
    return kl_inter_dist


def get_stylist_mean_order(distances):
    values_per_feat = distances.mean(axis=0)
    # small values => small inter-env distance => content features
    indexes = np.argsort(values_per_feat)
    return indexes


def build_count_matrix_from_pairs(distances):
    num_pairs = distances.shape[0]
    num_features = distances.shape[1]

    count_matrix = np.zeros((num_features, num_features))

    for pair_idx in range(num_pairs):
        pair = np.argsort(distances[pair_idx])
        for i in range(num_features):
            for j in range(i + 1, num_features):
                count_matrix[pair[i]][pair[j]] += 1

    return count_matrix


def get_stylist_medianranking_order(distances):
    count_matrix = build_count_matrix_from_pairs(distances)
    # high score => feat i is before other features many times => lower variance btw envs => content feature
    features_scores = count_matrix.sum(axis=1)
    # over -features_scores => content features are the first
    indexes = np.argsort(-features_scores)
    return indexes


def get_stylist_weightedranking_order(distances_):

    distances = distances_.copy()

    num_pairs = distances.shape[0]
    num_features = distances.shape[1]
    # for each pair
    distances = distances / distances.sum(axis=1)[:, None]
    # for each feature
    weighted_ranking = distances.sum(axis=0) / num_pairs
    # small values => small prob of being env feature
    indexes = np.argsort(weighted_ranking)
    return indexes
