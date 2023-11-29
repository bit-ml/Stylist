import sys
import numpy as np
from configs import datasets_norm_data_tags, datasets_anomaly_data_tags


def aux_get_samples(all_metadata, all_features,
                    selected_splits, selected_envs, data_tag, label):
    """Auxiliary function for extracting split samples 
    (selects either normal or anomalies) 

    Parameters 
    ----------
    all_metadata : dict
        Dictionary with metadata for all splits
    all_features: dict
        Dictionary with features for all samples
    selected_splits: list
        list of selected splits ('ID'/'ID_val'/'ID_test'/'OOD')
    selected_envs: list
        list of selected envs
        if empty => all envs will be considered
    data_tag: str or int 
        tag for the data type (normal/anomaly) - according to dataset 
    label: int
        label for the data type (normal/anomaly => 0/1)
    """

    all_samples = []
    all_env_labels = []

    for split in selected_splits:
        metadata = all_metadata[data_tag][split]
        if len(selected_envs) == 0:
            img_paths = [data[0] for data in metadata]
            env_labels = [data[2] for data in metadata]
        else:
            img_paths = [data[0]
                         for data in metadata if data[2] in selected_envs]
            env_labels = [data[2]
                          for data in metadata if data[2] in selected_envs]
        samples = [all_features[img_path].data.to(
            "cpu").numpy() for img_path in img_paths]
        all_samples += samples
        all_env_labels += env_labels
    labels = [label for i in range(len(all_samples))]
    return all_samples, labels, all_env_labels


def get_samples(dataset_name,
                all_metadata, all_features,
                add_normal_samples, add_anomaly_samples,
                selected_splits,
                selected_envs):
    """Returns split samples 

    Parameters 
    ----------
    dataset_name : str
        Name of the dataset
    all_metadata : dict
        Dictionary with metadata for all splits
    all_features: dict
        Dictionary with features for all samples
    add_normal_samples : bool
        Whether to add normal samples
    add_anomaly_samples : bool
        Whether to add anomaly samples
    selected_splits: list
        list of selected splits ('ID'/'ID_val'/'ID_test'/'OOD')
    selected_envs: list
        list of selected envs
        if empty => all envs will be considered
    """
    norm_data_tag = datasets_norm_data_tags[dataset_name]
    anomaly_data_tag = datasets_anomaly_data_tags[dataset_name]

    if add_normal_samples:
        normal_samples, normal_labels, normal_env_labels = aux_get_samples(all_metadata=all_metadata, all_features=all_features,
                                                                           selected_splits=selected_splits, selected_envs=selected_envs,
                                                                           data_tag=norm_data_tag, label=0)
    else:
        normal_samples, normal_labels, normal_env_labels = [], [], []

    if add_anomaly_samples:
        anomaly_samples, anomaly_labels, anomaly_env_labels = aux_get_samples(all_metadata=all_metadata, all_features=all_features,
                                                                              selected_splits=selected_splits, selected_envs=selected_envs,
                                                                              data_tag=anomaly_data_tag, label=1)
    else:
        anomaly_samples, anomaly_labels, anomaly_env_labels = [], [], []
    print(selected_splits, selected_envs)
    print('#normals - %d -- #anomalies - %d' %
          (len(normal_samples), len(anomaly_samples)))
    sys.stdout.flush()
    samples = normal_samples+anomaly_samples
    labels = normal_labels+anomaly_labels
    env_labels = normal_env_labels + anomaly_env_labels

    samples = np.array(samples).astype(np.float32)
    return samples, labels, env_labels
