# Main path for embeddings and data splits
data_path = './Stylist data'

# paths to precomputed embeddings
embeddings_main_path = {
    'DomainNet': data_path + '/embeddings/DomainNet',
    'FMoW': data_path + '/embeddings/FMoW',
    'COCOShift_balanced': data_path + '/embeddings/COCOShift',
    'COCOShift75': data_path + '/embeddings/COCOShift',
    'COCOShift90': data_path + '/embeddings/COCOShift',
    'COCOShift95': data_path + '/embeddings/COCOShift'
}

# paths to dataset splits
metadata_main_path = {
    'DomainNet': data_path + '/splits/DomainNet/DomainNet.pkl',
    'FMoW': data_path + '/splits/FMoW/FMoW.pkl',
    'COCOShift_balanced': data_path + '/splits/COCOShift/COCOShift_balanced.pkl',
    'COCOShift75': data_path + '/splits/COCOShift/COCOShift75.pkl',
    'COCOShift90': data_path + '/splits/COCOShift/COCOShift90.pkl',
    'COCOShift95': data_path + '/splits/COCOShift/COCOShift95.pkl'
}

# available datasets
DATASETS = ['COCOShift_balanced', 'COCOShift75',
            'COCOShift90', 'COCOShift95', 'DomainNet', 'FMoW']

# available features
FEATURES_TYPES = ['resnet18', 'clip']

# available features selection methods
RANKING_METHODS = ['random',
                   'PCA_loadings', 'InfoGain', 'FisherScore', 'MAD', 'Dispersion', 'Variance',
                   'Stylist']

# available novelty detection algorithms
ND_METHODS = ['kNN']

# baseline feature selection methods that consider style information
env_aware_baselines = ['InfoGain', 'FisherScore']

# selection method for which we apply standardization before ranking
STANDARDIZE_BEFORE_RANKING = {
    'random': 0,
    'PCA_loadings': 1,
    'InfoGain': 0,
    'FisherScore': 0,
    'MAD': 0,
    'Dispersion': 1,
    'Variance': 0,
    'Stylist': 0
}

# environments per dataset
id_ood_envs = {
    'DomainNet': {
        'id_envs': ['painting', 'real', 'clipart', 'infograph'],
        'ood_envs': ['sketch', 'quickdraw']
    },
    'FMoW': {
        'id_envs': [0, 1, 2, 3],
        'ood_envs': [4]
    },
    'COCOShift_balanced': {
        'id_envs': ['forest', 'mountain', 'field', 'rock', 'farm'],
        'ood_envs': ['lake', 'seaside', 'garden', 'sport_field']
    },
    'COCOShift75': {
        'id_envs': ['forest', 'mountain', 'field', 'rock', 'farm'],
        'ood_envs': ['lake', 'seaside', 'garden', 'sport_field']
    },
    'COCOShift90': {
        'id_envs': ['forest', 'mountain', 'field', 'rock', 'farm'],
        'ood_envs': ['lake', 'seaside', 'garden', 'sport_field']
    },
    'COCOShift95': {
        'id_envs': ['forest', 'mountain', 'field', 'rock', 'farm'],
        'ood_envs': ['lake', 'seaside', 'garden', 'sport_field']
    }
}

# normal class tags per dataets
datasets_norm_data_tags = {
    'FMoW': 0,
    'DomainNet': 0,
    'COCOShift_balanced': 'norm',
    'COCOShift75': 'norm',
    'COCOShift90': 'norm',
    'COCOShift95': 'norm'
}

# novelty class tags per dataset
datasets_anomaly_data_tags = {
    'FMoW': 1,
    'DomainNet': 1,
    'COCOShift_balanced': 'novelty0',
    'COCOShift75': 'novelty0',
    'COCOShift90': 'novelty0',
    'COCOShift95': 'novelty0'
}
