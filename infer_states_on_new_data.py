"""Example script for inferring states using a canonical HMM.

"""

import pickle
import numpy as np
from glob import glob

from osl_dynamics.models.hmm import Config, Model


def prepare_data_for_canonical_hmm(data, parcellation):
    available_parcellations = []
    if parcellation not in available_parcellations:
        raise ValueError(f"parcellation much be one of: {available_parcellations}")
    pca_components = np.load(f"models/{parcellation}/pca_components.npy")
    template_cov = np.load(f"models/{parcellation}/template_cov.npy")
    data.prepare({
        "align_channel_signs": {"template_cov": template_cov, "n_embeddings": 15},
        "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
        "standardize": {},
    })
    return data

def load_canonical_hmm(n_states, sequence_length=400, batch_size=64):
    available_parcellations = []
    if parcellation not in available_parcellations:
        raise ValueError(f"parcellation much be one of: {available_parcellations}")
    means = np.load(f"models/{parcellation}/{n_states:02d}_states/means.npy")
    covs = np.load(f"models/{parcellation}/{n_states:02d}_states/covs.npy")
    trans_prob = np.load(f"models/{parcellation}/{n_states:02d}_states/trans_prob.npy")
    initial_state_probs = np.load(f"models/{parcellation}/{n_states:02d}_states/initial_state_probs.npy")
    config = Config(
        n_states=n_states,
        n_channels=means.shape[-1],
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        initial_means=means,
        initial_covariances=covs,
        initial_trans_prob=trans_prob,
        initial_state_probs=initial_state_probs,
        batch_size=batch_size,
        learning_rate=0.01,  # we won't train the model, this hyperparameter doesn't matter
        n_epochs=20,  # we won't train the model, this hyperparameter doesn't matter
    )
    return Model(config)

# Load data
files = sorted(glob("output/sub-*/parc/lcmv-parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=8)
data = prepare_data_for_canonical_hmm(data, parcellation="38ROI_Giles")

# Load model
model = load_canonical_hmm(n_states=12, parcellation="38ROI_Giles")
model.summary()

# Estimate state probabilities
alp = model.get_alpha(data)
pickle.dump(alp, open("alp.pkl", "wb"))
