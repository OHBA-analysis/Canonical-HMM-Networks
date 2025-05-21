"""Example script for using a canonical HMM.

"""

import pickle
import numpy as np
from osl_dynamics.models.hmm import Config, Model

def load_canonical_hmm(n_states):
    means = np.load(f"giles_parcellation/{n_states:02d}_states/means.npy")
    covs = np.load(f"giles_parcellation/{n_states:02d}_states/covs.npy")
    trans_prob = np.load(f"giles_parcellation/{n_states:02d}_states/trans_prob.npy")
    config = Config(
        n_states=n_states,
        n_channels=80,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        initial_means=means,
        initial_covariances=covs,
        initial_trans_prob=trans_prob,
        batch_size=64,
        learning_rate=0.01,
        n_epochs=20,
    )
    return Model(config)

# Load data
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=8)

# Prepare
pca_components = np.load("giles_parcellation/pca_components.npy")
template_cov = np.load("giles_parcellation/template_cov.npy")
data.prepare({
    "align_channel_signs": {"template_cov": template_cov, "n_embeddings": 15},
    "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
    "standardize": {},
})

# Load model
model = load_canonical_hmm(n_states=12)
model.summary()

# Estimate state probabilities
alp = model.get_alpha(data)
pickle.dump(alp, open("alp.pkl", "wb"))
