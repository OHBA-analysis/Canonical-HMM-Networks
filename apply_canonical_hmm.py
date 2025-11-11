"""Example script for applying the HMM to new data.

This script can be used if you already have the data in a format ready
for the canonical HMM.

This script prepares the data in parallel. This is useful for large datasets.
"""

import pickle
from glob import glob
from osl_dynamics.data import Data
from modules import hmm

# Load data
files = sorted(glob("BIDS/derivatives/*/lcmv-parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=8)
data = hmm.prepare_data_for_canonical_hmm(data, parcellation="38ROI_Giles")

# Load model
model = hmm.load_canonical_hmm(n_states=12, parcellation="38ROI_Giles")
model.summary()

# Estimate state probabilities
alp = model.get_alpha(data)
pickle.dump(alp, open("alp.pkl", "wb"))
