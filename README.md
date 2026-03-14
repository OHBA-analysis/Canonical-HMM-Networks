# Canonical HMM Networks for Studying M/EEG

This repository contains canonical Hidden Markov Models (HMMs) that were pre-trained on the [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) dataset (using the [time-delay embedding](https://www.nature.com/articles/s41467-018-05316-z) approach).

## Preprint

https://www.biorxiv.org/content/10.1101/2025.10.21.683692v2

## How to use this repo

### Installation

To run the code you need to install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html) and [osl-dynamics](https://osl-dynamics.readthedocs.io/en/latest/install.html). Note, [osl-dynamics](https://osl-dynamics.readthedocs.io/en/latest/install.html) will install [MNE-Python](https://mne.tools/stable/index.html) automatically.

We recommend running the Jupyter Notebook scripts using [VSCode](https://code.visualstudio.com/). **Use the `osld` kernel (conda environment) when running the scripts.**

### Loading a canonical HMM

The `/models` directory contains all the data files that contain the model weights. This directory can be downloaded on it's own or you can clone the entire repository.

The `modules/hmm.py` file contains a useful function for loading a canonical HMM within the osl-dynamics package. If you are running scripts from outside this repository, you need to add the repository to your Python path:

```python
import sys
sys.path.append("/path/to/Canonical-HMM-Networks")

from modules import hmm

model = hmm.load_canonical_hmm(n_states=8, parcellation="Glasser52", models_dir="/path/to/Canonical-HMM-Networks/models")
```

If you are running from within the repository directory, you can simply do:

```python
from modules import hmm

model = hmm.load_canonical_hmm(n_states=8, parcellation="Glasser52", models_dir="./models")
```

The data processing pipeline (preprocessing, source reconstruction, parcellation, etc.) is provided by the `osl_dynamics.meeg` subpackage. See the tutorials below for usage examples.

### Tutorials

The data preparation (including preprocessing, source reconstruction and parcellation) needs to be (roughly) matched to the training data for the canonical HMM. The tutorials illustrate how to perform these steps on new data. [MNE-Python](https://mne.tools/stable/index.html) is used to do the data processing. The most important thing to match is the sampling frequency (250 Hz) and parcellation.

#### Elekta MEG

- See `elekta_meg_parcel.ipynb` for a start-to-end tutorial on applying the canonical HMM to parcellated Elekta MEG data.

- See `elekta_meg_sensor.ipynb` for a tutorial on applying the canonical HMM to sensor-level data.

#### CTF MEG

- See `ctf_meg.ipynb` for a start-to-end tutorial on applying the canonical HMM to parcellated CTF MEG data.

#### OPM

- See `opm.ipynb` for a start-to-end tutorial on applying the canonical HMM to parcellated OPM data.

#### EEG

- See `eeg.ipynb` for a start-to-end tutorial on applying the canonical HMM to parcellated EEG data. Note, to source reconstruct/parcellate EEG data we need medium/high-density EEG, e.g. ~64+ channels.

Also see the `/contributions` directory for further example scripts and tutorials.

### Parallelisation

For parallelised batch processing scripts, see [examples](https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc) in osl-dynamics.

## Models / Parcellations

Canonical HMMs are available for the following parcellations:

| Name      | parcellation\_file                                   |
|-----------|------------------------------------------------------|
| Giles38   | atlas-Giles\_nparc-38\_space-MNI\_res-8x8x8.nii.gz   |
| Glasser52 | atlas-Glasser\_nparc-52\_space-MNI\_res-8x8x8.nii.gz |
| DK54      | atlas-DK\_nparc-54\_space-MNI\_res-8x8x8.nii.gz      |

For more information regarding the parcellations, see [here](https://osl-dynamics.readthedocs.io/en/latest/parcellations/index.html).

There is also a sensor-level canonical HMM available for Elekta MEG. New parcellations could be made available on request, however, the rank of the [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) data limits us to ~50 parcels.

## Citation

If you find this resource useful, please cite the repository:

> **Gohil, C., & Woolrich, M. W. (2025). Canonical HMM Networks (Version 0.5.0) [Computer software]. https://doi.org/10.5281/zenodo.17583973.**

And following papers:

> **Gohil, C., Huang, R., Higgins, C., van Es, M. W., Quinn, A. J., Vidaurre, D., & Woolrich, M. W. (2025). Canonical Hidden Markov Model Networks for Studying M/EEG. bioRxiv, 2025-10.**

> **Gohil, C., Huang, R., Roberts, E., van Es, M. W., Quinn, A. J., Vidaurre, D., & Woolrich, M. W. (2024). osl-dynamics, a toolbox for modeling fast dynamic brain activity. Elife, 12, RP91949.**

## Getting help

Please open an [issue](https://github.com/OHBA-analysis/Canonical-HMM-Networks/issues) on this repository if you run into errors, need help or spot any typos. Alternatively, you can email chetan.gohil@psych.ox.ac.uk.
