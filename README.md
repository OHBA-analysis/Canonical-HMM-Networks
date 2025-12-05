# Canonical HMM Networks for Studying M/EEG

This repository contains canonical Hidden Markov Models (HMMs) that were pre-trained on the [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) dataset (using the [time-delay embedding](https://www.nature.com/articles/s41467-018-05316-z) approach).

## Preprint

https://www.biorxiv.org/content/10.1101/2025.10.21.683692v1

## Installation

To run these scripts you need to install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html) and [osl-dynamics](https://osl-dynamics.readthedocs.io/en/latest/install.html). Note, [osl-dynamics](https://osl-dynamics.readthedocs.io/en/latest/install.html) will install [MNE-Python](https://mne.tools/stable/index.html) automatically.

Once you have installed the `osld` environment, a couple extra Python packages are needed. These can be installed via a terminal:
```
conda activate osld
pip install fslpy ipyevents
```

We recommend running the Jupyter Notebook scripts using [VSCode](https://code.visualstudio.com/). Use the `osld` kernel (conda environment) when running the scripts.

## Example Scripts

The data preparation (including preprocessing, source reconstruction and parcellation) needs to be (roughly) matched to the training data for the canonical HMM. The examples scripts illustrate how to perform these steps on new data. [MNE-Python](https://mne.tools/stable/index.html) is used to do the data processing. The most important thing to match is the sampling frequency (250 Hz) and parcellation.

Canonical HMMs are available for the following parcellations:

| Name           | parcellation\_file                                                     |
|----------------|------------------------------------------------------------------------|
| 38ROI\_Giles   | fmri\_d100\_parcellation\_with\_PCC\_reduced\_2mm\_ss5mm\_ds8mm.nii.gz |
| 52ROI\_Glasser | Glasser52\_binary\_space-MNI152NLin6\_res-8x8x8.nii.gz                 |

There is also a sensor-level canonical HMM available for Elekta MEG. New parcellations could be made available on request, however, the rank of the [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) data limits us to ~50 parcels.

### Elekta MEG

- See `elekta_meg_parcel.ipynb` for a start-to-end example script for applying the canonical HMM to parcellated Elekta MEG data.

- See `elekta_meg_sensor.ipynb` for an example of how to apply the canonical HMM to sensor-level data.

### CTF MEG

- See `ctf_meg.ipynb` for a start-to-end example script for applying the canonical HMM to parcellated CTF MEG data.

### OPM

- See `opm.ipynb` for a start-to-end-example script for applying the canonical HMM to parcellated OPM data.

### EEG

- See `eeg.ipynb` for a start-to-end example script for applying the canonical HMM to parcellated EEG data. Note, to source reconstruct/parcellate EEG data we need medium/high-density EEG, e.g. ~64+ channels.

## Contributions

Contributions are welcome, e.g. if you develop a useful preprocessing function or tutorial. If you would like to contribute something, please add it to the `/contributions` directory and open a [pull request](https://github.com/OHBA-analysis/Canonical-HMM-Networks/pulls).

Alternatively, if you would like to request a new feature, please open an [issue](https://github.com/OHBA-analysis/Canonical-HMM-Networks/issues).

## Getting help

Please open an [issue](https://github.com/OHBA-analysis/Canonical-HMM-Networks/issues) on this repository if you run into errors, need help or spot any typos. Alternatively, you can email chetan.gohil@psych.ox.ac.uk.
