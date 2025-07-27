# Canonical HMMs

Hidden Markov Models (HMMs) pre-trained on [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) (TDE-PCA MEG).

## Preprocessing

`38ROI_Giles` and `52ROI_Glasser` are source (parcel) level canonical HMMs.

Source reconstruction was performed with a volumetric LCMV beamformer. Parcellated data must be at 250 Hz.

Corresponding parcellations:

- `38ROI_Giles`: `fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz`.
- `52ROI_Glasser`: `Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz`.

## Prerequisites

To run these scripts you need to install [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics).

## Getting help

Please open an issue on this repository if you run into errors, need help or spot any typos. Alternatively, you can email chetan.gohil@psych.ox.ac.uk.
