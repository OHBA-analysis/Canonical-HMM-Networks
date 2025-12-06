import mne
import numpy as np
from modules import preproc, rhino, source_recon, parcellation, utils

subject = "01"
task = "resteyesclosed"
id = f"sub-{subject}_task-{task}"
print("id:", id)

fns = utils.OSLFilenames(
    outdir="BIDS/derivatives/osl",
    id=id,
    preproc_file=f"BIDS/derivatives/preprocessed/{id}_preproc-raw.fif",
    surfaces_dir=f"BIDS/derivatives/anat_surfaces/sub-{subject}",  # use "mni152_surfaces" for the standard brain
)
print(fns)

raw = mne.io.read_raw_fif(fns.preproc_file)

source_recon.lcmv_beamformer(fns, raw, chantypes=["mag", "grad"], rank={"meg": 60})

voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)

parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

parcel_data = parcellation.parcellate(
    fns,
    voxel_data,
    voxel_coords,
    method="spatial_basis",
    orthogonalisation="symmetric",
    parcellation_file=parcellation_file,
)

parc_fif = f"BIDS/derivatives/osl/{id}/lcmv-parc-raw.fif"
parcellation.save_as_fif(
    parcel_data,
    raw,
    extra_chans="stim",
    filename=parc_fif,
)

parcellation.plot_psds(parc_fif, parcellation_file=parcellation_file, filename="psd.png")
