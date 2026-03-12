"""Step 4: Forward Model, Source Reconstruction and Parcellation.

Computes the forward model, LCMV beamformer, parcellates the voxel data
and applies symmetric orthogonalisation for each session in parallel.
"""

import sys
import traceback
from pathlib import Path
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

# --------- Config ---------
project_dir = Path("/Users/gohil/Desktop/Canonical-HMM-Networks")
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/4_source_recon")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
gridstep = 8  # dipole grid resolution in mm
chantypes = ["mag", "grad"]
rank = {"meg": 60}
parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"
parcellation_method = "spatial_basis"
orthogonalisation = "symmetric"
use_mni152 = False
n_workers = 4
# ---------------------------

sys.path.append(str(project_dir))

from modules import rhino, source_recon, parcellation, utils
from modules.utils import SessionLogger


def process_session(session):
    """Source reconstruct and parcellate a single session."""
    id, info = session
    with SessionLogger(id, log_dir) as logger:
        try:
            preproc_file = output_dir / "preprocessed" / f"{id}_preproc-raw.fif"

            if use_mni152:
                surfaces_dir = str(project_dir / "mni152_surfaces")
            else:
                surfaces_dir = str(output_dir / "anat_surfaces" / info["subject"])

            fns = utils.OSLFilenames(
                outdir=str(output_dir / "osl"),
                id=id,
                preproc_file=str(preproc_file),
                surfaces_dir=surfaces_dir,
            )

            logger.log("Computing forward model...")
            rhino.forward_model(fns, model="Single Layer", gridstep=gridstep)

            logger.log("Computing LCMV beamformer...")
            source_recon.lcmv_beamformer(fns, chantypes=chantypes, rank=rank)

            logger.log("Applying LCMV beamformer...")
            voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns)

            logger.log("Parcellating...")
            parcel_data = parcellation.parcellate(
                fns,
                voxel_data,
                voxel_coords,
                method=parcellation_method,
                orthogonalisation=orthogonalisation,
                parcellation_file=parcellation_file,
            )

            logger.log("Saving parcellated data...")
            raw = mne.io.read_raw_fif(str(preproc_file), preload=True)
            parc_fif = str(output_dir / "osl" / id / "lcmv-parc-raw.fif")
            parcellation.save_as_fif(
                parcel_data,
                raw,
                extra_chans="stim",
                filename=parc_fif,
            )

            logger.log("Saving plots...")
            session_plots_dir = plots_dir / id
            session_plots_dir.mkdir(parents=True, exist_ok=True)
            parcellation.plot_psds(
                parc_fif,
                parcellation_file=parcellation_file,
                filename=str(session_plots_dir / "4_psd_topo.png"),
            )
            plt.close("all")

            logger.log("Done.")
            return id, True
        except Exception as e:
            logger.error(str(e))
            traceback.print_exc()
            return id, False


if __name__ == "__main__":
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_session, sessions.items())
    failed = [id for id, ok in results if not ok]
    if failed:
        print(f"\nStep 4 finished with errors in: {', '.join(failed)}")
    else:
        print("\nStep 4 complete.")
