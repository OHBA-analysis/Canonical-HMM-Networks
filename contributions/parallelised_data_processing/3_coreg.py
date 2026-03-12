"""Step 3: Coregistration.

Coregisters MEG to MRI using Polhemus headshape points for each
session in parallel. Interactive HTML coregistration plots are saved
to plots_dir.
"""

import shutil
import sys
import traceback
from pathlib import Path
from multiprocessing import Pool

# --------- Config ---------
project_dir = Path("/Users/gohil/Desktop/Canonical-HMM-Networks")
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/3_coreg")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
use_nose = False
allow_mri_scaling = False  # set True if using MNI152 standard brain
use_mni152 = False  # set True to use standard brain instead of subject MRI
n_workers = 4
# ---------------------------

sys.path.append(str(project_dir))

from modules import rhino, utils
from modules.utils import SessionLogger


def process_session(session):
    """Coregister a single session."""
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

            logger.log("Extracting fiducials and headshape...")
            rhino.extract_fiducials_and_headshape_from_fif(fns)

            logger.log("Coregistering MEG to MRI...")
            rhino.coregister_head_and_mri(
                fns,
                use_nose=use_nose,
                allow_mri_scaling=allow_mri_scaling,
                plot_type="html",
            )

            logger.log("Copying plot...")
            session_plots_dir = plots_dir / id
            session_plots_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                Path(fns.coreg_dir) / "coreg.html",
                session_plots_dir / "3_coreg.html",
            )

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
        print(f"\nStep 3 finished with errors in: {', '.join(failed)}")
    else:
        print("\nStep 3 complete.")
