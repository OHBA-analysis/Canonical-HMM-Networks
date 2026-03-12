"""Step 2: Surface Extraction.

Extracts inner skull, outer skin and brain surfaces from each subject's
structural MRI using FSL.
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
log_dir = Path("logs/2_surfaces")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
include_nose = False
n_workers = 4
# ---------------------------

sys.path.append(str(project_dir))

from modules import rhino
from modules.utils import SessionLogger

# Get unique subjects (extract_surfaces only needs to run once per subject)
subjects = list({info["subject"] for info in sessions.values()})


def process_subject(subject):
    """Extract surfaces for a single subject."""
    with SessionLogger(subject, log_dir) as logger:
        try:
            logger.log("Extracting surfaces...")

            mri_file = input_dir / subject / "anat" / f"{subject}_T1w.nii.gz"
            outdir = output_dir / "anat_surfaces" / subject

            rhino.extract_surfaces(
                mri_file=str(mri_file),
                outdir=str(outdir),
                include_nose=include_nose,
            )

            logger.log("Copying plots...")
            for id, info in sessions.items():
                if info["subject"] == subject:
                    session_plots_dir = plots_dir / id
                    session_plots_dir.mkdir(parents=True, exist_ok=True)
                    for png in outdir.glob("*.png"):
                        shutil.copy(png, session_plots_dir / f"2_{png.name}")

            logger.log("Done.")
            return subject, True
        except Exception as e:
            logger.error(str(e))
            traceback.print_exc()
            return subject, False


if __name__ == "__main__":
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_subject, subjects)
    failed = [s for s, ok in results if not ok]
    if failed:
        print(f"\nStep 2 finished with errors in: {', '.join(failed)}")
    else:
        print("\nStep 2 complete.")
