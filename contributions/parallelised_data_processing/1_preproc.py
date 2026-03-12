"""Step 1: Preprocessing.

Filters, downsamples and detects bad segments/channels for each session
in parallel using multiprocessing.
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
log_dir = Path("logs/1_preproc")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
n_workers = 4
# ---------------------------

sys.path.append(str(project_dir))

from modules import preproc
from modules.utils import SessionLogger


def process_session(session):
    """Preprocess a single session."""
    id, info = session
    with SessionLogger(id, log_dir) as logger:
        try:
            logger.log("Loading raw data...")
            raw_file = input_dir / info["subject"] / "meg" / info["file"]
            raw = mne.io.read_raw_fif(raw_file, preload=True)

            raw = raw.crop(tmax=30)

            logger.log("Filtering and downsampling...")
            raw = raw.resample(sfreq=250)
            raw = raw.filter(
                l_freq=1, h_freq=45,
                method="iir",
                iir_params={"order": 5, "ftype": "butter"},
            )
            raw = raw.notch_filter([50, 100])

            logger.log("Detecting bad segments...")
            raw = preproc.detect_bad_segments(raw, picks="mag")
            raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
            raw = preproc.detect_bad_segments(raw, picks="grad")
            raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")

            logger.log("Detecting bad channels...")
            raw = preproc.detect_bad_channels(raw, picks="mag")
            raw = preproc.detect_bad_channels(raw, picks="grad")

            logger.log("Saving plots...")
            session_plots_dir = plots_dir / id
            session_plots_dir.mkdir(parents=True, exist_ok=True)

            preproc.plot_sum_square_time_series(raw)
            plt.savefig(session_plots_dir / "1_sum_square.png", dpi=150, bbox_inches="tight")
            plt.close("all")

            preproc.plot_sum_square_time_series(raw, exclude_bads=True)
            plt.savefig(session_plots_dir / "1_sum_square_exclude_bads.png", dpi=150, bbox_inches="tight")
            plt.close("all")

            preproc.plot_channel_stds(raw)
            plt.savefig(session_plots_dir / "1_channel_stds.png", dpi=150, bbox_inches="tight")
            plt.close("all")

            logger.log("Saving preprocessed data...")
            preproc_out_dir = output_dir / "preprocessed"
            preproc_out_dir.mkdir(parents=True, exist_ok=True)
            outfile = preproc_out_dir / f"{id}_preproc-raw.fif"
            raw.save(outfile, overwrite=True)

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
        print(f"\nStep 1 finished with errors in: {', '.join(failed)}")
    else:
        print("\nStep 1 complete.")
