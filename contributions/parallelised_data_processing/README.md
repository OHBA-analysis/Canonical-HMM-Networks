# Parallelised Pipeline for Elekta MEG

This directory contains Python scripts that run the canonical HMM pipeline on multiple sessions in parallel using `multiprocessing`.

## Input Data

The scripts expect data in BIDS format:
```
BIDS/
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── meg/
│       └── sub-01_task-rest.fif
├── sub-02/
│   ├── ...
```

Output is written to `BIDS/derivatives/`.

## Scripts

Run the scripts **in order**. Each script processes all sessions in parallel.

| Script | Step | Description |
|--------|------|-------------|
| `1_preproc.py` | Preprocessing | Downsample (250 Hz), bandpass filter (1-45 Hz), notch filter (50/100 Hz), bad segment and bad channel detection |
| `2_surfaces.py` | Surface Extraction | Extract inner skull, outer skull and scalp surfaces from structural MRI using FSL BET |
| `3_coreg.py` | Coregistration | Coregister MEG to MRI using Polhemus headshape points |
| `4_source_recon_and_parc.py` | Forward Model, Source Reconstruction and Parcellation | Compute forward model (8 mm dipole grid), LCMV beamformer, parcellate voxel data, apply symmetric orthogonalisation |

## Usage

These scripts can be copied and run from anywhere on your system — they do not need to live inside the `Canonical-HMM-Networks` repository. The `project_dir` variable in each script tells it where to find the repository's `modules` and `models` directories.

```bash
conda activate osld

python 1_preproc.py
python 2_surfaces.py
python 3_coreg.py
python 4_source_recon_and_parc.py
```

## Configuration

Each script has a **Config** block at the top. Edit these variables before running:

```python
# --------- Config ---------
project_dir = Path("/path/to/Canonical-HMM-Networks")
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
log_dir = Path("logs/1_preproc")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
n_workers = 4
# ---------------------------
```

- `project_dir` — Path to the cloned `Canonical-HMM-Networks` repository. This is used to find the `modules` and `models` directories. The scripts themselves can live anywhere.
- `input_dir` — Path to your BIDS directory containing the raw data.
- `output_dir` — Path to the output directory for derivatives.
- `log_dir` — Directory for per-session log files.
- `sessions` — Dictionary of sessions to process. Each key is a session ID used for naming output files and logs. Each value contains the `subject` (BIDS subject directory) and `file` (MEG filename).
- `n_workers` — Number of sessions to process in parallel.

Some scripts have additional settings (e.g. `plots_dir`, `parcellation_file`, `gridstep`). See the Config block in each script for details.

## Output Structure

```
BIDS/derivatives/
├── preprocessed/
│   ├── sub-01_task-rest_preproc-raw.fif
│   └── ...
├── anat_surfaces/
│   ├── sub-01/
│   └── ...
└── osl/
    ├── sub-01_task-rest/
    │   ├── bem/
    │   ├── coreg/
    │   │   └── model-fwd.fif
    │   ├── src/
    │   │   └── filters-lcmv.h5
    │   └── lcmv-parc-raw.fif
    └── ...

plots/
├── sub-01_task-rest/
│   ├── 1_sum_square.png
│   ├── 1_sum_square_exclude_bads.png
│   ├── 1_channel_stds.png
│   ├── 2_inskull.png
│   ├── 2_outskin.png
│   ├── 2_outskull.png
│   ├── 3_coreg.html
│   └── 4_psd_topo.png
├── sub-02_task-rest/
│   └── ...
```

## Logging

Full verbose output (from MNE, osl-dynamics, etc.) is saved to per-session log files in the `log_dir` directory, e.g. `logs/1_preproc/sub-01_task-rest.log`.

## Notes

- Steps must be run sequentially (each depends on the output of the previous step), but within each step all sessions are processed in parallel.
- If you do not have a structural MRI for a subject, set `use_mni152 = True` and `allow_mri_scaling = True` in `3_coreg.py` (and `4_source_recon_and_parc.py`) to use the standard MNI152 brain. You can skip `2_surfaces.py` in this case.
- Set `n_workers` based on the number of CPU cores available. For memory-intensive steps (source reconstruction, parcellation), you may need to reduce this.
