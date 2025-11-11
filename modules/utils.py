"""Utility functions and classes."""

import os
import mne
import scipy
import shutil
import numpy as np
import pandas as pd
import nibabel as nib


class OSLFilenames:
    def __init__(self, outdir, id, preproc_file, surfaces_dir, pos_file=None):
        self.outdir = outdir
        self.id = id

        self.preproc_file = preproc_file

        self.surfaces_dir = surfaces_dir
        self.surfaces = SurfaceFilenames(surfaces_dir)

        self.bem_dir = f"{outdir}/{id}/bem"
        os.makedirs(self.bem_dir, exist_ok=True)

        self.coreg_dir = f"{outdir}/{id}/coreg"
        self.coreg = CoregFilenames(self.coreg_dir)
        self.fwd_model = f"{self.coreg_dir}/model-fwd.fif"
        self.pos_file = pos_file  # only needed for CTF data

        self.src_dir = f"{outdir}/{id}/src"
        os.makedirs(self.src_dir, exist_ok=True)
        self.filters = f"{self.src_dir}/filters-lcmv.h5"

    def __str__(self):
        lines = [
            f"OSLFilenames for {self.id}:",
            f"  Output directory:  {self.outdir}",
            f"  Preprocessed file: {self.preproc_file}",
            f"  Surfaces directory: {self.surfaces_dir}",
            f"  BEM directory:     {self.bem_dir}",
            f"  Coreg directory:   {self.coreg_dir}",
            f"    └─ Forward model: {self.fwd_model}",
            f"  Source directory:  {self.src_dir}",
            f"    └─ lcmv filters:  {self.filters}",
        ]
        if self.pos_file is not None:
            lines += [
                f"  pos file:  {self.pos_file}",
            ]
        return "\n".join(lines)

    def __repr__(self):
        # For convenience when inspecting in interactive mode
        return f"<OSLFilenames id='{self.id}' outdir='{self.outdir}'>"


class SurfaceFilenames:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.fsl_dir = os.environ["FSLDIR"]

        # Nifti files
        self.smri_file = f"{root}/smri.nii.gz"
        self.std_brain = f"{self.fsl_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz"

        # Transformations
        self.mni2mri_flirt_xform_file = f"{root}/mni2mri_flirt_xform.txt"
        self.mni_mri_t_file = f"{root}/mni_mri-trans.fif"

        # BET mesh / surfaces
        self.bet_outskin_mesh_vtk_file = f"{root}/outskin_mesh.vtk"
        self.bet_inskull_mesh_vtk_file = f"{root}/inskull_mesh.vtk"
        self.bet_outskull_mesh_vtk_file = f"{root}/outskull_mesh.vtk"
        self.bet_outskin_mesh_file = f"{root}/outskin_mesh.nii.gz"
        self.bet_inskull_mesh_file = f"{root}/inskull_mesh.nii.gz"
        self.bet_outskull_mesh_file = f"{root}/outskull_mesh.nii.gz"


class CoregFilenames:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

        # Nifti files
        self.smri_file = f"{root}/scaled_smri.nii.gz"

        # Fif files
        self.info_fif_file = f"{root}/info-raw.fif"
        self.head_scaledmri_t_file = f"{root}/head_scaledmri-trans.fif"
        self.head_mri_t_file = f"{root}/head_mri-trans.fif"
        self.ctf_head_mri_t_file = f"{root}/ctf_head_mri-trans.fif"
        self.mrivoxel_scaledmri_t_file = f"{root}/mrivoxel_scaledmri_t_file-trans.fif"

        # Fiducials / polhemus
        self.mni_nasion_mni_file = f"{root}/mni_nasion.txt"
        self.mni_rpa_mni_file = f"{root}/mni_rpa.txt"
        self.mni_lpa_mni_file = f"{root}/mni_lpa.txt"
        self.smri_nasion_file = f"{root}/smri_nasion.txt"
        self.smri_rpa_file = f"{root}/smri_rpa.txt"
        self.smri_lpa_file = f"{root}/smri_lpa.txt"
        self.polhemus_nasion_file = f"{root}/polhemus_nasion.txt"
        self.polhemus_rpa_file = f"{root}/polhemus_rpa.txt"
        self.polhemus_lpa_file = f"{root}/polhemus_lpa.txt"
        self.polhemus_headshape_file = f"{root}/polhemus_headshape.txt"

        # Freesurfer mesh in native space
        # - these are the ones shown in the surface plot
        # - these are also used by MNE forward modelling
        self.bet_outskin_surf_file = f"{root}/scaled_outskin.surf"
        self.bet_inskull_surf_file = f"{root}/scaled_inskull.surf"
        self.bet_outskull_surf_file = f"{root}/scaled_outskull.surf"

        # BET mesh / surfaces in native space
        self.bet_outskin_mesh_vtk_file = f"{root}/scaled_outskin_mesh.vtk"
        self.bet_inskull_mesh_vtk_file = f"{root}/scaled_inskull_mesh.vtk"
        self.bet_outskull_mesh_vtk_file = f"{root}/scaled_outskull_mesh.vtk"
        self.bet_outskin_mesh_file = f"{root}/scaled_outskin_mesh.nii.gz"
        self.bet_inskull_mesh_file = f"{root}/scaled_inskull_mesh.nii.gz"
        self.bet_outskull_mesh_file = f"{root}/scaled_outskull_mesh.nii.gz"


def system_call(cmd, verbose=True):
    if verbose:
        print(cmd)
    os.system(cmd)


def convert_notts_opm_files_to_fif(
    mat_file, smri_file, tsv_file, out_fif_file, out_fixed_smri_file
):
    """Convert Nottingham OPM data from matlab file to fif file.

    Parameters
    ----------
    mat_file : str
        The matlab file containing the OPM data.
    smri_file : str
        The structural MRI file.
    tsv_file : str
        The tsv file containing the sensor locations and orientations.
    out_fif_file : str
        The output fif file.
    out_fixed_smri_file : str
        The output structural MRI file with corrected sform.

    Notes
    -----
    The matlab file is assumed to contain a variable called 'data' which is
    a matrix of size nSamples x nChannels.
    The matlab file is assumed to contain a variable called 'fs' which is
    the sampling frequency.
    The tsv file is assumed to contain a header row, and the following columns:
    name, type, bad, x, y, z, qx, qy, qz
    The x,y,z columns are the sensor locations in metres.
    The qx,qy,qz columns are the sensor orientations in metres.
    """
    # correct sform for smri
    sform_std_fixed = correct_notts_mri(smri_file, out_fixed_smri_file)

    # Note that later in this function, we will also apply this sform to
    # the sensor coordinates and orientations.
    # This is because, with the OPM Notts data, coregistration on the sensor
    # coordinates has already been carried out, and so the sensor coordinates
    # need to stay matching the coordinates used in the MRI

    # Convert passed in OPM matlab file and tsv file to fif file

    # Load in chan info
    chan_info = pd.read_csv(tsv_file, header=None, skiprows=[0], sep="\t")

    sensor_names = chan_info.iloc[:, 0].to_numpy().T
    sensor_locs = chan_info.iloc[:, 4:7].to_numpy().T  # in metres
    sensor_oris = chan_info.iloc[:, 7:10].to_numpy().T
    sensor_bads = chan_info.iloc[:, 3].to_numpy().T

    # import pdb; pdb.set_trace()

    # Need to undo orginal sform on sensor locs and oris and then apply new sform
    smri = nib.load(smri_file)
    overall_xform = sform_std_fixed @ np.linalg.pinv(smri.header.get_sform())

    # This trans isn't really mri to head, it is mri to "mri_fixed",
    # but mri_fixed is not available as an option
    overall_xform_trans = mne.transforms.Transform("mri", "head", overall_xform)

    # Note sensor_locs are in metres, overall_xform_trans is in mm
    sensor_locs = (
        mne.transforms.apply_trans(overall_xform_trans, sensor_locs.T * 1000).T / 1000
    )
    sensor_oris = (
        mne.transforms.apply_trans(overall_xform_trans, sensor_oris.T * 1000).T / 1000
    )

    # -------------------------------------------
    # Create fif file from mat file and chan_info
    # -------------------------------------------

    Fs = scipy.io.loadmat(mat_file)["fs"][0, 0]  # Hz

    # see https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html

    info = mne.create_info(ch_names=sensor_names.tolist(), ch_types="mag", sfreq=Fs)

    # get names of bad channels
    select_indices = list(np.where(sensor_bads == "bad")[0])
    info["bads"] = sensor_names[select_indices].tolist()

    dig_montage = mne.channels.make_dig_montage(hsp=sensor_locs.T)
    info.set_montage(dig_montage)

    # MEG (device): dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # We assume that device space, head space and mri space are all the same
    # space and that the sensor locations and fiducials (if there are any) are
    # already in that space.
    # This means that dev_head_t is identity
    # This means that dev_mri_t is identity

    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.identity(4))

    # --------------------------------
    # Set sensor locs and oris in info
    # --------------------------------

    def _cartesian_to_affine(loc, ori):

        # The orientation, ori, defines an orientation as a 3D cartesian vector
        # (in x,y,z) taken from the origin.
        # The location, loc, is a 3D cartesian vector (in x,y,z) taken from the
        # origin.

        # To convert the cartesian orientation vector to an affine rotation
        # matrix, we first convert the cartesian coordinates into spherical
        # coords.
        # See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

        # r = 1
        theta = np.arccos(ori[2] / np.sqrt(np.sum(np.square(ori))))
        if ori[0] > 0:
            phi = np.arctan(ori[1] / ori[0])
        elif ori[0] < 0 and ori[1] >= 0:
            phi = np.arctan(ori[1] / ori[0]) + np.pi
        elif ori[0] < 0 and ori[1] < 0:
            phi = np.arctan(ori[1] / ori[0]) - np.pi
        elif ori[0] == 0 and ori[1] > 0:
            phi = np.pi / 2
        elif ori[0] == 0 and ori[1] < 0:
            phi = -np.pi / 2

        # We next convert the spherical coords into an affine rotation matrix.
        #
        # See: "Rotation matrix from axis and angle" at
        # https://en.wikipedia.org/wiki/Rotation_matrix
        #
        # Plus see:
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
        # We will use the Physics convention for spherical coordinates
        #
        # MNE assumes that affine transform to determine sensor
        # location/orientation is applied to a unit vector along the z-axis
        #
        # First we do a rotation to the x-axis
        # i.e. rotation pi/2 around y-axis
        # i.e. axis of rotation (ux,uy,uz) = (0,1,0)
        deg = np.pi / 2
        Rdeg = np.array(
            [
                [np.cos(deg), 0, np.sin(deg), 0],
                [0, 1, 0, 0],
                [-np.sin(deg), 0, np.cos(deg), 0],
                [0, 0, 0, 1],
            ]
        )

        # Second we then do a rotation of phi around the z-axis
        # i.e. axis of rotation (ux,uy,uz) = (0,0,1)
        phin = phi
        Rphi = np.array(
            [
                [np.cos(phin), -np.sin(phin), 0, 0],
                [np.sin(phin), np.cos(phin), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Third we  do a rotation of -(pi/2-theta) around the
        # axis of rotation (ux,uy,uz) = (-np.sin(phi), np.cos(phi), 0)
        ux = -np.sin(phi)
        uy = np.cos(phi)
        thetan = -(np.pi / 2.0 - theta)
        Rtheta = np.array(
            [
                [
                    ux * ux * (1 - np.cos(thetan)) + np.cos(thetan),
                    ux * uy * (1 - np.cos(thetan)),
                    uy * np.sin(thetan),
                    0,
                ],
                [
                    ux * uy * (1 - np.cos(thetan)),
                    uy * uy * (1 - np.cos(thetan)) + np.cos(thetan),
                    -ux * np.sin(thetan),
                    0,
                ],
                [-uy * np.sin(thetan), ux * np.sin(thetan), np.cos(thetan), 0],
                [0, 0, 0, 1],
            ]
        )

        # We also want to combine the rotation matrix with the translation.
        # So, finally we do the translation
        translate = np.array(
            [[1, 0, 0, loc[0]], [0, 1, 0, loc[1]], [0, 0, 1, loc[2]], [0, 0, 0, 1]]
        )

        affine = translate @ Rtheta @ Rphi @ Rdeg

        return affine

    # test:
    # affine_from_loc_ori([0, 0, 0], [0, 1, 1]/np.sqrt(2))@[1, 0, 0, 1]

    for cc in range(len(info["chs"])):
        affine_mat = _cartesian_to_affine(sensor_locs[:, cc], sensor_oris[:, cc])
        info["chs"][cc]["loc"] = mne._fiff.tag._coil_trans_to_loc(affine_mat)
        info["chs"][cc][
            "coil_type"
        ] = mne.io.constants.FIFF.FIFFV_COIL_POINT_MAGNETOMETER

    # Finally, put data and info together and save to out_fif_file
    data = scipy.io.loadmat(mat_file)["data"].T * 1e-15  # fT
    raw = mne.io.RawArray(data, info)
    raw.save(out_fif_file, overwrite=True)


def correct_notts_mri(smri_file, smri_fixed_file):
    """Correct the sform in the structural MRI file.

    Parameters
    ----------
    smri_file : str
        The structural MRI file.
    smri_fixed_file : str
        The output structural MRI file with corrected sform.

    Returns
    -------
    sform_std : ndarray
        The new sform.

    Notes
    -----
    The sform is corrected so that it is in standard orientation.
    """
    shutil.copyfile(smri_file, smri_fixed_file)

    smri = nib.load(smri_fixed_file)
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)

    # sform_std[0, 0:4] = [-1, 0, 0, 128]
    # sform_std[1, 0:4] = [0, 1, 0, -128]
    # sform_std[2, 0:4] = [0, 0, 1, -90]

    sform_std[0, 0:4] = [1, 0, 0, -90]
    sform_std[1, 0:4] = [0, -1, 0, 126]
    sform_std[2, 0:4] = [0, 0, -1, 72]

    system_call(
        "fslorient -setsform {} {}".format(
            " ".join(map(str, sform_std.flatten())), smri_fixed_file
        )
    )

    return sform_std
