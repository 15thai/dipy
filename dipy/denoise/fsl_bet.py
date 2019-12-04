from nipype.interfaces import fsl
import os
import nibabel as nib
import sys
import numpy as np

# Checking if environmental variable is correct
os.environ['PATH'] += os.pathsep + '/raid1b/STBBapps/fsl6/fsl/bin'
print(os.getenv('FSLDIR'))
fsl.FSLCommand.set_default_output_type('NIFTI')
print(os.getenv('PATH'))

def fsl_bet_mask(input_image_fn, output_image_fn = None, binary_mask = True):
    myb = fsl.BET()
    myb.inputs.in_file = input_image_fn
    myb.inputs.frac = 0.3

    if binary_mask:
        myb.inputs.mask = True
    if output_image_fn is None:
        output_image_fn = input_image_fn.split(".nii")[0] + "_fsl_mask.nii"
    myb.inputs.out_file = output_image_fn
    print(myb.cmdline)
    myb.run()

    return output_image_fn


def fsl_mask(b0_arr,
             b0_affine = None,
             save_to_dir =  None,
             b0_fn = "temp_b0.nii",
             frac = 0.3,
             binary_mask = True):
    if b0_affine is None:
        b0_affine = np.eye(4)
    if save_to_dir is None:
        save_to_dir = '.'
    b0_image = nib.Nifti1Image(b0_arr, b0_affine)
    b0_image_fn = os.path.join(save_to_dir, b0_fn)  # temp_b0.nii
    # save to file
    nib.save(b0_image, b0_image_fn)

    b0_mask_fn = b0_image_fn.strip(".nii") + "fslmask.nii"  # temp_b0mask.nii
    b0_mask_bi_fn = b0_mask_fn.strip(".nii") + '_mask.nii'  # temo_b0mask_mask.nii
    myb = fsl.BET()
    myb.inputs.in_file = b0_image_fn
    myb.inputs.out_file = b0_mask_fn
    myb.inputs.frac = frac
    if binary_mask:
        myb.inputs.mask = True
    print(myb.cmdline)

    myb.run()
    b0_mask_image = nib.load(b0_mask_fn)
    b0_mask_mask_image = nib.load(b0_mask_bi_fn)
    return b0_mask_image, b0_mask_mask_image