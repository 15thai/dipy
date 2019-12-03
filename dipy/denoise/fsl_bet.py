from nipype.interfaces import fsl
import os
import nibabel
import sys

# Checking if environmental variable is correct
os.environ['PATH'] += os.pathsep + '/raid1b/STBBapps/fsl6/fsl/bin'
print(fsl.Info().version())
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
#
# if __name__ == "__main__":
#     input_image = sys.argv[1]
#     if len(sys.argv) <3:
#         output_image = input_image.split('.nii')[0] + "fsl_bet.nii"
#     else:
#         output_image = sys.argv[2]
#     binary_mask = sys.argv[3]
#
#     fsl_bet_mask(input_image, output_image, binary_mask )