from dipy.align.imwarp import get_direction_and_spacings

from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass
import time, os
import numpy as np
import pymp
import nibabel as nib
from dipy.align import sub_processes
from dipy.denoise.fsl_bet import fsl_bet_mask

def get_angles_list (step = 45):
    angles_list = []
    for x in range(-180, 180, step):
        for y in range(-180, 180, step):
            for z in range(-180, 180, step):
                mx = x / 180 * np.pi
                my = y / 180 * np.pi
                mz = z / 180 * np.pi
                angles_list.append([mx, my, mz])
    return angles_list

def register_images (target_arr, target_affine,
                     moving_arr, moving_affine,
                     phase,
                     lim_arr=None,
                     registration_type='quadratic',
                     initialize=False,
                     optimizer_setting=False):
    fixed_image, moving_image = sub_processes.set_images_in_scale(lim_arr,
                                                                  target_arr,
                                                                  moving_arr)

    sz = np.array(fixed_image.shape)
    moving_sz = np.array(moving_image.shape)

    dim = len(fixed_image.shape)

    orig_fixed_grid2world = target_affine
    orig_moving_grid2world = moving_affine

    mid_index = (sz - 1) / 2.

    # Physical Coordinate of the center index = 0,0,0
    new_orig_temp = - orig_fixed_grid2world[0:3, 0:3].dot(mid_index)
    fixed_grid2world = orig_fixed_grid2world.copy()
    fixed_grid2world[0:3, 3] = new_orig_temp

    new_orig_temp = - orig_moving_grid2world[0:3, 0:3].dot(mid_index)
    moving_grid2world = orig_moving_grid2world.copy()
    moving_grid2world[0:3, 3] = new_orig_temp

    _, resolution = get_direction_and_spacings(fixed_grid2world, dim)

    transformRegister = QuadraticRegistration(phase, registration_type=registration_type)
    OptimizerFlags = transformRegister.optimization_flags
    initializeTransform = QuadraticMap(phase)

    if initialize and (np.sum(OptimizerFlags[0:3]) != 0):
        initializeTransform = transform_centers_of_mass(fixed_image, moving_image, phase,
                                                        static_grid2world=fixed_grid2world,
                                                        moving_grid2world=moving_grid2world)

    grad_scale = sub_processes.get_gradients_params(resolution, sz)

    transformRegister = QuadraticRegistration(phase)

    transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
    finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                                phase=phase,
                                                static_grid2world=fixed_grid2world,
                                                moving_grid2world=moving_grid2world,
                                                grad_params=grad_scale)
    finalparams = finalTransform.get_QuadraticParams()

    finalTransform = QuadraticMap(phase, finalparams, fixed_image.shape, fixed_grid2world,
                                  moving_image.shape, moving_grid2world)
    image_transform = finalTransform.transform(image = moving_arr,QuadraticParams=finalparams)
    return finalparams,image_transform




def test ():
    # b0_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0.nii"
    # moving_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/100408_LR_proc.nii"
    # mask_target_image = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test2/process/temp_b0_mask_mask.nii"
    # b0_target = nib.load(b0_target_image)
    # moving_image = nib.load(moving_image)
    # mask_target = nib.load(mask_target_image)
    # b0_arr = b0_target.get_data()
    # mask_arr = mask_target.get_data()
    image_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test4/cont_21_0_AP_b1100_proc.nii"

    log_folder = os.path.join(os.path.dirname(image_fn), 'log_py')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    b0_id = 0

    image = nib.load(image_fn)
    moving_image = image.get_data()

    b0_arr = moving_image[...,b0_id]
    phase = 'vertical'
    moving_image_shr = pymp.shared.array((moving_image.shape), dtype = np.float32)
    moving_image_shr[:] = image.get_data()
    mask_arr = np.ones_like(b0_arr)

    # save temp b0s
    b0_image = nib.Nifti1Image(b0_arr, image.affine)
    b0_image_fn = os.path.join(os.path.dirname(image_fn), "temp_b0.nii")      # temp_b0.nii
    b0_mask_fn  = b0_image_fn.split(".nii")[0]+ "mask.nii"                     # temp_b0mask.nii
    b0_mask_bi_fn = b0_mask_fn.split(".nii")[0]+ "_mask.nii"                  # temo_b0mask_mask.nii

    nib.save(b0_image, b0_image_fn)
    fsl_bet_mask(b0_image_fn,
                 b0_image_fn.split(".nii")[0] + "mask.nii",
                 )

    lim_arr = pymp.shared.array((4, moving_image.shape[-1]), dtype=np.float32)

    transformation = pymp.shared.array((moving_image.shape[-1],21), dtype=np.float64)
    start_time = time.time()

    #with pymp.Parallel() as p:
    #    for index in p.range(1, moving_image.shape[-1]):
    #        curr_vol = moving_image_shr[:, :, :, index]
    #        lim_arr[:, index] = sub_processes.choose_range(b0_arr,
    #                                                       curr_vol,
    #                                                       mask_arr)

    b0_binary_mask = nib.load(b0_mask_bi_fn)
    b0_mask_mask = b0_binary_mask.get_data()

    b0_img_target, b0_mask_img = sub_processes.dmc_make_target(b0_image_fn, b0_mask_mask)

    # with pymp.Parallel() as p:
    # for index in p.range(1, moving_image.shape[-1]):
    for index in range(1, moving_image.shape[-1]):
        curr_vol = moving_image_shr[:, :, :, index]

        lim_arr[:, index] = sub_processes.choose_range(b0_arr,
                                                       curr_vol,
                                                       b0_mask_img)

        transformation[index,:],moving_image_shr[:,:,:,index] =  register_images(b0_img_target, b0_image.affine,
                             curr_vol, image.affine,
                             phase,
                            lim_arr=lim_arr[:,index],
                            registration_type='quadratic',
                            initialize=True,
                            optimizer_setting=False)
    np.savetxt(os.path.join(log_folder, 'transformations_test_2.txt'.format(index)), transformation)

    print("Time cost {}", time.time() - start_time)

    image_out = nib.Nifti1Image(moving_image_shr, image.affine)
    image_out_fn = image_fn.split(".nii")[0] + "_image_eddy_test_python_2.nii"
    nib.save(image_out, image_out_fn)

test()