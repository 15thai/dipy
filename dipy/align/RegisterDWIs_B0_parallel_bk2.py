from dipy.align.imwarp import get_direction_and_spacings

from dipy.align.ImageQuadraticMap import QuadraticMap, QuadraticRegistration, transform_centers_of_mass
import time, os
import numpy as np
import pymp
import nibabel as nib
from dipy.align import sub_processes
from dipy.denoise.fsl_bet import fsl_mask
# For Debugging purpose.
import matplotlib.pyplot as plt

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
                     ):
    fixed_image, moving_image = sub_processes.set_images_in_scale(lim_arr,
                                                                  target_arr,
                                                                  moving_arr)

    sz = np.array(fixed_image.shape)
    moving_sz = np.array(moving_image.shape)

    dim = len(fixed_image.shape)

    orig_fixed_grid2world = target_affine
    fixed_grid2world = target_affine
    orig_moving_grid2world = moving_affine

   # mid_index = (sz - 1) / 2.
   # # Physical Coordinate of the center index = 0,0,0
   # new_orig_temp = - orig_fixed_grid2world[0:3, 0:3].dot(mid_index)
   # fixed_grid2world = orig_fixed_grid2world.copy()
   # fixed_grid2world[0:3, 3] = new_orig_temp



    mid_index_moving = (moving_sz - 1) / 2.
    new_orig_temp = - orig_moving_grid2world[0:3, 0:3].dot(mid_index_moving)
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
    grad_scale2 = grad_scale.copy()
    grad_scale2[3:6] = grad_scale2[3:6] * 2

    transformRegister.factors = [4,2,1]
    transformRegister.sigmas = [1.,0.25,0]
    transformRegister.levels = 3



    flag2 = transformRegister.optimization_flags
    transformRegister.set_optimizationflags(flag2)

    transformRegister.initial_QuadraticParams = initializeTransform.get_QuadraticParams()
    finalTransform = transformRegister.optimize(fixed_image, moving_image,
                                                phase=phase,
                                                static_grid2world=fixed_grid2world,
                                                moving_grid2world=moving_grid2world,
                                                grad_params=grad_scale)
    finalparams = finalTransform.get_QuadraticParams()

    image_transform = finalTransform.transform(image = moving_arr,QuadraticParams=finalparams)
    return finalparams,image_transform




def test ():
    image_fn = "/qmi_home/anht/Desktop/DIFFPREP_test_data/test4/cont_21_0_AP_b1100_proc.nii"

    log_folder = os.path.join(os.path.dirname(image_fn), 'log_py')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    image = nib.load(image_fn)
    moving_image = image.get_data()

    # Getting B0_volume
    b0_id = 0
    b0_arr = moving_image[...,b0_id]
    phase = 'vertical'


    b0_masked_image, b0_masked_mask_image = fsl_mask(b0_arr,
                                                     image.affine,
                                                     save_to_dir= os.path.dirname(image_fn))


    lim_arr = pymp.shared.array((4, moving_image.shape[-1]), dtype=np.float32)

    transformation = pymp.shared.array((moving_image.shape[-1],21), dtype=np.float64)
    start_time = time.time()

    #with pymp.Parallel() as p:
    #    for index in p.range(1, moving_image.shape[-1]):
    #        curr_vol = moving_image_shr[:, :, :, index]
    #        lim_arr[:, index] = sub_processes.choose_range(b0_arr,
    #                                                       curr_vol,
    #                                                       mask_arr)

    b0_image = nib.Nifti1Image(b0_arr, image.affine)
    b0_img_target_dmc, b0_img_target_affine, b0_mask_img = sub_processes.dmc_make_target(b0_image, b0_masked_image)

    # moving_image_shr = pymp.shared.array((moving_image.shape), dtype = np.float32)
    # moving_image_shr[:] = image.get_data()
    moving_image_shr = image.get_data()


    # with pymp.Parallel() as p:
    # for index in p.range(1, moving_image.shape[-1]):
    for index in range(1, moving_image.shape[-1]):
        curr_vol = moving_image_shr[:, :, :, index].copy()

        lim_arr[:, index] = sub_processes.choose_range(b0_arr,
                                                       curr_vol,
                                                       b0_masked_mask_image.get_data())


        transformation[index,:],moving_image_shr[:,:,:,index] =  register_images(b0_img_target_dmc,
                                                                                 b0_img_target_affine,
                             curr_vol, image.affine,
                             phase,
                            lim_arr=lim_arr[:,index],
                            registration_type='quadratic',
                            initialize=True,
                           )
        np.savetxt(os.path.join(log_folder, 'transformations_test_{}_init_op.txt'.format(index)), transformation)

    print("Time cost {}", time.time() - start_time)

    image_out = nib.Nifti1Image(moving_image_shr, image.affine)
    image_out_fn = image_fn.split(".nii")[0] + "_image_eddy_test_python_init_op.nii"
    nib.save(image_out, image_out_fn)

test()