import numpy as np
import numpy.linalg as npl
import scipy.ndimage as ndimage
import math
from dipy.align.imwarp import get_direction_and_spacings
from dipy.align.parzenhist import (ParzenJointHistogram,
                                   compute_parzen_mi)
from dipy.align.scalespace import  IsotropicScaleSpace
from dipy.align.quadratictransform import quadratic_transform
from dipy.align import VerbosityLevels
from dipy.align.DIFFPREPOptimizer import DIFFPREPOptimizer

_interp_options = ['nearest', 'linear', 'quadratic']
_NQUADPARAMS = 21

class QuadraticInversionError(Exception):
    pass
class QuadraticInvalidValuesError(Exception):
    pass


class QuadraticMap(object):
    def __init__(self, phase, QuadraticParams, affine,
                 domain_grid_shape, domain_grid2world, codomain_grid_shape, codomain_grid2world):
