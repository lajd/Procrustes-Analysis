__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np


class RotationalOrthogonalProcrustes(Procrustes):

    """
    This method deals with the Rotational-Orthogonal Procrustes Problem
    Constrain the transformation matrix U to be pure rotational
    """

    def __init__(self, array_a, array_b, translate=False, scale=False, hide_padding=True):
        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        Procrustes.__init__(self, array_a, array_b, translate=self.translate, scale=self.scale,
                            hide_padding=self.hide_padding)

    def calculate(self):

        """
        Calculates the optimum rotational-orthogonal transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        r, array_transformed, error
        r = the optimum rotation transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by r
        error = the error as described by the single-sided procrustes problem
        """

        array_a = self.array_a
        array_b = self.array_b

        # Compute SVD of the product matrix
        product_matrix = np.dot(array_a.T, array_b)
        u, s, v_trans = self.singular_value_decomposition(product_matrix)

        # Constrain transformation matrix to be a pure rotation matrix by replacing the least
        # significant singular value with sgn(|U*V^t|). The rest of the diagonal entries are unity.
        replace_singular_value = np.sign(np.linalg.det(np.dot(u, v_trans)))
        n = array_a.shape[1]
        # All singular values should be unity, except the smallest, which is replaced.
        s = np.eye(n)
        # Replace the least significant singular value to force a rotation. If already pure rotation,
        # replace the value anyways.
        s[n-1, n-1] = replace_singular_value

        # Calculate the optimum rotation matrix r
        rotation_optimum = np.dot(np.dot(u, s), v_trans)

        # Map array_a to array_b with the optimum rotation array
        self.map_a_b(rotation_optimum)

        # Compute the real error
        real_error = self.single_sided_procrustes_error(self.a_transformed, self.b_in)

        return rotation_optimum, self.a_transformed, real_error, self.transformation

