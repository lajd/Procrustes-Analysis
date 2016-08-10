__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np


class SymmetricProcrustes(Procrustes):
    """
    This method deals with the symmetric Procrustes problem

    We require symmetric input arrays for this problem

    """

    def __init__(self, array_a, array_b, translate=False, scale=False, hide_padding=True):
        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        Procrustes.__init__(self, array_a, array_b, translate=self.translate, scale=self.scale,
                            hide_padding=self.hide_padding)

    def calculate(self):
        """
        Calculates the optimum symmetric transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        x, array_transformed, error
        x = the optimum symmetric transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by x
        error = the error as described by the single-sided procrustes problem
        """
        array_a = self.array_a
        array_b = self.array_b

        m, n = self.array_a.shape

        # Compute SVD of array_a
        u, s, v_trans = self.singular_value_decomposition(array_a)

        # Add zeros to the eigenvalue array, s, so that the total length of s is = n
        s_concatenate = np.zeros(n - len(s))
        np.concatenate((s, s_concatenate))

        # Define the array, c
        c = np.dot(np.dot(u.T, array_b), v_trans.T)

        # Create the intermediate array, y
        y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if s[i]**2 + s[j]**2 == 0:
                    y[i, j] = 0
                else:
                    y[i, j] = (s[i]*c[i, j] + s[j]*c[j, i]) / (s[i]**2 + s[j]**2)

        # solve for the optimum symmetric transformation array
        symmetric_optimum = np.dot(np.dot(v_trans.T, y), v_trans)

        # map array_a to array_b with the optimum symmetric array
        self.map_a_b(symmetric_optimum)

        # Compute the real error
        real_error = self.single_sided_procrustes_error(self.a_transformed, self.b_in)

        return symmetric_optimum, self.a_transformed, real_error, self.transformation


# Reference
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.4378&rep=rep1&type=pdf
