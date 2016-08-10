__author__ = 'Jonny'

import hungarian.hungarian_algorithm as hm
from procrustes.base import Procrustes
import numpy as np


class PermutationProcrustes(Procrustes):
    """
    This method deals with the Permutation Procrustes problem
    """

    def __init__(self, array_a, array_b, translate=False, scale=False, preserve_symmetry=False, hide_padding=True,
                 translate_symmetrically=False):
        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.translate_symmetrically=translate_symmetrically
        Procrustes.__init__(self, array_a, array_b, translate=self.translate, scale=self.scale,
                            preserve_symmetry=self.preserve_symmetry, hide_padding=self.hide_padding,
                            translate_symmetrically=self.translate_symmetrically)

    def calculate(self):
        """
         Calculates the optimum permutation transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        perm_optimum, array_transformed, total_potential error
        perm_optimum = the optimum permutation transformation matrix satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by perm_optimum
        total_potential = The total 'profit', i.e. the trace of the transformed input array
        error = the error as described by the single-sided procrustes problem
        """
        array_a = self.array_a
        array_b = self.array_b

        # Define the profit array and applying the hungarian algorithm
        profit_array = np.dot(array_a.T, array_b)
        hungary = hm.Hungarian(profit_array, is_profit_matrix=True)
        hungary.calculate()

        # Obtain the optimum permutation transformation
        perm_hungarian = hungary.get_results()
        # perm_hungarian is not in a numpy 2d array format... Create perm_optimum, a 2d numpy array
        permutation_optimum = np.zeros(profit_array.shape)
        # convert hungarian output into array form, and store in perm_optimum
        for k in range(len(perm_hungarian)):
            # Get the indices of the 1's in the permutation array
            i, j = perm_hungarian[k]
            # Create the permutation array, perm_optimum
            permutation_optimum[i, j] = 1

        # Calculate the total potential (trace)
        total_potential = hungary.get_total_potential()

        # Map array_a to array_b with the optimum permutation array
        self.map_a_b(permutation_optimum,
                     preserve_symmetry=self.preserve_symmetry, translate_symmetrically=self.translate_symmetrically)

        # Compute the real error
        real_error = self.single_sided_procrustes_error(self.a_transformed, self.b_in)

        return permutation_optimum, self.a_transformed, total_potential, real_error, self.transformation
