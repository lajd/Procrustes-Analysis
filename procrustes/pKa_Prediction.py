__author__ = 'Jonathan'

import numpy as np
import procrustes
from math import *
from scipy.stats import linregress
from pylab import plot
import matplotlib


full_matrices = ['BANO2.sum_LIDI_MATRIX.txt','BACN.sum_LIDI_MATRIX.txt','BACOCH3.sum_LIDI_MATRIX.txt',
                 'BACOH.sum_LIDI_MATRIX.txt','BACl.sum_LIDI_MATRIX.txt','BAF.sum_LIDI_MATRIX.txt',
                 'BA.sum_LIDI_MATRIX.txt', 'BANHCOCH3.sum_LIDI_MATRIX.txt','BACH3.sum_LIDI_MATRIX.txt',
                 'BAOCH3.sum_LIDI_MATRIX.txt','BAOH.sum_LIDI_MATRIX.txt','BANH2.sum_LIDI_MATRIX.txt',
                 'BANCH3CH3.sum_LIDI_MATRIX.txt', 'BANHCH3.sum_LIDI_MATRIX.txt']

pruned_matrices = ['BANO2.sum_PRUNED_MATRIX.txt', 'BACN.sum_PRUNED_MATRIX.txt', 'BACOCH3.sum_PRUNED_MATRIX.txt',
                   'BACOH.sum_PRUNED_MATRIX.txt', 'BACl.sum_PRUNED_MATRIX.txt', 'BAF.sum_PRUNED_MATRIX.txt',
                   'BA.sum_PRUNED_MATRIX.txt', 'BANHCOCH3.sum_PRUNED_MATRIX.txt', 'BACH3.sum_PRUNED_MATRIX.txt',
                   'BAOCH3.sum_PRUNED_MATRIX.txt', 'BAOH.sum_PRUNED_MATRIX.txt', 'BANH2.sum_PRUNED_MATRIX.txt',
                   'BANCH3CH3.sum_PRUNED_MATRIX.txt', 'BANHCH3.sum_PRUNED_MATRIX.txt']


error_list_full = [0.0, 40.940935171732548, 35.832071883153404, 42.22096571645919, 190.99846617533032,
                   73.854892238748462, 85.721223614804416, 56.079586743170893, 75.126081557217773, 30.592080893511401,
                   73.854892238748462, 108.02258397891291, 71.081345609757179, 56.10305260405655]

error_list_pruned = [0.0, 36.589698202134848, 36.634522607688666, 37.030868674524953, 113.9039561582056, 94.411572512972313,
                     142.12902454491004, 135.20798583349227, 98.180785029917075, 66.498186671286334, 171.505074686402,
                     145.3946412237442, 99.323572950360926, 77.700318211200013]

file_labels = [full_matrices, pruned_matrices]

# From left to right, populate with Full Matrices, Pruned Matrices, -COOH subgraph, and -OH subgraph
matrices_labels = [[], [], [], []]

double_perm_error = [error_list_full, error_list_pruned]
double_perm_correlation = []

double_perm_orthogonal_error = [error_list_full, error_list_pruned]
double_perm_orthogonal_correlation = []

orthogonal_error = [error_list_full, error_list_pruned]
orthogonal_correlation = []

rotational_orthogonal_error = [error_list_full, error_list_pruned]
rotational_orthogonal_correlation = []

# cd /Applications/Python 2.7/Projects/4.0/procrustes/procrustes_project/Txt.Format

# Populate matrices_labels
# First, do Full and Pruned matrices

for data_set in range(len(file_labels)):

    for k in range(len(file_labels[data_set])):
        # Make a matrix of string elements from the corresponding text file
        string_file = file_labels[data_set][k]
        fh = open(string_file)
        # Create a list of string matrix elements
        string_matrix_element_list = []
        for line in fh.readlines():
            string_element = [element for element in line.split()]
            string_matrix_element_list.append(string_element)
        fh.close()

        # Convert the matrix elements from string into float
        float_matrix = []
        for i in range(len(string_matrix_element_list)):
            string_element = string_matrix_element_list[i]
            float_element_list = []
            for j in range(len(string_element)):
                t = float(string_element[j])
                float_element_list.append(t)
            float_matrix.append(float_element_list)
        # Add the matrix obtained from this file
        matrices_labels[data_set].append(np.array(float_matrix))

# Populate matrices_labels with the subgraphs
for i in range(len(full_matrices)):
    full_matrix = matrices_labels[1][i]
    oh_submatrix = full_matrix[0:2, 0:2]
    cooh_submatrix = full_matrix[0:4, 0:4]
    matrices_labels[2].append(cooh_submatrix)
    matrices_labels[3].append(oh_submatrix)


#for data_set in range(len(matrices_labels)):
for data_set in range(2, 4):
    # Procrustes Analysis

    # Reference molecule for the test
    reference = matrices_labels[data_set][0]
    # This is the error (distance) list obtained from the representations of this set of molecules
    # Add an error list to the error_in_file list
    double_perm_error.append([])
    orthogonal_error.append([])
    double_perm_orthogonal_error.append([])
    rotational_orthogonal_error.append([])



    # Perform procrustes analysis on each of the molecules in the data set, with respect to the reference molecule
    for i in range(len(matrices_labels[data_set])):
        comparing_matrix = matrices_labels[data_set][i]
        if data_set is 0:
            two_sided_permutation = procrustes.procrustes.TwoSidedPermutationSingleTransformationProcrustes\
                (comparing_matrix, reference, hide_padding=False, translate_symmetrically=False)

        if data_set is 1:
            two_sided_permutation = procrustes.procrustes.TwoSidedPermutationSingleTransformationProcrustes\
                (comparing_matrix, reference, hide_padding=False, translate_symmetrically=False)
        else:
            two_sided_permutation = procrustes.procrustes.TwoSidedPermutationSingleTransformationProcrustes\
                (comparing_matrix, reference, hide_padding=False, translate_symmetrically=False)
        print 'data set {0}, at file label {1}'.format(data_set, i)

        # Double sided perm
        least_error_perm, least_error_array_transformed, real_error, transformation = two_sided_permutation.calculate()
        double_perm_error[data_set].append(real_error)

        # Double sided perm and orthogonal
        orthogonal = procrustes.procrustes.RotationalOrthogonalProcrustes(least_error_array_transformed, reference,
                                                                hide_padding=False, translate=False, scale=False)
        u_optimum, a_transformed, real_error, transformation = orthogonal.calculate()
        double_perm_orthogonal_error[data_set].append(real_error)

        # Orthogonal
        orthogonal = procrustes.procrustes.OrthogonalProcrustes(comparing_matrix, reference,
                                                                hide_padding=False, translate=True, scale=True)
        u_optimum, a_transformed, real_error, transformation = orthogonal.calculate()
        orthogonal_error[data_set].append(real_error)

        # Rotational Orthogonal
        roto = procrustes.procrustes.RotationalOrthogonalProcrustes(comparing_matrix, reference,
                                                                hide_padding=False, translate=True, scale=True)
        rotation_optimum, a_transformed, real_error, transformation = roto.calculate()
        rotational_orthogonal_error[data_set].append(real_error)

pka = [3.44, 3.55, 3.74, 3.77, 3.98, 4.14, 4.19, 4.30, 4.37, 4.47, 4.57, 4.82, 5.03, 5.04]


for i in range(4):
    double_perm_correlation.append([])
    orthogonal_correlation.append([])
    double_perm_orthogonal_correlation.append([])
    rotational_orthogonal_correlation.append([])

    slope, intercept, r_value, p_value, stderr = linregress(double_perm_error[i], pka)
    double_perm_correlation[i].append(r_value**2)
    slope, intercept, r_value, p_value, stderr = linregress(orthogonal_error[i], pka)
    orthogonal_correlation[i].append(r_value**2)
    slope, intercept, r_value, p_value, stderr = linregress(double_perm_orthogonal_error[i], pka)
    double_perm_orthogonal_correlation[i].append(r_value**2)
    slope, intercept, r_value, p_value, stderr = linregress(rotational_orthogonal_error[i], pka)
    rotational_orthogonal_correlation[i].append(r_value**2)







# rotation

b = matrices_labels[0][0]
a = matrices_labels[0][1]
two_sided_permutation = procrustes.procrustes.TwoSidedPermutationSingleTransformationProcrustes\
(a, b, hide_padding=False, translate_symmetrically=False)
least_error_perm, least_error_array_transformed, real_error, transformation = two_sided_permutation.calculate()

# Linear Regression of Obtained Data
pka = [3.44, 3.55, 3.74, 3.77, 3.98, 4.14, 4.19, 4.30, 4.37, 4.47, 4.57, 4.82, 5.03, 5.04]

error_list_full = [0.0, 40.940935171732548, 35.832071883153404, 42.22096571645919, 190.99846617533032,
                   73.854892238748462, 85.721223614804416, 56.079586743170893, 75.126081557217773, 30.592080893511401,
                   73.854892238748462, 108.02258397891291, 71.081345609757179, 56.10305260405655]

error_list_pruned = [0.0, 36.589698202134848, 36.634522607688666, 37.030868674524953, 113.9039561582056, 94.411572512972313,
                     142.12902454491004, 135.20798583349227, 98.180785029917075, 66.498186671286334, 171.505074686402,
                     145.3946412237442, 99.323572950360926, 77.700318211200013]

error_list_oh_sub = [0.0, 1.3783527765368293e-05, 0.00014067357170306262, 6.9776250318819537e-05,
                     0.00021115037453113149, 0.0002727688329754312, 0.00043956270758456365,
                     0.00037755709663410571, 0.00062170345295850657, 0.00084923317510632302,
                     0.00065661759772695915, 0.0012364019870860048, 0.0015908232630705958, 0.0015061532106047475]

error_list_cooh_sub = [0.0, 5.6059701583084747e-07, 9.8908338677075482e-06, 4.4659552512188247e-06,
                       1.128049430447712e-05, 1.3402122288930652e-05, 2.6439648410890758e-05, 1.8462635490097431e-05,
                       3.6192807120437876e-05, 4.3031892143096034e-05, 3.421898313625481e-05, 6.1367259374466929e-05,
                       8.1465048297124648e-05, 7.5165367626130834e-05]

paper_error = [[],[],[],[]]

counter = 0
for data_set in matrices_labels[2:4]:
    reference = data_set[0]
    for i in range(len(data_set)):
        comparing_matrix = data_set[i]
        difference = abs(reference - comparing_matrix)
        squared_difference = np.sum(difference**2)
        paper_error[counter].append(np.sqrt(squared_difference))
    counter += 1

slope, intercept, r_value, p_value, stderr = linregress(error_list_full, pka)
d_range = np.linspace(min(error_list_full), max(error_list_full))
#plot(error_list_full, pka, 'yo', d_range, slope*d_range+intercept, '--k')

slope, intercept, r_value, p_value, stderr = linregress(error_list_pruned, pka)
d_range = np.linspace(min(error_list_pruned), max(error_list_pruned))
#plot(error_list_pruned, pka, 'yo', d_range, slope*d_range+intercept, '--k')

slope, intercept, r_value, p_value, stderr = linregress(error_list_oh_sub, pka)
d_range = np.linspace(min(error_list_oh_sub), max(error_list_oh_sub))
#plot(error_list_oh_sub, pka, 'yo', d_range, slope*d_range+intercept, '--k')

slope, intercept, r_value, p_value, stderr = linregress(error_list_cooh_sub, pka)
d_range = np.linspace(min(error_list_cooh_sub), max(error_list_cooh_sub))
#plot(error_list_cooh_sub, pka, 'yo', d_range, slope*d_range+intercept, '--k')

slope, intercept, r_value, p_value, stderr = linregress(paper_error[0], pka)
d_range = np.linspace(min(paper_error[0]), max(paper_error[0]))
#plot(error_list_cooh_sub, pka, 'yo', d_range, slope*d_range+intercept, '--k')

slope, intercept, r_value, p_value, stderr = linregress(paper_error[1], pka)
d_range = np.linspace(min(paper_error[1]), max(paper_error[1]))
#plot(error_list_cooh_sub, pka, 'yo', d_range, slope*d_range+intercept, '--k')