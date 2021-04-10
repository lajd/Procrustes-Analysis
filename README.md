# Note: This repository has moved
Please see https://github.com/theochem/procrustes for an updated version of this respository.

# Procrustes Analysis
Procrustes analysis is a method for comparing the similarity between two objects, represented as matrices. The analysis attempts to find a transformation of one object (call it A) which minimizes the distance to another object (call it A0), where the distance metric used in this work is the Frobenius distance. 
The transformation is also a matrix, and is constrained to be a particular type of motion (eg. only translations/reflections, permutations, etc.)

This package computes the optimal transformations under the following constraints:
* Orthogonal
* Permutation
* Rotational Orthogonal
* Symmetric 
* Two-sided orthogonal
* Two-sided orthogonal single transformation
* Two-sided permutation single transformation


