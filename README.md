# NonHermitianNRG

Julia script that implements a non-Hermitian Numerical Renormalization Group (NRG) procedure to solve \
- The non-Hermitian Kondo model \
- The non-Hermitian Anderson Impurity model.

Written with Julia v1.10.

Required packages:\
LinearAlgebra\
SparseArrays\
JLD2\
GenericSchur\
DelimitedFiles\

The functions `intialise_ham_QN_NonHermKondo(...)` and `get_wilson_params_imag_Kondo(...)` initialize the system ($H_0$) as the Kondo model with complex coupling $J = J_R + iJ_I$.
The rest of the script is general and performs the iterative construction of the Wilson chain as described in the paper.
