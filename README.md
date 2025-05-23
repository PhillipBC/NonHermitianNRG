# NonHermitianNRG

_We ask that if you use this code in your research, please cite our accompanying paper on arXiv, which describes the procedure:_ [[arXiv:2504.07019](https://arxiv.org/abs/2504.07019)]


Julia script that implements a non-Hermitian Numerical Renormalization Group (NRG) procedure to solve 
- The non-Hermitian Kondo model 
- The non-Hermitian Anderson Impurity model.

Written in **Julia v1.11.**

In `Kondo_QNum_nonHerm.jl`, the functions `intialise_ham_QN_NonHermKondo(...)` and `get_wilson_params_imag_Kondo(...)` initialize the system ($H_0$) as the Kondo model with complex coupling $J = J_R + iJ_I$.
The rest of the script is general and performs the iterative construction of the Wilson chain as described in the paper.

Similarly, in `AIM_QNum_nonHerm.jl` for the Anderson Impurity model.

In all scripts, parameters can be loaded by setting the variable `do_load` to `true` in the script, and providing a correctly structured parameter file such as the two provided in the Inputs folder. 
Default values are also provided for all parameters within the scripts.

_Note: The NH-NRG procedure developed assumes that within a symmetry sector of the Hamiltonian there are no exact degeneracies, and thus the left and right eigenvectors form a bi-orthogonal basis._

## Required packages

- **LinearAlgebra**
- **SparseArrays** -- Sparse matrices for less RAM use
- **JLD2** -- For saving Julia data structures to file
- **GenericSchur** -- High precision eigensolver
- **DelimitedFiles** -- For reading in parameter files
- _PyPlot_ -- For plotting (Optional)

## General parameters

- `lmax (Integer)` -- Number of iterations to perform
- `rlim (Integer)` -- Hilbert space dimension to truncate to (Max Hilbert space dimension)
- `lambda (Float64)` -- Logarthmic discretisation parameter ($\Lambda>1$)
- `elim (Float64)` -- Maximum allowed eigenvalue magnitude during truncation
- `n_pots (Integer)` -- Number of Wilson chain sites with potentials
- `gamma (ComplexF64)` --  Chain potential strength
- `sort_type (String)` -- Sorting method to be used in truncation of complex eigenvalues

### Kondo specific parameters

- `J (ComplexF64)` -- Impurity-bath coupling
- `W (Float64)` -- Potential of zeroth bath site
- `magfield (Float64)` -- $S^z$ field on impurity

### Anderson Impurity model specific parameters

- `U (ComplexF64)` -- Spin-spin interaction
- `eps (ComplexF64)` -- On-site spin energy
- `V (ComplexF64)` -- Impurity-bath hybridization
- `magfield (Float64)` -- $S^z$ field on impurity

## Output

The scripts will write the energies (`energies`), residuals (`diffs`), bi-orthogonality measures (`biorths`), quantum number dictionary (`QN`), and indexing of kept states (`rkept`) to Julia-specific data files (`.jld`), for ease of loading and saving Julia objects.
This can be changed to suit the users' needs by updating the relevant saving function in the scripts.

The scripts provide plotting functions for the residuals, the bi-othogonality measures, and the eigenvalues versus iteration via the `PyPlot` package.
These functions illustrate how to access the quantum number dictionary correctly to extract eigenvalues at the relevant iteration.

## Density of states choice

The flat-band density of states for the bath is used in both the Kondo and Anderson Impurity models.
The density of states structure is encoded in the Wilson chain parameters $t_n$, $\epsilon_n$, with the flat-band parameters provided e.g., in [[Bulla 2008](https://doi.org/10.1103/RevModPhys.80.395)].
To alter the bath density of states, one needs to change the Wilson chain parameters, e.g., to the power-law density of states for the pseudogap Kondo model [[Bulla 1997](http://dx.doi.org/10.1088/0953-8984/9/47/014)].

