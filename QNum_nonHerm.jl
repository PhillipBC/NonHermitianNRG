"""
    NRG utilizing quantum numbers to block diagonalize - Phil July 2024 
    (Adapted from Andrews Fortran code for Hermitian systems)
"""
#
using SparseArrays  # Sparse matrices for fast multiplcations
using LinearAlgebra # LinearAlgebramethods
using Distributions # Random Distributions (Only if you want noise added)
using JLD2          # Julia file saving 
using GenericSchur  # Schur decomposition for accurate eigedecomp. 

function qn(QN::Dict, l::Int, q1::Int, q2::Int)
    # the 'get(D,x,i)' function returns the value of the key 'x' in the Dictionary 'D'
    # If there is no key 'x', then it will return the default value 'i' provided

    # So, return the index of the quantum number (l,q1,q2) in the QN Dict, or return 0 if it doesn't exist
    return get(QN, (l, q1, q2), 0)
end
function new_qn(QN::Dict, iter_count::Array, l::Int, q1::Int, q2::Int)
    # Take in QN as the Quantum Number Dictionary
    # and the array 'last' keeping track of how many unique quanutm numbers there have been this iteration
    # increase the value of last[l] by 1, and then store the new quantum number with that updated counter
    pos = iter_count[l] += 1
    # Add the quantum number to the QN Dict with value pos
    QN[(l, q1, q2)] = pos

    return QN, iter_count, pos
end
function get_LR_eig(A::Matrix)
    # Get the left and right eigenvectors of the matrix A
    # We get them both using a schur decomposition of the matrix,
    # We also get the eigenvalues and sort them accordnigly
    # The eigevctors are then sorted in the same manner 

    S = schur(A)
    evals = S.values
    Ur = eigvecs(S)
    Ul = eigvecs(S, left=true)

    if sort_type == "LowRe" #|| sort_type == "LowReMag"  
        e_index = sortperm(evals, by=x -> (real(x), imag(x))) # sort by real component (and then by imag)
        evals = evals[e_index] 
        Ur = Ur[:, e_index]
        Ul = Ul[:, e_index]
    elseif sort_type == "LowMag" || sort_type == "LowReMag" 
        e_index = sortperm(evals, by=x -> abs(x)) # sort by real component (and then by imag)
        evals = evals[e_index]
        Ur = Ur[:, e_index]
        Ul = Ul[:, e_index]
    elseif sort_type == "LowImag"
        e_index = sortperm(evals, by=x -> (imag(x), real(x))) # sort by real component (and then by imag)
        evals = evals[e_index]
        Ur = Ur[:, e_index]
        Ul = Ul[:, e_index]
    else
        throw("BAD SORT METHOD")
    end 
    # Can take Random combinations of eigenvectors with common eigenvalues to avoid self orthogonalilty 
    do_rand_comb = false
    if do_rand_comb
        j = 2
        D = length(evals)
        while j <= length(evals)
            if abs(evals[j] - evals[j-1]) < 1e-9
                #println("degen at $j")
                inds = findall(x -> abs(x - evals[j]) < 1e-9, evals)
                #println("combinations of $(inds)")
                degen = length(inds)
                #println("Num of degens: ", degen)
                #combs = collect(combinations(1:length(inds),2))
                lv = Ul[:, inds]
                rv = Ur[:, inds]
                Rs = zeros(ComplexF64, D, degen)
                Ls = zeros(ComplexF64, D, degen)
                for j in eachindex(inds)
                    R = Rs[:, j]
                    L = Ls[:, j]
                    coeffs = rand(degen)
                    for n in eachindex(inds)
                        R += coeffs[n] .* rv[:, n]
                        L += coeffs[n] .* lv[:, n]
                    end
                    Rs[:, j] = R ./ norm(R)
                    Ls[:, j] = L ./ norm(L)
                end
                Ul[:, inds] = Ls
                Ur[:, inds] = Rs
                j = inds[end] + 1
            else
                j += 1
            end
        end
    end 

    # Bi-normalize the eigenvectors for projection operator to be the identity
    LR = 1 ./ (diag(Ul' * Ur)) # inverse of common eigenvector overlaps 
    Ul = LR' .* Ul # multiplies the conjugate of jth element of LR by the jth column of Ul  

    return evals, Ul, Ur
end

# NRG params
lmax = 120        # Number of iterations
rlim = 600      # Hilbert space dimension to truncate to (Max Hilbert space dim)
lambda = 3.0     # Logarthmic discretisation parameter (>1)
elim = 1e5       # Maximum allowed eigenvalue during truncation  

# System params
coupled_params = false # impurity coupled to chain or not (false for comparison with TB chain)
if coupled_params
    U = 0.3        # Spin-spin iteraction
    eps = -0.15     # Energy of spin
    V = 0.08        # impurity coupling
else
    U = 0.0        # Spin-spin iteraction
    eps = -20.0     # Energy of spin
    V = 0.0        # impurity coupling
end

magfield = 0.0  # Magnetic field strength
gamma = -0.1    # non-herm parameter
noise = 0.0     # noise magnitude (can add noise to diagonal of Hamiltonian)

n_pots = 100 #lmax

p_eps = eps;
p_U = U; # store the values for saving data later

sort_type = "LowRe" # "LowMag" "LowRe" "LowReMag"

szsym = 1      # Enforce Sz symmetry ? (1 yes, 0 no)
qsym = 0      # Enforce Q symmetry ? (1 yes, 0 no)

qmax = 15       # maximum Q value
szmax = 15      # maximum Sz value
qnmax = 200     # maximum number of QNs to loop over

##--------------

numqns = 0      # counter for number of unique QNs 

# Storage for Hilbert space dimensions per quantum number
rmax = zeros(Int, lmax + 2, qnmax + 1);
rmaxofk = zeros(Int, lmax + 2, qnmax + 1, 4);
rkept = zeros(Int, lmax + 2, qnmax + 1);
rstartk = zeros(Int, lmax + 2, qnmax + 1, 4);

#=
    # Basis Ordering --------------------------------------
    # Define states of added site, N+1:
    # |N+1;k=-1> = |down>        = c+_{down}|vac>
    # |N+1;k=0>  = |->           = |vac>
    # |N+1;k=+1> = |up>          = c+_{up}|vac>
    # |N+1;k=2>  = |up & down> = c+_{up}c+_{down}|vac>
    # Define matrix product state of full system
    # |N+1;k,r> = |N+1;k> X |N;r>
    # where |N+1;r> = Sum_{k,r'} U_{N+1}^r(k,r') |N+1;k,r'> are the diagonalized states at iteration N+1.
    # Matrix elements between states of added site:
    # M[x,y,z] = <N+1;k=y| c_{x} |N+1;k=z>, with x=1 for up and x=-1 for down, FOR ANY ITERATION N.
    # x = {-1,+1} -> {1,2}
    # y = {-1,0,1,2} -> {1,2,3,4}
    # z = {-1,0,1,2} -> {1,2,3,4}
=#

# Set up the starting NRG structure
# Start with matrix for appending a new site to the chain
# first index is spin (sigma), then index of bra site, index of ket site

M = zeros((2, 4, 4)) # store annihilation operators
# up spin
M[2, 2, 3] = +1.0 # |vac>  -> |up>
M[2, 1, 4] = +1.0 # |down> -> |up & down> 
# down spin
M[1, 2, 1] = +1.0 # |vac> -> |down>
M[1, 3, 4] = -1.0 # |up>  -> |down & up> = -|up & down> (Minus sign from anti-commutation relation)

ks = collect(-1:1:2)    # loop over the 4 basis states

function get_wilson_params_imag(gamma::Number, lambda::Number, lmax::Number)
    # EVERYTHING SHIFTED BY INDEX OF 2!!
    # Get parameters for the wilson chain mapping, want to decrease with growing length
    # Be careful with convention!!
    tn = zeros(Float64, lmax + 2)  # -1,0,1,2,....,lmax-1,lmax (hopping coeffs)
    en = zeros(ComplexF64, lmax + 2)  # -1,0,1,2,....,lmax-1,lmax - storage for eps params (mag fields)
    ns = collect(0:lmax)

    tn[2:lmax+2] = (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-ns .- 1)) .*
                   ((1.0 .- lambda .^ (-2 .* ns .- 1)) .* (1.0 .- lambda .^ (-2 .* ns .- 3))) .^ (-0.5)
    # ^^^ Equation (32) in Bulla Review, without the (lambda^(-l * 0.5)) term, incorporated elsewhere?
    # also Equation (2.15) in Krisna-Murthy with extra terms
    # This is Andrews convention -> Removes need to rescale the impurity Hamiltonian?

    #en[2:lmax+2] = (1im * gamma) .* (lambda .^ (0.5 .- ns)) #tn[2:lmax+2] #(lambda .^ (0.5 .- ns))
    #en[2:lmax+2] = (1im * gamma) .* tn[2:lmax+2] #(lambda .^ (0.5 .- ns))
    #en[2] = (1im * gamma) .* tn[2]
    nse = collect(2:n_pots+1)
    if n_pots < lmax
        en[nse] = (1.0im * gamma) .* tn[nse] #(lambda .^ (0.5 .- ns))
    else
        en[2:lmax+2] = (1.0im * gamma) .* tn[2:lmax+2] #(lambda .^ (0.5 .- ns))
    end

    # Hopping between impurity and Wilson chain 'zero orbital' is V (not tn).
    # A_\lambda >> Equation (22) in Bulla -> Depends on hybridization function \Delta(\omega)
    alambda = (lambda + 1.0) / (lambda - 1.0) * log(lambda) / 2.0
    tn[1] = sqrt(alambda / lambda) * V # Square root of lambda from iterative relation
    #en[1] = sqrt(alambda / lambda) * (1im * gamma)
    # sqrt of alambda as technically use t^2?
    #tn .*= rand(Uniform(0.95, 1.05), lmax + 2)
    #en .*= rand(Uniform(0.95, 1.05), lmax + 2)

    return tn, en
end
function intialise_ham_QN_NonHerm(eps::Number, U::Number, magfield::Number, szsym::Int, qsym::Int, iter_count::Array, rmax::Array, rkept::Array)
    #= 
        Initial Ham will initialize the following
        > lmin = 0
        > Will add 4 quantum number tuples (-1,q1,q2) to QN map
        > While doing this it will also update an array rmax, that is keeping track of how many unique quantum numbers there are in the initial Hamiltonian
        > Then it will create an energies array that stores the eigenvalues for H with just the impurity, based on their QN
        > Does something similar for the elem matrix, which I think is storing the creation operator coefficients?
    =#
    if szsym == 1
        println("---------------------------------")
        println("Enforcing particle-hole symmetry")
    end
    if qsym == 1
        println("---------------------------------")
        println("Enforcing spin-flip symmetry")
    end
    println("---------------------------------")

    lmin = -1 # very first iteration -> just the impurity itself 
    QN = Dict{Tuple,Number}() # intialise empty Dict

    QN, iter_count, qnind1 = new_qn(QN, iter_count, lmin + 2, -1, +0)  # Add (l-1,-1,0) to QN map (no charge, 0 sz spin)
    rmax[lmin+2, qnind1] = 1                                       # Indicate that at this iteration, this quantum number has states
    QN, iter_count, qnind1 = new_qn(QN, iter_count, lmin + 2, +1, +0)  # Add (l-1,+1,0) to QN map (2 charge, 0 sz spin)
    rmax[lmin+2, qnind1] = 1
    QN, iter_count, qnind1 = new_qn(QN, iter_count, lmin + 2, +0, -1)  # Add (l-1,0,-1) to QN map (1 charge, negative sz spin)
    rmax[lmin+2, qnind1] = 1
    QN, iter_count, qnind1 = new_qn(QN, iter_count, lmin + 2, +0, +1)  # Add (l-1,0,+1) to QN map (1 charge, positive sz spin)
    rmax[lmin+2, qnind1] = 1

    rkept[lmin+2, :] = rmax[lmin+2, :]  # rkept keeps track of number of "kept states" per iteration -> part of truncation process later
    #println(rkept[lmin+2,:])

    energies = [spzeros(ComplexF64, Int(2 * rlim)) for _ in 1:lmax+2, _ in 1:qnmax+1] # 2D array indexed by l,QN, with each element a 1D sparse array
    eground = zeros(ComplexF64, lmax + 2)

    # Set up inital eigenvalues/vectors (i.e. the diagonalized Hamiltonian)
    # for just the impurity
    # qn(l,q,sz) -> Check for iteration l, what index the state with
    # electron number q1, and total spin q2 has
    energies[lmin+2, qn(QN, lmin + 2, -1, +0)][1] = 0.0                      # Vaccuum energy
    energies[lmin+2, qn(QN, lmin + 2, +1, +0)][1] = 2 * eps + U              # spin up & down -> 2 electron energy
    energies[lmin+2, qn(QN, lmin + 2, +0, -1)][1] = eps - (magfield * 0.5)   # spin down -> 1 down electron
    energies[lmin+2, qn(QN, lmin + 2, +0, +1)][1] = eps + (magfield * 0.5)   # spin up   -> 1 up electron

    eground[lmin+2] = 0.0 # store ground state energy 

    # energies of basis states
    # spin up/down >> eps (mag field -> - for spin down, + for spin up)
    # empty >> 0
    # spin up & down >> 2*eps + interaction (U)

    # eigenvector matrix UM -> Matrix elements between impurity states : c^dagger operator
    UM = [spzeros((Int(2 * rlim)), (Int(2 * rlim))) for _ in 1:2, _ in 1:qnmax+1] # 2D array indexed by l,QN, with each element a 1D sparse array
    UMd = [spzeros((Int(2 * rlim)), (Int(2 * rlim))) for _ in 1:2, _ in 1:qnmax+1] # 2D array indexed by l,QN, with each element a 1D sparse array

    UM[2, qn(QN, lmin + 2, +0, -1)][1, 1] = +1.0 # c^dagger_up on a state with 1 down electron
    UM[2, qn(QN, lmin + 2, -1, +0)][1, 1] = +1.0 # c^dagger_up on a state with no electrons
    UM[1, qn(QN, lmin + 2, +0, +1)][1, 1] = -1.0 # c^dagger_down on a state with 1 up electron
    UM[1, qn(QN, lmin + 2, -1, +0)][1, 1] = +1.0 # c^dagger_down on a state with no electrons

    UMd[2, qn(QN, lmin + 2, +0, +1)][1, 1] = +1.0 # c_up on a state with 1 up electron
    UMd[2, qn(QN, lmin + 2, +1, +0)][1, 1] = +1.0 # c_up on a state with 2 electrons
    UMd[1, qn(QN, lmin + 2, +0, -1)][1, 1] = +1.0 # c_down on a state with 1 down electron
    UMd[1, qn(QN, lmin + 2, +1, +0)][1, 1] = -1.0 # c_down on a state with 2 electrons

    #println("rmax after intialise_ham_QN(): \n", rmax, "\n")

    return energies, UM, UMd, eground, rmax, rkept, QN, iter_count
end
function get_iter_Ham_QN_NonHerm(qnind1::Int, q0::Array, sz0::Array, rmax::Array, rmaxofk::Array, rstartk::Array, rkept::Array, l::Number, UM::Array, UMd::Array, M::Array, tn::Array, energies::Array, epsn::Array, prev_UldUr::Array)
    # Currently have qnind1 (index of state with QN=(q,sz) in the current iteration (l+2))
    H = zeros(ComplexF64, rmax[l+2, qnind1], rmax[l+2, qnind1])
    sigs = [-1, 1] # spins down/up
    m = 0 # counter for diagonal terms

    # loop over basis states
    for (jkp, kp) in enumerate(ks)
        (rmaxofk[l+2, qnind1, jkp] == 0) && continue    # if size of Hilbert space of this QN is zero, then skip to next loop step
        # get qnind2 (associated with state from previous iteration (l+1), i.e. H will link that state with the current state via their QNs)
        qnind2 = qn(QN, l + 1, q0[jkp], sz0[jkp])         # Find index of the desired QN (returns zero if there were no states with that QN in prev iteration)
        (qnind2 == 0) && continue                       # if there were no states in prev iteration with desired QN, skip

        for (jk, k) in enumerate(ks) # loop over possible states 
            (rmaxofk[l+2, qnind1, jk] == 0) && continue # if the size of the Hilbert space sector for this k is zero, then skip to the next cycle
            # Hop an up-spin (sigma = +1) or a down-spin (sigma=-1) electron from site (l-1) to (l)
            # so loop over sigma indices
            for s in eachindex(sigs)
                if !iszero(M[s, jk, jkp])
                    # loop over eigenstates with this qn
                    for r in 1:rmaxofk[l+2, qnind1, jk], rp in 1:rmaxofk[l+2, qnind1, jkp]
                        # loop over eigenstate inner products
                        for rpp in findall(x -> !iszero(x), prev_UldUr[qnind2, qnind2][:, rp]) #1:rmaxofk[l+2,qnind1,jkp]
                            #rpp = rp
                            H[rstartk[l+2, qnind1, jk]+r, rstartk[l+2, qnind1, jkp]+rp] +=
                                tn[l+1] * M[s, jk, jkp] * ((-1)^k) * UM[s, qnind2][r, rp] * prev_UldUr[qnind2, qnind2][rpp, rp]
                        end
                    end # (r,rp) loop
                end # non-zero M element

                # Assuming non-Hermitian, can't just do H += H', do the same thing but using the UMd matrix
                if !iszero(M[s, jkp, jk])
                    # loop over eigenstates with this qn
                    for r in 1:rmaxofk[l+2, qnind1, jk], rp in 1:rmaxofk[l+2, qnind1, jkp]
                        # loop over eigenstate inner products
                        for rpp in findall(x -> !iszero(x), prev_UldUr[qnind2, qnind2][:, rp]) #1:rmaxofk[l+2,qnind1,jkp]
                            #rpp = rp
                            H[rstartk[l+2, qnind1, jk]+r, rstartk[l+2, qnind1, jkp]+rp] +=
                                tn[l+1] * M[s, jkp, jk] * ((-1)^kp) * UMd[s, qnind2][r, rp] * prev_UldUr[qnind2, qnind2][rpp, rp]
                        end
                    end # (r,rp) loop
                end # non-zero M element

            end # sigma loop

            # diagonal elements
            #
            if k == kp
                # diagonal elements (not necessarily diagonal for Non Herm)
                for r in 1:rkept[l+1, qnind2]
                    for rp in 1:rkept[l+1, qnind2]
                        #rp = r
                        # m is used to keep track of where in the super space we are (r is the subspace)
                        H[r+m, rp+m] = (sqrt(lambda) * energies[l+1, qnind2][rp] + abs(k) * epsn[l+2]) * prev_UldUr[qnind2, qnind2][r, rp] # abs(k) counts the number of electrons in the state and applies that many epsilons
                    end # rp
                end # r
                m += rkept[l+1, qnind2] # increase m by this subspace dimension so the next update starts at the end of this update
            end
            #

        end # k loop
    end # kp

    return H
end
function enforce_QN_symmetry(l::Int, rmax::Array, energies::Array)
    if qsym == 1
        # loop over QNs (only loop over positive q)
        for q in 1:qmax
            for sz in -szmax:szmax
                qnind1 = qn(QN, l + 2, +q, sz) # get index of QN with +q
                qnind2 = qn(QN, l + 2, -q, sz) # get index of QN with -q
                if (qnind1 == 0) || (qnind2 == 0) || (rmax[l+2, qnind1] == 0) || (rmax[l+2, qnind2] == 0) # check there are states
                    continue
                end
                # first replace the +q state eval with the average of the ±q state evals
                energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] = 0.5 * (
                    energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] + energies[l+2, qnind2][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])])
                # then copy this to the -q state
                energies[l+2, qnind2][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] = energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])]
            end # sz
        end # q
    end # qsym

    if szsym == 1
        # loop over QNs (only loop over positive sz)
        for q in -qmax:qmax
            for sz in 1:szmax
                qnind1 = qn(QN, l + 2, q, +sz) # get index of QN with +sz
                qnind2 = qn(QN, l + 2, q, -sz) # get index of QN with -sz
                if (qnind1 == 0) || (qnind2 == 0) || (rmax[l+2, qnind1] == 0) || (rmax[l+2, qnind2] == 0) # check there are states
                    continue
                end
                # first replace the +q state eval with the average of the ±q state evals
                energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] = 0.5 * (
                    energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] + energies[l+2, qnind2][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])])
                # then copy this to the -q state
                energies[l+2, qnind2][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])] = energies[l+2, qnind1][1:min(rmax[l+2, qnind1], rmax[l+2, qnind2])]
            end # sz
        end # q
    end # szsym

    return energies
end
function truncate_QN(l::Int, QN::Dict, rmax::Array, rkept::Array, energies::Array, eground::Array)
    # Truncation by updating the rkept[] array, which keeps track of which eigenvalues/vectors are relevant
    # First picks a goal number of states to keep
    # Checks that is below some upper bound and adjusts if necessary 
    # Checks that that point is not in the middle of a 'clump' of (near) degenerate values
    # then truncates via the rkept[]

    # Store all the eigenvalues in a large 1D energies array - etot -------------------------------------------------------
    #println(rmax[l+2,:])
    #println("rmax[$l,:] at truncation step: \n", rmax[l+2,:], "\n")
    E = zeros(ComplexF64, sum(rmax[l+2, :])) # sum(rmax(l,:)) would be the number of valid QNs in this iteration

    m = 0 # counter
    for qnind1 in [v for (k, v) in QN if (k[1] == l + 2)] # find all the values in QN at iteration l 
        for r in 1:rmax[l+2, qnind1] # loop over states with this QN
            m += 1 # increase counter
            E[m] = energies[l+2, qnind1][r]
        end # r 
    end # q

    # Now sort the 1D array of energies
    if sort_type == "LowRe" || sort_type == "LowReMag"
        E = E[sortperm(E, by=x -> (real(x), imag(x)))] # sort by lowest real part
    elseif sort_type == "LowMag"
        E = E[sortperm(E, by=x -> abs(x))] # sort by absolute value
    elseif sort_type == "LowImag"
        E = E[sortperm(E, by=x -> (imag(x), real(x)))] # sort by lowest imag part
    else
        throw("Error: Incompatible sort_type parameter passed to truncation scheme. \nAccepts only 'LowRe' and 'LowMag' ")
    end
    eground[l+2] = E[1] # store the ground state energy for this iteration

    println("Length(E) = $(length(E))")
    println("|E[1]| = $(abs(E[1])), |E[end]| = $(abs(E[length(E)]))")

    # Now subtract the ground state energy from all eigenvalues -----------------------------------------------------------
    for qnind1 in [v for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
        for r in 1:rmax[l+2, qnind1] # loop over states with this QN
            energies[l+2, qnind1][r] = energies[l+2, qnind1][r] - eground[l+2] # If it is, subtract the ground state from the energy in the original eigenvalue storage
            #println("energies[$l,$qnind1][$r] = ", energies[l+2, qnind1][r])
        end # r 
    end # q
    E = E .- eground[l+2] # also subtract from 1D array 

    # if doing LowReMag truncation, sort again after rescaling 
    if sort_type == "LowReMag"
        E = E[sortperm(E, by=x -> abs(x))] # sort by absolute value
    end
    
    # Truncation by total number of states below ecut with no clumps: -----------------------------------------------------
    # Keep all the states if there are less of them than the limit, rlim
    rtot = min(rlim, sum(rmax[l+2, :]))
    println("min(rlim,sum(rmax[l+2,:])) = ", min(rlim, sum(rmax[l+2, :])))

    # if the last eigenvalue that would be stored after the truncation, is greater than the limit elim
    #if sort_type == "LowRe"
    #    if (abs(E[rtot]) > elim)
    #        println("E[rlim] > elim")
    #        r = findfirst(x -> real(x) > elim, E) # find the first eigenvalue above elim
    #        rtot = r-1 # replace the truncation point with the eigenvalue before that
    #    end # if 
    #elseif sort_type == "LowMag" || sort_type == "LowReMag"
    if (abs(E[rtot]) > elim)
        println("--------------------------------------")
        println("E[rlim] > elim")
        r = findfirst(x -> abs(x) > elim, E) # find the first eigenvalue above elim
        rtot = r - 1 # replace the truncation point with the eigenvalue before that
    end # if 
    #else
    #    throw("Error: Incompatible sort_type parameter passed to truncation scheme. ")
    #end

    # Now we check for ~degeneracies / clumping -------------------------------------------------------------------
    # we do this by starting at the truncation point, and checking that successive eigenvalues
    # are not approximately equal to this eigenvalue, and update the point if they are

    #if sort_type == "LowRe"
    if (rtot < sum(rmax[l+2, :]) - 1) && (abs(E[rtot+1] - E[rtot]) <= 1e-10) # check if current truncation point is in a clump of approximately equal eigenvalues
        println("rtot = $rtot -> CLUMP")
        r = findfirst(x -> abs(E[x+1] - E[x]) > 1e-10, rtot+1:sum(rmax[l+2, :])-1) # if so, find the end of the clump
        rtot = rtot + (r) # make this the truncation point
        println("rtot = $rtot")
    end
    #elseif sort_type == "LowMag" || sort_type == "LowReMag"
    #    if (rtot < sum(rmax[l+2, :]) - 1) && (abs(E[rtot+1] - E[rtot]) <= 1e-6) # check if current truncation point is in a clump of approximately equal eigenvalues
    #        #println("--------------------------------------")
    #        #println("rtot = $rtot -> CLUMP")
    #        r = findfirst(x -> abs(E[x+1] - E[x]) > 1e-6, rtot+1:sum(rmax[l+2, :])-1) # if so, find the end of the clump
    #        rtot = rtot + (r) # make this the truncation point
    #        #println("rtot = $rtot")
    #        println("rtot -> $rtot")
    #    end
    #else
    #    throw("Error: Incompatible sort_type parameter passed to truncation scheme. ")
    #end

    # ecut is now the last eigenvalue that is under elim and not in the middle of a 'clump'
    # loop over QNs 
    #println("Kept states before final truncation step: sum(rkept[$(l),:]) = ", sum(rkept[l+2,:]))
    if sort_type == "LowRe"
        ecut = real(E[rtot])
        println("ECUT = $ecut @ index = $rtot")

        rkept[l+2, :] = rmax[l+2, :] # keep track of states we will keep
        for qnind1 in [v for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
            for r in 1:rmax[l+2, qnind1]
                # if the energy is above the ecut bound we previously determined
                if (real(energies[l+2, qnind1][r]) > ecut)
                    #println("CUT HAPPENED")
                    rkept[l+2, qnind1] = r - 1 # then we don't keep this state, set its value to zero in the rkept array
                    break # break out of the inner for loop
                end
            end # r loop
        end # q
    elseif sort_type == "LowMag" || sort_type == "LowReMag" || sort_type == "LowImag"
        ecut = abs(E[rtot])
        println("ECUT = $ecut @ index = $rtot")

        rkept[l+2, :] = rmax[l+2, :] # keep track of states we will keep
        for qnind1 in [v for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
            for r in 1:rmax[l+2, qnind1]
                # if the energy is above the ecut bound we previously determined
                if (abs(energies[l+2, qnind1][r]) > ecut)
                    #println("CUT HAPPENED @ $(r-1)")
                    rkept[l+2, qnind1] = r - 1 # then we don't keep this state, set its value to zero in the rkept array
                    break # break out of the inner for loop
                end
            end # r loop
        end # q
    end
    #println("Kept states after final truncation step: sum(rkept[$(l),:]) = ", sum(rkept[l+2, :]))

    # save energies to file
    # TO DO !!!

    return energies, eground, rkept, ecut
end
function update_UM_QN_NonHerm_diffQNs(l::Int, QN::Dict, rkept::Array, rmaxofk::Array, rstartk::Array, M::Array, hr::Array, hl::Array, prev_UldUr::Array)
    # matrix elements of c+_{l,sigma}, where sigma=-1 is down-spin and sigma=+1 is up-spin.
    # NOTE: only calculate matrix elements between KEPT states.
    UM = [spzeros(ComplexF64, 4 * maximum(rmax[l+2, :]), 4 * maximum(rmax[l+2, :])) for _ in 1:2, _ in 1:qnmax+1] # 2D array indexed by sigma, QN, with each element a 1D sparse array
    UMd = [spzeros(ComplexF64, 4 * maximum(rmax[l+2, :]), 4 * maximum(rmax[l+2, :])) for _ in 1:2, _ in 1:qnmax+1] # 2D array indexed by sigma, QN, with each element a 1D sparse array

    qofk = abs.(ks) .- 1            # charge number is |k|-1 (-1 -> no electrons, 0 -> 1 electron, 1 -> 2 electrons)
    szofk = ks .* (2 .- abs.(ks))   # total sz is k*(2-|k|) (-1 -> down spin, 0 -> no spin (0 or 2 electrons), 1 -> up spin)

    sigs = [-1, 1]

    for (q, sz, qnind1) in [(k[2], k[3], v) for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
        (rkept[l+2, qnind1] == 0) && continue # if no states with this QN were "kept" (or if the QN isnt in the map) then move on.

        for (sig, sigma) in enumerate(sigs)
            # Quantum number of state post creation operator action
            qnind2_cre = qn(QN, l + 2, q + 1, sz + sigma) # look up the index of the resulting QN if the charge is increased and the spin is altered by sigma
            # Quantum number of state post annihilation operator action
            qnind2_ani = qn(QN, l + 2, q - 1, sz - sigma) # look up the index of the resulting QN if the charge is decreased and the spin is altered by sigma

            # then either both are possible transitions, or just one, or none
            if (qnind2_cre != 0 && qnind2_ani != 0)
                (rkept[l+2, qnind2_cre] == 0 && rkept[l+2, qnind2_ani] == 0) && continue # if there were no states this iteration with said QN, then skip
                # loop over the states via k,kp
                for (jk, k) in enumerate(ks)#
                    # quanutm number of state from previous iteration that must have existed for us to get to these states
                    prev_qnind2_cre = qn(QN, l + 1, q + 1 - qofk[jk], sz + sigma - szofk[jk])
                    prev_qnind2_ani = qn(QN, l + 1, q - 1 - qofk[jk], sz - sigma - szofk[jk])
                    for (jkp, kp) in enumerate(ks)
                        prev_qnind1 = qn(QN, l + 1, q - qofk[jkp], sz - szofk[jkp])
                        # If the states will have non-zero overlap, and the states being considered appear in this QN sector
                        if (M[sig, jkp, jk]) != 0 && (rmaxofk[l+2, qnind2_cre, jk] != 0) && (rmaxofk[l+2, qnind1, jkp] != 0)

                            # Store the matrix product (M[k,kp] * (Udagger * U)) in UM 
                            # loop over kept states
                            for r in 1:rkept[l+2, qnind2_cre], rp in 1:rkept[l+2, qnind1]
                                # loop over eigenstate overlaps
                                for s in 1:rmaxofk[l+2, qnind2_cre, jk], sp in 1:rmaxofk[l+2, qnind1, jkp]
                                    #sp = s
                                    #println("$(UldUr[qnind2_cre][s,sp])")
                                    UM[sig, qnind1][r, rp] += (M[sig, jkp, jk]*
                                                               conj(hl[qnind2_cre][rstartk[l+2, qnind2_cre, jk]+s, r])*
                                                               (hr[qnind1][rstartk[l+2, qnind1, jkp]+sp, rp])*prev_UldUr[prev_qnind2_cre, prev_qnind1][s, sp])[1]
                                end # s, sp loop
                            end # r, rp loop
                        end # if check

                        if (M[sig, jk, jkp]) != 0 && (rmaxofk[l+2, qnind2_ani, jk] != 0) && (rmaxofk[l+2, qnind1, jkp] != 0)
                            # Store the matrix product (M[k,kp] * (Udagger * U)) in UM 
                            # loop over kept states
                            for r in 1:rkept[l+2, qnind2_ani], rp in 1:rkept[l+2, qnind1]
                                # loop over eigenstate overlaps
                                for s in 1:rmaxofk[l+2, qnind2_ani, jk], sp in 1:rmaxofk[l+2, qnind1, jkp]
                                    #sp = s
                                    #println("$(UldUr[qnind2_ani][s,sp])")
                                    UMd[sig, qnind1][r, rp] += (M[sig, jk, jkp]*
                                                                conj(hl[qnind2_ani][rstartk[l+2, qnind2_ani, jk]+s, r])*
                                                                (hr[qnind1][rstartk[l+2, qnind1, jkp]+sp, rp])*prev_UldUr[prev_qnind2_ani, prev_qnind1][s, sp])[1]
                                end # s, sp loop
                            end # r, rp loop
                        end # if check

                    end # kp
                end # k
            elseif (qnind2_cre != 0 && qnind2_ani == 0)
                (rkept[l+2, qnind2_cre] == 0) && continue # if there were no states this iteration with said QN, then skip
                # loop over the states via k,kp
                for (jk, k) in enumerate(ks)
                    prev_qnind2_cre = qn(QN, l + 1, q + 1 - qofk[jk], sz + sigma - szofk[jk])
                    for (jkp, kp) in enumerate(ks)
                        prev_qnind1 = qn(QN, l + 1, q - qofk[jkp], sz - szofk[jkp])
                        # If the states will have non-zero overlap, and the states being considered appear in this QN sector
                        if (M[sig, jkp, jk]) != 0 && (rmaxofk[l+2, qnind2_cre, jk] != 0) && (rmaxofk[l+2, qnind1, jkp] != 0)
                            # loop over kept states
                            for r in 1:rkept[l+2, qnind2_cre], rp in 1:rkept[l+2, qnind1]
                                # loop over eigenstate overlaps
                                for s in 1:rmaxofk[l+2, qnind2_cre, jk], sp in 1:rmaxofk[l+2, qnind1, jkp]
                                    #sp = s
                                    #println("------------\n$(UldUr[qnind2_cre][s,sp])")
                                    UM[sig, qnind1][r, rp] += (M[sig, jkp, jk]*
                                                               conj(hl[qnind2_cre][rstartk[l+2, qnind2_cre, jk]+s, r])*
                                                               (hr[qnind1][rstartk[l+2, qnind1, jkp]+sp, rp])*prev_UldUr[prev_qnind2_cre, prev_qnind1][s, sp])[1]
                                end # s, sp loop
                            end # r, rp loop
                        end # if check

                    end # kp
                end # k
            elseif (qnind2_cre == 0 && qnind2_ani != 0)
                (rkept[l+2, qnind2_ani] == 0) && continue # if there were no states this iteration with said QN, then skip
                # loop over the states via k,kp
                for (jk, k) in enumerate(ks)
                    prev_qnind2_ani = qn(QN, l + 1, q - 1 - qofk[jk], sz - sigma - szofk[jk])
                    for (jkp, kp) in enumerate(ks)
                        prev_qnind1 = qn(QN, l + 1, q - qofk[jkp], sz - szofk[jkp])
                        if (M[sig, jk, jkp]) != 0 && (rmaxofk[l+2, qnind2_ani, jk] != 0) && (rmaxofk[l+2, qnind1, jkp] != 0)
                            # Store the matrix product (M[k,kp] * (Udagger * U)) in UM 
                            # loop over kept states
                            for r in 1:rkept[l+2, qnind2_ani], rp in 1:rkept[l+2, qnind1]
                                # loop over eigenstate overlaps
                                for s in 1:rmaxofk[l+2, qnind2_ani, jk], sp in 1:rmaxofk[l+2, qnind1, jkp]
                                    #sp = s
                                    #println("$(UldUr[qnind2_ani][s,sp])")
                                    UMd[sig, qnind1][r, rp] += (M[sig, jk, jkp]*
                                                                conj(hl[qnind2_ani][rstartk[l+2, qnind2_ani, jk]+s, r])*
                                                                (hr[qnind1][rstartk[l+2, qnind1, jkp]+sp, rp])*prev_UldUr[prev_qnind2_ani, prev_qnind1][s, sp])[1]
                                end # s, sp loop
                            end # r, rp loop
                        end # if check

                    end # kp
                end # k
            end

        end # sigma
        #println("UMd = Um' ? ", norm(Umd[1, qnind1] - Um[1, qnind1]'))
    end # q

    return UM, UMd
end
function save_data_QN_NonHerm(lmax, rlim, lambda, elim, p_U, p_eps, V, gamma, noise, energies, QN, rkept)
    if qsym == 1 && szsym == 1
        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "energies", energies)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "QN", QN)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "rkept", rkept)
    elseif szsym == 1 && qsym != 1
        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "energies", energies)

        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "QN", QN)

        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "rkept", rkept)
    elseif szsym != 1 && qsym == 1
        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "energies", energies)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "QN", QN)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "rkept", rkept)
    else
        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "energies", energies)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "QN", QN)

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        save(fstring, "rkept", rkept)
    end
end

# An array to keep track of the last position for each value of 'l' - Initialized to zero.
iter_count = zeros(Int, lmax + 2);

# Get wilson chain params
tn, en = get_wilson_params_imag(gamma, lambda, lmax); # for flat band
# Due to rescaling of Hamiltonian at each step, need to rescale parameters of H_0 also!
eps = p_eps / lambda;
U = p_U / lambda;
magfield = magfield / lambda

# intialise the Hamiltonian elements, and get first iteration impurity e_vals
energies, UM, UMd, eground, rmax, rkept, QN, iter_count = intialise_ham_QN_NonHerm(eps, U, magfield, szsym, qsym, iter_count, rmax, rkept);

function iterative_loop_NonHerm(rmax::Array, rkept::Array, UM::Array, UMd::Array, energies::Array, eground::Array, QN::Dict, iter_count::Array, numqns::Number, noise::Float64)
    # will need to keep track of eigenvector overlaps -> starts with just identity for initial orthogonal basis
    prev_UldUr = [sparse(diagm(ones(4))) for _ in 1:qnmax+1, _ in 1:qnmax+1]

    max_dim = 4 # keeping track of largest matrix diagonalized

    #  ks = {-1,0,1,2}
    qofk = abs.(ks) .- 1            # charge number is |k|-1 (-1 -> no electrons, 0 -> 1 electron, 1 -> 2 electrons)
    szofk = ks .* (2 .- abs.(ks))   # total sz is k*(2-|k|) (-1 -> down spin, 0 -> no spin (0 or 2 electrons), 1 -> up spin)

    q0 = zeros(Int, 4) # storage for q and sz values of previous iteration
    sz0 = zeros(Int, 4) 

    eground = zeros(ComplexF64, lmax + 2) # store ground states

    # perform lmax iterations
    for l in 0:lmax
        lQN = length(QN)
        if (l) % 1 == 0
            println("--------------------------------------\nIteration $l")
        end
        # Assign QN indices to QNs and define Hilbert spaces for this iteration:
        # > loop over quantum numbers
        for q in -qmax:qmax
            for sz in -szmax:szmax
                # QNs of states of previous iteration, q0(k) and sz0(k), 
                # which combine with added site (labelled k) to give current QN, q and sz
                q0 = q .- qofk      # The q that a state from the previous iteration must have had to end up in this state when the new site was added
                sz0 = sz .- szofk   # same for sz

                # size of (kept) Hilbert space of previous iteration, with QN q0(k) and sz0(k)
                running_HS = 0          # keeps tracking of cumulative Hilbert space size (will +1 later to account for 1 indexing )
                rstartk_temp = zeros(Int, 4) # keeps track of starting indices of Hilbert space sector (as does not get updated with final running update)
                rmaxofk_temp = zeros(Int, 4) # keeps track of the maximum size of the Hilbert space sector

                for (jk) in eachindex(ks)    # for the 4 possible states the new site can be in
                    tmp_qn = qn(QN, l + 1, q0[jk], sz0[jk]) # quantum number of state (from prev iteration)
                    if tmp_qn != 0
                        rmaxofk_temp[jk] = rkept[l+1, tmp_qn] # update this index with the number of states from the previous iteration with this desired QN 
                        rstartk_temp[jk] = running_HS    # store running value before its updated (Starting index for this Hilbert space sector)
                        running_HS += rmaxofk_temp[jk]   # update running with the number of states we found (possibly zero) -> Total size of Hilbert space processed so far
                    end
                end # k loop

                # if the current QN cannot be generated, continue to next QN
                # i.e. if the number of states from previous generation that can lead to current state is zero
                (sum(rmaxofk_temp) == 0.0) && continue # (condition) && (execute if true)

                # OTHERWISE -------------------------------------------------------------
                # Add current QN to QN map at this iteration l
                QN, iter_count, qnind1 = new_qn(QN, iter_count, l + 2, q, sz)

                # size of current Hilbert space, with QN q and sz
                rstartk[l+2, qnind1, :] = rstartk_temp[:] # store the starting indices of this QN Hilbert space sectors permanently for this iteration 
                rmaxofk[l+2, qnind1, :] = rmaxofk_temp[:] # store the sizes of this QN Hilbert space sectors permanently for this iteration 
                rmax[l+2, qnind1] = sum(rmaxofk_temp[:]) # store the total size of the Hilbert space sectors permanently for this iteration 
                #println("rmax[$(l),:] after updating with rmaxofk_temp: \n", rmax[l+2,:], "\n")
            end # sz loop 
        end # q loop
        # Generate Hamiltonian and diagonalize:
        # loop over quantum numbers
        # Make an empty Hamiltonian for each quantum number
        hr = [spzeros(ComplexF64, maximum(rmax[l+2, :]), maximum(rmax[l+2, :])) for _ in 1:qnmax+1] # 1D array indexed by QN, with each element a 2D sparse array (right eigenvector storage)
        hl = [spzeros(ComplexF64, maximum(rmax[l+2, :]), maximum(rmax[l+2, :])) for _ in 1:qnmax+1] # 1D array indexed by QN, with each element a 2D sparse array (left eigenvector storage)
        Hs = [spzeros(ComplexF64, maximum(rmax[l+2, :]), maximum(rmax[l+2, :])) for _ in 1:qnmax+1] # 1D array indexed by QN, with each element a 2D sparse array (left eigenvector storage)
        UldUr = [spzeros(ComplexF64, 4 * maximum(rmax[l+2, :]), 4 * maximum(rmax[l+2, :])) for _ in 1:qnmax+1, _ in 1:qnmax+1] # 1D array indexed by QN, wdiagm(ones(rlim)) #max(rmax[l+2, qnind1], rmax[l+2, qnind2]))) #ith each element a 2D sparse array
        for (q, sz, qnind1) in [(k[2], k[3], v) for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
            q0 = q .- qofk[:]      # The q that a state from the previous iteration must have had to end up in this state when the new site was added
            sz0 = sz .- szofk[:]   # same for sz

            # construct Hamiltonian matrix:
            H = get_iter_Ham_QN_NonHerm(qnind1, q0, sz0, rmax, rmaxofk, rstartk, rkept, l, UM, UMd, M, tn, energies, en, prev_UldUr)
            d = size(H)[1]
            if d > max_dim
                max_dim = d
            end
            if abs(noise) > 0 #&& l < 3
                #t_noise = noise * (10^(-l*1.0))
                t_noise = noise
                H += diagm((rand(Uniform(t_noise * 0.1, t_noise), (d)))) #+ 1im * diagm((rand(Normal(0, noise), (rmax))))
            end
            Hs[qnind1][1:d, 1:d] = H

            # Store the eigenvalues in the energies array, and the eigenvectors in H
            energies[l+2, qnind1][1:d], hl[qnind1][1:d, 1:d], hr[qnind1][1:d, 1:d] = get_LR_eig(H) 

            # Update the overlap matrix for the next iteration
            for (k2, qnind2) in [(k2, v) for (k2, v) in QN if (k2[1] == l + 2)]
                UldUr[qnind2, qnind1] = hl[qnind2][1:d, 1:d]' * hr[qnind1][1:d, 1:d]
            end

        end # q loop

        # Enforce symmetries  
        energies = enforce_QN_symmetry(l, rmax, energies)

        # Truncate the eigenvalues (via the rkept array)
        energies, eground, rkept, ecut = truncate_QN(l, QN, rmax, rkept, energies, eground)

        if (l) % 1 == 0
            println("Total states generated at this iteration : ", sum(rmax[l+2, :]))
            println("Energy cutoff for truncation : ", ecut)
            println("Kept states : ", sum(rkept[l+2, :]))
            println("\n--------------------------------------------------------------------------------")
        end

        if l == lmax
            return QN, iter_count, energies, rkept, Hs
        end

        if l != lmax
            # Calculate new matrix elements -> M[k,kp] * Udagger * U (From Notes)
            # For use in next iterative diagonalization step (Ham construction)
            UM, UMd = update_UM_QN_NonHerm_diffQNs(l, QN, rkept, rmaxofk, rstartk, M, hr, hl, prev_UldUr)

            # Update the overlap matrix for the next iteration 
            prev_UldUr = [spzeros(ComplexF64, rlim, rlim) for _ in 1:length(QN)-lQN, _ in 1:length(QN)-lQN]
            for (k1, qnind1) in [(k1, v) for (k1, v) in QN if (k1[1] == l + 2)] # find all the keys in QN at iteration l
                for (k2, qnind2) in [(k2, v) for (k2, v) in QN if (k2[1] == l + 2)]
                    prev_UldUr[qnind2, qnind1] = UldUr[qnind2, qnind1]
                end
            end
        end
        # How many QN combinations were created in this iteration?
        m = 0
        for (q, sz, qnind1) in [(k[2], k[3], v) for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
            if (rmax[l+2, qnind1] > 0)
                m += 1 # if there was a state with this QN, then increase counter
            end
        end # q
        if (m > numqns)
            numqns = m # if the number of QNs created this iteration exceeded the previous max record, increase the record.
        end

    end # l loop
    println("Maximum number of quantum number combinations = ", numqns)
    println("Maximum Hilbert space diagonalized : $max_dim")

    #save_data_QN_NonHerm(lmax, rlim, lambda, elim, p_U, p_eps, V, gamma, noise, energies, QN, rkept)
    return QN, iter_count, energies, rkept
end

Results = @timed iterative_loop_NonHerm(rmax, rkept, UM, UMd, energies, eground, QN, iter_count, numqns, noise)
QN, iter_count, energies, rkept = Results.value 
println("Time Taken - $(Results.time) s")

#
##-------------------------------------------------------------------
function plot_lowest_flow(energies::Array, rkept::Array)
    # Plot lowest energy values
    PyPlot.rc("mathtext", fontset="stix")
    PyPlot.rc("font", family="STIXGeneral", size=23)
    dpi_val = 100

    flow = zeros(ComplexF64, lmax + 2, maximum(sum.(eachrow(rkept))))
    for l in 0:lmax
        d = sum(rkept[l+2, :])
        E = zeros(ComplexF64, d) # sum(rmax(l,:)) would be the number of valid QNs in this iteration
        m = 0 # counter
        for q in -qmax:qmax # loop over QNs
            for sz in -szmax:szmax
                qnind1 = qn(QN, l + 2, q, sz) # get the index of this QN
                (qnind1 == 0) && continue
                (rkept[l+2, qnind1] == 0) && continue # if the no states with this QN were "kept" (or if the QN isnt in the map) then move on.
                for r in 1:rkept[l+2, qnind1] # loop over states with this QN
                    m += 1 # increase counter
                    E[m] = energies[l+2, qnind1][r]
                end # r
            end # sz
        end # q

        # Now sort the 1D array of energies
        if sort_type == "LowRe" #|| sort_type == "LowReMag"
            E = E[sortperm(E, by=x -> (real(x), imag(x)))]
        elseif sort_type == "LowMag" || sort_type == "LowReMag"
            E = E[sortperm(E, by=x -> abs(x))]
        elseif sort_type == "LowImag"
            E = E[sortperm(E, by=x -> (imag(x), real(x)))]
        else
            throw("Bad sort type")
        end
        flow[l+1, 1:d] = E
    end
    fig, (ax1, ax2) = subplots(1, 2, constrained_layout=true, figsize=(10.5, 4.25), dpi=dpi_val)
    n_vals = 32
    msz = 2.5
    lw = 0.01
    period = 2
    for j in 1:n_vals
        ax1.plot(collect(2:period:lmax), real.(flow[2:period:lmax, j]), marker="o", markersize=msz, lw=lw)
    end
    ax1.set_xlabel("\$n\$")
    ax1.set_ylabel("Re(\$E\$)")
    ax1.set_ylim(-0.1, 3)

    for j in 1:n_vals
        ax2.plot(collect(2:period:lmax), imag.(flow[2:period:lmax, j]), marker="o", markersize=msz, lw=0.05)
    end
    ax2.set_xlabel("\$n\$")
    ax2.set_ylabel("Im(\$E\$)")
    ax2.set_ylim(-0.5, 0.5)
    return flow
end
flow = plot_lowest_flow(energies, rkept);

function plot_flow_QN(QN::Dict, energies::Array, rkept::Array)
    PyPlot.rc("mathtext", fontset="stix")
    PyPlot.rc("font", family="STIXGeneral", size=23)
    dpi_val = 100
    #figure(constrained_layout=true, dpi=dpi_val, figsize=(6.5,5)) # no legend

    flow = zeros(ComplexF64, lmax + 1, maximum(sum.(eachrow(rkept))))
    for l in 0:lmax
        d = sum(rkept[l+2, :])
        E = zeros(ComplexF64, d)
        m = 0 # counter
        for (q, sz, qnind1) in [(k[2], k[3], v) for (k, v) in QN if (k[1] == l + 2)] # find all the keys in QN at iteration l
            #for q in -qmax:qmax # loop over QNs
            #    for sz in -szmax:szmax
            #qnind1 = qn(QN, l + 2, q, sz) # get the index of this QN
            #(qnind1 == 0) && continue
            (rkept[l+2, qnind1] == 0) && continue # if the no states with this QN were "kept" (or if the QN isnt in the map) then move on.
            for r in 1:rkept[l+2, qnind1] # loop over states with this QN
                m += 1 # increase counter
                E[m] = energies[l+2, qnind1][r]
            end # r
            #    end # sz
        end # q

        # Now sort the 1D array of energies
        if sort_type == "LowRe"
            eind = sortperm(E, by=x -> (real(x), imag(x))) # sort by real component (and then by imag)
        elseif sort_type == "LowMag" || sort_type == "LowReMag"
            eind = sortperm(E, by=x -> abs(x)) # sort by magnitude
        elseif sort_type == "LowImag"
            eind = sortperm(E, by=x -> (imag(x), real(x))) # sort by magnitude
        end
        flow[l+1, 1:d] = E[eind]
    end

    #tmp_qn = [k for (k, v) in QN if (k[1] == 2)] # find all the keys with l = 2
    #qns = [(0,0),(1,1),(0,2)]
    #n_qns = [5,3,3]
    #qns = [(k[2],k[3]) for (k, v) in QN if (k[1] == lmax + 2)]
    #n_qns = zeros(Int,length(qns))
    #for j in eachindex(n_qns)
    #    n_qns[j] = rkept[lmax+2, QN[lmax+2,qns[j][1],qns[j][2]]]
    #end
    #n_qns = [5,3,3]
    qns = [(1, 1), (1, -1)]
    n_qns = [3, 3]
    cols = ["red", "blue"]

    lw = 0.7
    msz = 2.5
    total_states = 0
    period = 1

    figure(constrained_layout=true, dpi=dpi_val, figsize=(6.5, 5.5))
    for (j, (q, sz)) in enumerate(qns)
        Re_E = [[] for _ in 1:n_qns[j]]

        ms = [[] for _ in 1:n_qns[j]]
        for n in eachindex(Re_E)
            for l in 0:period:lmax
                qnind1 = qn(QN, l + 2, q, sz)
                (qnind1 == 0) && continue # if not there, then just skip to next QN
                push!(Re_E[n], real(energies[l+2, qnind1][n]))
                push!(ms[n], l)
                #print(E)
            end
            plot(ms[n], Re_E[n], marker="o", markersize=msz, lw=lw, label="(q, sz, r) = $q, $sz, $n", color=cols[j])
        end
        total_states += length(ms)
    end
    println(total_states)
    xlabel("\$n\$")
    ylabel("Re(\$E\$)")
    legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.45, 1.18), ncol=4, fancybox=false, shadow=false)

    figure(constrained_layout=true, dpi=dpi_val, figsize=(6.5, 5.5))
    for (j, (q, sz)) in enumerate(qns)
        Im_E = [[] for _ in 1:n_qns[j]]

        ms = [[] for _ in 1:n_qns[j]]
        for n in eachindex(Im_E)
            for l in 0:period:lmax
                qnind1 = qn(QN, l + 2, q, sz)
                (qnind1 == 0) && continue # if not there, then just skip to next QN
                push!(Im_E[n], imag(energies[l+2, qnind1][n]))
                push!(ms[n], l)
                #print(E)
            end
            plot(ms[n], Im_E[n], marker="o", markersize=msz, lw=lw, label="(q, sz, r) = $q, $sz, $n", color=cols[j])
        end
    end
    xlabel("\$n\$")
    ylabel("Im(\$E\$)")
    legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.45, 1.18), ncol=4, fancybox=false, shadow=false)
end
#plot_flow_QN(QN, energies, rkept); 

##------------------------------------------------------------------

## THE FOLLOWING FUNCTIONS ARE FOR LOADING DATA FROM SAVED DATA
function load_data_QN_NonHerm_npots(lmax, rlim, lambda, elim, p_U, p_eps, V, gamma, noise)

    if qsym == 1 && szsym == 1
        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        energies = load(fstring, "energies")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        QN = load(fstring, "QN")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        rkept = load(fstring, "rkept")
    elseif szsym == 1 && qsym != 1
        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        energies = load(fstring, "energies")

        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        QN = load(fstring, "QN")

        fstring = "/QuantumNumbers/Data/eval_flow/Sz_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        rkept = load(fstring, "rkept")
    elseif szsym != 1 && qsym == 1
        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        energies = load(fstring, "energies")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        QN = load(fstring, "QN")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_conserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        rkept = load(fstring, "rkept")
    else
        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_energies_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        energies = load(fstring, "energies")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_QN_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        QN = load(fstring, "QN")

        fstring = "/QuantumNumbers/Data/eval_flow/Q_Sz_nonconserved/$(sort_type)/NHNRG_QNs_rkept_n$(lmax)_r$(rlim)_Lam$(lambda)_elim$(elim)_U$(p_U)_eps$(p_eps)_V$(V)_en_is_tn_gam$(gamma)_n$(n_pots)_eta$(noise)"
        fstring = "." * replace(fstring, "." => "_") * ".jld2"
        rkept = load(fstring, "rkept")
    end

    return energies, QN, rkept
end
energies, QN, rkept = load_data_QN_NonHerm_npots(lmax, rlim, lambda, elim, p_U, p_eps, V, gamma, noise);
# -------------------------------------------------------------

#------------------------------------------------------------

## THE FOLLOWING FUNCTIONS ARE USED FOR COMPARING THE NON-COUPLED MODEL
## TO THE TIGHT BINDING CHAIN

# Tight binding model comparison 
function plot_lowest_flow_realORimag(energies::Array, rkept::Array, do_real::Bool=true)
    PyPlot.rc("mathtext", fontset="stix")
    PyPlot.rc("font", family="STIXGeneral", size=23)
    dpi_val = 100

    flow = zeros(ComplexF64, lmax + 1, maximum(sum.(eachrow(rkept))))
    for l in 0:lmax
        d = sum(rkept[l+2, :])
        E = zeros(ComplexF64, d) # sum(rmax(l,:)) would be the number of valid QNs in this iteration
        m = 0 # counter
        for q in -qmax:qmax # loop over QNs
            for sz in -szmax:szmax
                qnind1 = qn(QN, l + 2, q, sz) # get the index of this QN
                (qnind1 == 0) && continue
                (rkept[l+2, qnind1] == 0) && continue # if the no states with this QN were "kept" (or if the QN isnt in the map) then move on.
                for r in 1:rkept[l+2, qnind1] # loop over states with this QN
                    m += 1 # increase counter
                    E[m] = energies[l+2, qnind1][r]
                end # r
            end # sz
        end # q

        # Now sort the 1D array of energies
        if sort_type == "LowRe"
            E = E[sortperm(E, by=x -> (real(x), imag(x)))]
        elseif sort_type == "LowMag" || sort_type == "LowReMag"
            E = E[sortperm(E, by=x -> abs(x))]
        elseif sort_type == "LowImag"
            E = E[sortperm(E, by=x -> (imag(x), real(x)))]
        else
            throw("Bad sort type")
        end
        flow[l+1, 1:d] = E
    end
    figure(constrained_layout=true, dpi=dpi_val)
    n_vals = 32
    msz = 2.5
    lw = 0.2
    period = 2
    if do_real
        for j in 1:n_vals
            plot(collect(2:period:lmax), real.(flow[2:period:lmax, j]), marker="o", markersize=msz, lw=lw)
        end
        xlabel("\$n\$")
        ylabel("Re(\$E\$)")
        ylim(-0.1, 3)
    else
        for j in 1:n_vals
            plot(collect(1:period:lmax), imag.(flow[1:period:lmax, j]), marker="o", markersize=msz, lw=lw)
        end
        xlabel("\$n\$")
        ylabel("Im(\$E\$)")
        ylim(-0.5, 0.5)
    end
    return flow
end#
function plot_lowest_flow_realANDimag(energies::Array, rkept::Array, mb_vals, L::Int=-1)
    PyPlot.rc("mathtext", fontset="stix")
    PyPlot.rc("font", family="STIXGeneral", size=23)
    dpi_val = 100

    flow = zeros(ComplexF64, lmax + 1, maximum(sum.(eachrow(rkept))))
    for l in 0:lmax
        d = sum(rkept[l+2, :])
        E = zeros(ComplexF64, d) # sum(rmax(l,:)) would be the number of valid QNs in this iteration
        m = 0 # counter
        for q in -qmax:qmax # loop over QNs
            for sz in -szmax:szmax
                qnind1 = qn(QN, l + 2, q, sz) # get the index of this QN
                (qnind1 == 0) && continue
                (rkept[l+2, qnind1] == 0) && continue # if no states with this QN were "kept" (or if the QN isnt in the map) then move on.
                for r in 1:rkept[l+2, qnind1] # loop over states with this QN
                    m += 1 # increase counter
                    E[m] = energies[l+2, qnind1][r]
                end # r
            end # sz
        end # q

        # Now sort the 1D array of energies
        if sort_type == "LowRe"
            E = E[sortperm(E, by=x -> (real(x), imag(x)))]
        elseif sort_type == "LowMag" || sort_type == "LowReMag"
            E = E[sortperm(E, by=x -> abs(x))]
        elseif sort_type == "LowImag"
            E = E[sortperm(E, by=x -> (imag(x), real(x)))]
        else
            throw("Bad sort type")
        end
        flow[l+1, 1:d] = E
    end
    #figure(constrained_layout=true, dpi=dpi_val)
    fig, (ax1, ax2) = subplots(1, 2, constrained_layout=true, figsize=(10.5, 4.25), dpi=dpi_val)
    n_vals = 128
    msz = 2.75
    lw = 0.01
    if L != -1
        period = 1
        ender = L
    else
        starter = 2
        period = 2
        ender = lmax + 1
    end
    for j in 1:n_vals
        ax1.plot(collect(starter:period:ender), real.(flow[starter:period:ender, j]), marker="o", markersize=msz, lw=lw)#, color="red")
    end
    ax1.axhline.(real.(mb_vals), linestyle="--", color="grey")  #, alpha=0.4)
    ax1.set_xlabel("\$n\$")
    ax1.set_ylabel("Re(\$E\$)")
    ax1.set_ylim(-0.1, 3)

    for j in 1:n_vals
        ax2.plot(collect(starter:period:ender), imag.(flow[starter:period:ender, j]), marker="o", markersize=msz, lw=lw) #color="red")
    end
    #if !coupled_params
        ax2.axhline.(imag.(mb_vals), linestyle="--", color="grey")  #, alpha=0.4)
    #end
    ax2.set_xlabel("\$n\$")
    ax2.set_ylabel("Im(\$E\$)")
    ax2.set_ylim(-0.4, 0.4)

    return flow
end#x

function tb_ham_old(L::Integer, gamma::Float64=0.0)
    #tn = zeros(L) # -1,0,1,2,....,lmax-1,lmax
    ns = collect(1:L+1)
    en = zeros(ComplexF64, L + 1)

    # normal t_n / e_n  from usual code
    tn = (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(ns) .- 1)) .*
         ((1.0 .- lambda .^ (-2 .* ns .- 1)) .* (1.0 .- lambda .^ (-2 .* ns .- 3))) .^ (-0.5)
    #en = (1im*gamma) .* tn
    en[1] = (1im * gamma) * tn[1]

    # Now multiply by \lambda^{-n/2}
    tn .*= (lambda .^ (-ns .* 0.5))
    en .*= (lambda .^ (-ns .* 0.5))
    H = zeros(ComplexF64, L, L)
    H[1:L, 1:L] += diagm(1 => tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tn[1:L-1] .* ones(L - 1))
    H[1:L, 1:L] += diagm(en[1:L])
    #H[L+1:2*L, L+1:2*L] += diagm(1 => tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tn[1:L-1] .* ones(L - 1))
    #H[L+1:2*L, L+1:2*L] += diagm(en[2:L+1])
    return H
end;
function tb_ham_NH_imag(L::Integer, gamma::Float64=0.0, tuner::Float64=1.0)
    #tn = zeros(L) # -1,0,1,2,....,lmax-1,lmax
    ns = collect(0:L)
    tb_en = zeros(ComplexF64, L + 1)

    # normal t_n / e_n  from usual code
    tb_tn = (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(ns) .- 1)) .* ((1.0 .- lambda .^ (-2 .* ns .- 1)) .* (1.0 .- lambda .^ (-2 .* ns .- 3))) .^ (-0.5)

    if n_pots == lmax || n_pots >= L
        nse = collect(1:L)
    else
        nse = collect(1:n_pots)
    end
    if length(nse) > 1
        tb_en[nse] = (gamma * (1.0im)) .* tb_tn[nse]
        #tb_en[nse[1]] = (tuner * gamma * (1.0)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse[1]) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse[1]) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse[1]) .- 3))) .^ (-0.5)
        #tb_en[nse[2:end]] = (gamma * (1.0)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse[2:end]) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse[2:end]) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse[2:end]) .- 3))) .^ (-0.5)
    else
        #tb_en[nse] = (gamma * (1im)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse) .- 3))) .^ (-0.5)
        tb_en[nse] = (gamma * (1.0im)) .* tb_tn[nse]
    end
    #en = (gamma) .* tn
    #en[1] = (gamma*1im) * (0.5 * (1.0 + lambda^(-1.0))) * (1.0 - lambda^(-(1) - 1)) *
    #        ((1.0 - lambda^(-2 * (1) - 1)) * (1.0 - lambda^(-2 * (1) - 3)))^(-0.5)
    #en[2] = (gamma) * (0.5 * (1.0 + lambda^(-1.0))) * (1.0 - lambda^(-(2) - 1)) *
    #        ((1.0 - lambda^(-2 * (2) - 1)) * (1.0 - lambda^(-2 * (2) - 3)))^(-0.5)

    # Now multiply by \lambda^{-n/2}
    tb_tn .*= (lambda .^ (-(ns .+ 2) .* 0.5))
    tb_en .*= (lambda .^ (-(ns .+ 1) .* 0.5))
    H = zeros(ComplexF64, L, L)
    H[1:L, 1:L] += diagm(1 => tb_tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tb_tn[1:L-1] .* ones(L - 1))
    H[1:L, 1:L] += diagm(tb_en[1:L])
    #H[L+1:2*L, L+1:2*L] += diagm(1 => tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tn[1:L-1] .* ones(L - 1))
    #H[L+1:2*L, L+1:2*L] += diagm(en[2:L+1])
    if abs(noise) > 0.0
        t_noise = 1e-15
        #H -= diagm((rand(Uniform(t_noise * 0.1, t_noise), (L))))
    end
    return H
end;
function tb_ham_NH_complex(L::Integer, V::Float64=0.0, gamma::Float64=0.0, tparam::Float64=1.0)
    #tn = zeros(L) # -1,0,1,2,....,lmax-1,lmax
    ns = collect(0:L)
    tb_en = zeros(ComplexF64, L + 1)

    # normal t_n / e_n  from usual code
    tb_tn = (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(ns) .- 1)) .* ((1.0 .- lambda .^ (-2 .* ns .- 1)) .* (1.0 .- lambda .^ (-2 .* ns .- 3))) .^ (-0.5)

    if n_pots < L 
        nse = collect(1:n_pots)
    else 
        nse = collect(1:L)
    end
    if length(nse) > 1
        #tb_en[nse] = (V) .* tb_tn[nse]
        tb_en[nse] = (V + (gamma * (1.0im))) .* tb_tn[nse]
        #tb_en[nse[1]] = (tuner * gamma * (1.0)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse[1]) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse[1]) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse[1]) .- 3))) .^ (-0.5)
        #tb_en[nse[2:end]] = (gamma * (1.0)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse[2:end]) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse[2:end]) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse[2:end]) .- 3))) .^ (-0.5)
    else
        #tb_en[nse] = (gamma * (1im)) .* (0.5 * (1.0 + lambda^(-1.0))) .* (1.0 .- lambda .^ (-(nse) .- 1)) .* ((1.0 .- lambda .^ (-2 .* (nse) .- 1)) .* (1.0 .- lambda .^ (-2 .* (nse) .- 3))) .^ (-0.5)
        #tb_en[nse] = (V) .* tb_tn[nse]
        tb_en[nse] = (V + (gamma * (1.0im))) .* tb_tn[nse]
    end
    #en = (gamma) .* tn
    #en[1] = (gamma*1im) * (0.5 * (1.0 + lambda^(-1.0))) * (1.0 - lambda^(-(1) - 1)) *
    #        ((1.0 - lambda^(-2 * (1) - 1)) * (1.0 - lambda^(-2 * (1) - 3)))^(-0.5)
    #en[2] = (gamma) * (0.5 * (1.0 + lambda^(-1.0))) * (1.0 - lambda^(-(2) - 1)) *
    #        ((1.0 - lambda^(-2 * (2) - 1)) * (1.0 - lambda^(-2 * (2) - 3)))^(-0.5)

    # Now multiply by \lambda^{-n/2}
    tb_tn .*= (lambda .^ (-(ns .+ 2) .* 0.5))
    tb_en .*= (lambda .^ (-(ns .+ 1) .* 0.5))
    H = zeros(ComplexF64, L, L)
    H[1:L, 1:L] += diagm(1 => tb_tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tb_tn[1:L-1] .* ones(L - 1))
    H[1:L, 1:L] += diagm(tb_en[1:L])
    #H[L+1:2*L, L+1:2*L] += diagm(1 => tn[1:L-1] .* ones(L - 1)) + diagm(-1 => tn[1:L-1] .* ones(L - 1))
    #H[L+1:2*L, L+1:2*L] += diagm(en[2:L+1])
    if noise > 0.0
        H += diagm((rand(Normal(0, 1e-11), (L))))
    end
    return H
end;

L = 40 # Number of iterations (wilson chain of length L+1)1.5
lambda = 3.0    # Logarthmic discretisation parameter (>1)
H = tb_ham_NH_complex(L, gamma, 0.0, 1.0)
#H = tb_ham_NH_imag(L, gamma, 1.0)

# Single particle eigenvalues
svals = schur(H).values
#svals = eigvals(H)
#svals = svals .- svals[1] 
svals = sort(vcat(svals, svals), by=x -> (real(x), imag(x))) # every level can be filled by 2 electrons (spin up & down)
#svals = sort(vcat(svals, svals), by=x -> abs(x)) # every level can be filled by 2 electrons (spin up & down)


using Combinatorics
function get_all_mb_vals(svals, D)
    L = length(svals)
    mb_vals = [0.0 + 0.0im] # storage for many body values
    j = 1
    counter = 0
    while counter < D && j <= min(length(svals), 24)
        #println(j)
        temp = (sum.(collect(combinations(svals, j))))
        #fe = -0.039041142601818624
        #if fe in temp
        #    x = findfirst(x -> x == fe, temp)
        #    println("FOUND IT - $j combinations @ $(collect(combinations(1:L, j))[x])")
        #end
        #mb_vals = unique(vcat(mb_vals, temp)) # take combinations
        mb_vals = vcat(mb_vals, temp) # take combinations
        #mb_vals = vcat(mb_vals, temp)

        counter = length(mb_vals)
        println("j = $j, counter = $counter")
        j += 1
    end
    sort!(mb_vals, by=x -> (real(x), imag(x)))
    return mb_vals
end
function get_lowest_mb_vals_NH(svals, D)
    L = length(svals)
    mb_vals = [0.0 + 0.0im] # storage for many body values
    #j = findfirst(x->abs(x) > 1e-2,svals)
    counter = 0
    j = 1
    while counter < D && j <= length(svals) #min(L, 24)
        if j % 5 == 0
            println(j)
        end
        temp = [svals[j]]
        for i in 1:j
            combs = collect(combinations(svals[2:j], i))
            for c in eachindex(combs)
                temp = vcat(temp, sum(combs[c]))
            end
        end
        mb_vals = (vcat(mb_vals, temp)) # take combinations
        #mb_vals = vcat(mb_vals, temp)

        counter = length(mb_vals)
        if j % 5 == 0
            println("j = $j, counter = $counter")
        end
        j += 1
    end
    sort!(mb_vals, by=x -> (real(x), imag(x)))
    #sort!(mb_vals, by=x -> (abs(x)))
    return mb_vals
end
function get_lowest_mb_vals(svals, D)
    Le = length(svals)
    N = findfirst(x -> real(x) >= 0.0, svals) - 1 # find the first non-negative value
    E0 = sum(svals[1:N]) # lowest energy state is just the negative energy levels filled up

    mb_vals = [E0] #sum(svals[1:N])
    j = 1
    counter = length(mb_vals)
    #println("Starting with $N lowest levels filled")
    while counter < D && j <= max(N, Le / 2)
        #println("j = $j, counter = $counter")
        #e = 1
        innies = [] # indices of eigenvalues to be added in 
        for c in 1:j
            innies = vcat(innies, collect(combinations(N+1:N+j, c))) # create combinations of c eigenvalues between N+1 and N+j
        end
        temp = zeros(ComplexF64, length(innies))
        if innies != [[0]]
            for m in eachindex(temp)
                temp[m] = E0 + sum(svals[innies[m]])
            end
            #temp = (sum.(collect(combinations(svals, N - j))))
            mb_vals = unique(vcat(mb_vals, temp)) # take combinations
        end

        #if j < N 
        outties = [] # indices of eigenvalues to be take out
        for c in 1:j
            outties = vcat(outties, collect(combinations(N-min(j, N - 1):N, c)))
        end
        temp = zeros(ComplexF64, length(outties))
        #println(outties)
        if outties != [[0]]
            for m in eachindex(temp)
                temp[m] = E0 - sum(svals[outties[m]])
            end
            #temp = (sum.(collect(combinations(svals, N - j))))
            mb_vals = unique(vcat(mb_vals, temp)) # take combinations
        end

        if outties != [[0]] && innies != [[0]] # take out and put in at the same time
            temp = zeros(ComplexF64, length(innies) * length(outties))
            c = 1
            for m in eachindex(innies)
                for k in eachindex(outties)
                    temp[c] = E0 + sum(svals[innies[m]]) - sum(svals[outties[k]])
                    c += 1
                end
            end
            mb_vals = unique(vcat(mb_vals, temp)) # take combinations
        end
        #end

        #temp = (sum.(collect(combinations(svals, N + j))))
        #mb_vals = unique(vcat(mb_vals, temp)) # take combinations

        counter = length(mb_vals)
        j += 1
    end
    mb_vals = unique(vcat(mb_vals, [0.0])) # take combinations
    sort!(mb_vals, by=x -> (real(x), imag(x)))
    return mb_vals
end
#mb_vals = get_lowest_mb_vals(svals, 10000)

#mb_vals = (mb_vals[sortperm(mb_vals, by=x -> (real(x), imag(x)))])
D = 1e3 #4^L

#mb_vals = get_all_mb_vals(svals, D)
#mb_vals = get_lowest_mb_vals_NH(svals, D)
mb_vals = get_lowest_mb_vals(svals, D)
#norm(all_mb_vals[1:400] - mb_vals[1:400])


mb_vals .-= mb_vals[1]

mb_vals .*= lambda^((L - 0) / 2)

mb_vals = sort(unique(mb_vals),by=x->(real(x),imag(x)))
#mb_vals = sort((mb_vals), by=x -> (real(x), imag(x)))
#mb_vals = sort(unique(mb_vals),by=x->abs(x))

D = length(mb_vals)
#flow = plot_lowest_flow_realANDimag(energies, rkept, conj(mb_vals[1:D]), -1)#.+rand(Normal(0,1e-10),D));
#figure()
#axhline.(real.(mb_vals[1:D]), linestyle="--", color="grey");  #, alpha=0.4) 
function plot_single_iteration(energies, rkept, mb_vals, L)#.+rand(Normal(0,1e-10),D)
    PyPlot.rc("mathtext", fontset="stix")
    PyPlot.rc("font", family="STIXGeneral", size=23)
    dpi_val = 110

    flow = zeros(ComplexF64, lmax + 1, maximum(sum.(eachrow(rkept))))
    for l in 0:lmax
        d = sum(rkept[l+2, :])
        E = zeros(ComplexF64, d) # sum(rmax(l,:)) would be the number of valid QNs in this iteration
        m = 0 # counter
        for q in -qmax:qmax # loop over QNs
            for sz in -szmax:szmax
                qnind1 = qn(QN, l + 2, q, sz) # get the index of this QN
                (qnind1 == 0) && continue
                (rkept[l+2, qnind1] == 0) && continue # if no states with this QN were "kept" (or if the QN isnt in the map) then move on.
                for r in 1:rkept[l+2, qnind1] # loop over states with this QN
                    m += 1 # increase counter
                    E[m] = energies[l+2, qnind1][r]
                end # r
            end # sz
        end # q

        # Now sort the 1D array of energies
        if sort_type == "LowRe"
            E = E[sortperm(E, by=x -> (real(x), imag(x)))]
        elseif sort_type == "LowMag" || sort_type == "LowReMag"
            E = E[sortperm(E, by=x -> abs(x))]
        elseif sort_type == "LowImag"
            E = E[sortperm(E, by=x -> (imag(x), real(x)))]
        else
            throw("Bad sort type")
        end
        flow[l+1, 1:d] = E
    end
    #figure(constrained_layout=true, dpi=dpi_val)
    fig, (ax1) = subplots(1, 1, constrained_layout=true, figsize=(5.3, 4.25), dpi=dpi_val)
    n_vals = length(mb_vals)
    msz = 4.0
    lw = 0.0
    ax1.plot(real.(flow[L, :]), imag.(flow[L, :]), marker="o", markersize=msz, lw=lw, label="NRG", color = "red")
    ax1.plot(real.(mb_vals), imag.(mb_vals), linestyle="none", marker="d", markersize=0.6 * msz, label="Tight-binding", color = "black") #, alpha=0.4)
    ax1.set_xlabel("\$\\Re(E)\$")
    ax1.set_ylabel("\$\\Im(E)\$")
    ax1.set_xlim(-0.5, 5.0)
    ax1.set_ylim(-1.7, 1.6)
    legend(fontsize=10, loc="upper right", ncol=2)
    return flow
end
plot_single_iteration(energies, rkept, (mb_vals[1:D]), L) 

