# In [1]:

import qutip
import matplotlib.pyplot as plt 
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg
import math, cmath

# In [1]:

def A_mmplustwo_matrix_elmt(cohrnc, time, power_law_factor = .5):
    """
    This module construct the weight of the m-th
    coherence's interaction with the forward next-nearest-
    neighbour (NNN) coherence, ie. the (m+2)-th coherence, 
    according to the unperturbed Hamiltonian H0. 
        It takes the followings parameters as inputs:
    
        ***. 1. cohrnc: the m-th coherence,
        ***. 2. time: time
        ***. 3. power_law_factor: a real valued number, 
                                  between 0 and 1, 
                                  which characterizes 
                                  our approximation 
                                  for the coherence's
                                  time evolution as 
                                  the exponential of 
                                  a power law of time. 
    
     ===> It returns a real number: e**(cohrnc * time**(-power_law_factor))
         
    """
    return np.e**(cohrnc * time**(-power_law_factor))

def B_mmminustwo_matrix_elmt(cohrnc, time, power_law_factor):
    """
    This module construct the weight of the m-th
    coherence's interaction with the previous next-nearest-
    neighbour (NNN) coherence, ie. the (m-2)-th coherence, 
    according to the unperturbed Hamiltonian H0. 
        It takes the followings parameters as inputs:
    
        ***. 1. cohrnc: the m-th coherence,
        ***. 2. time: time
        ***. 3. power_law_factor: a real valued number, 
                                  between 0 and 1, 
                                  which characterizes 
                                  our approximation 
                                  for the coherence's
                                  time evolution as 
                                  the exponential of 
                                  a power law of time. 
    
     ===> It returns a real number: e**(cohrnc * time**(-power_law_factor))
         
    """
    return np.e**(cohrnc * time**(-power_law_factor))

def C_m_matrix_elmt(cohrnc, time, power_law_factor):
    """
    This module construct the diagonal interactions corresponding
    to the m-th's coherence <<self-interaction>>,
    associated with the perturbation Hamiltonian Sigma. 
        It takes the followings parameters as inputs:
    
        ***. 1. cohrnc: the m-th coherence,
        ***. 2. time: time
        ***. 3. power_law_factor: a real valued number, 
                                  between 0 and 1, 
                                  which characterizes 
                                  our approximation 
                                  for the coherence's
                                  time evolution as 
                                  the exponential of 
                                  a power law of time. 
    
     ===> It returns a real number: e**(cohrnc * time**(-power_law_factor))
    
    It returns a real number. 
    """
    return cohrnc * np.e**(cohrnc * time**(-power_law_factor))

def diag_mm_matrix_elmt(cohrnc, time, power_law_factor, p):
    """
    This module constructs the effective self-interaction for 
    the m-th coherence, due to both the unperturbed and the
    perturbation Hamiltonians. 
        It takes the followings parameters as inputs:
    
        ***. 1. cohrnc: the m-th coherence,
        ***. 2. time: time
        ***. 3. power_law_factor: a real valued number, 
                                  between 0 and 1, 
                                  which characterizes 
                                  our approximation 
                                  for the coherence's
                                  time evolution as 
                                  the exponential of 
                                  a power law of time. 
    
     ===> It returns a real number: 
    """
    return (-A_mmplustwo_matrix_elmt(cohrnc, time, power_law_factor)
            - B_mmminustwo_matrix_elmt(cohrnc, time, power_law_factor) 
            + p * C_m_matrix_elmt(cohrnc, time, power_law_factor))

def legacy_gen_func_complete_M_matrix(parameters, timet, 
                                           closed_boundary_conditions = False):
    """
    This matrix constructs the explicit sparse (triangular-like)
    form of the M-matrix, wherein M_{m, m'} is the weight of 
    the m-m' coherence term. It takes the following as input:
    ***. a dictionary of parameter
         **. where, in said dictionary,
             1. the total number of coherences to be 
                considered is explicited,
             2. the strength of the Sigma-interaction 
                Hamiltonian, labelled p,
             3. and the power law factor for its submodules.
    ***. an initial configuration for the coherences at time 0,
    ***. a mesh for the times, 
    ***. a boolean option, not implemented as of yet. 
    
    ===> Returns a real valued, triangular-like sparse matrix.
    """
    M = parameters["total_no_cohrs"]; p = parameters["p_factor"]; a = parameters["power_law_factor"]
    cm_list = init_configurations
    m_matrix_list = []; t = timespan;
    
    timet = t
    
    for m in range(M):
        if m == 0:
            m_matrix_list.append(np.array([diag_mm_matrix_elmt(m,t,a, p)] + [0]
                                    + [A_mmplustwo_matrix_elmt(m+2,t,a)] + [0 for k in range(M-3)])) 
        if m == 1:
            local_list = [0] + [diag_mm_matrix_elmt(m,t,a,p)] + [0] + [A_mmplustwo_matrix_elmt(m+2, t, a)]
            local_length = len(np.array(local_list))
            m_matrix_list.append(local_list + [0 for k in range(M-local_length)])
        if (m > 1) and (m < M - 2):  
            local_list = [0 for j in range(m-2)]; local_length = len(local_list)
            local_list += ([B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] + [diag_mm_matrix_elmt(m,t,a,p)] + [0] 
                        + [A_mmplustwo_matrix_elmt(m+2,t,a)])
            local_list += [0 for j in range(M-local_length - 5)]
            m_matrix_list.append(np.array(local_list))
        if m == M - 2: 
            m_matrix_list.append(np.array([0 for i in range(m-2)] + [B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] 
                                 + [diag_mm_matrix_elmt(m,t,a,p)] + [0]))
        if m == M - 1:
            m_matrix_list.append(np.array([0 for i in range(m-2)] + [B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] 
                                 + [diag_mm_matrix_elmt(m,t,a,p)]))
            
    return qutip.Qobj(m_matrix_list)

def gen_func_even_cohr_M_matrix(parameters, init_configurations, timet, 
                                            closed_boundary_conditions = False,
                                            visualization = False,
                                            as_qutip_qobj = False):
    """
    This module constructs the even-coherences weight matrix,
    M, wherein M_{m, m'} is the weight of the m-m' coherence 
    term, with both m and m' even numbered-coherences. The odd
    coherences are disregardes for these do not factor into the 
    M-matrix. It takes the following as input:
    ***. a dictionary of parameters
         **. where, in said dictionary,
             1. the total number of even coherences to be 
                considered is explicited,
             2. the strength of the Sigma-interaction 
                Hamiltonian, labelled p,
             3. and the power law factor for its submodules.
    ***. a mesh for the times, 
    ***. a boolean option, not implemented as of yet. 
    ***. a boolean option for visualizing a
             1. plot of the M-matrix's eigenvalues,
             2. and the M-matrix coefficients in a Hinton
                diagram.
    ***. a boolean option for returning the M-matrix as 
         quantum object.     
    
        ===> Returns a real-valued matrix. 
    *Note that his module does not return a sparse matrix. 
    """
    M = parameters["total_no_cohrs"]; p = parameters["p_factor"]; a = parameters["power_law_factor"]
    m_matrix_list = []; t = timet;
    
    for m in range(M):
        if m == 0:
            local_array = np.array([A_mmplustwo_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a) + 
                                           p * C_m_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2, time = t, 
                                                                   power_law_factor = a)] 
                                          + [0 for i in range(M-2)])
            local_array = local_array/(sum(local_array))
            m_matrix_list.append(local_array)
            local_array = None
        if (m > 0 and m < M-1):
            list_with_zeros = [0 for i in range(m-1)]
            local_array = np.array(list_with_zeros
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)]
                                          + [diag_mm_matrix_elmt(cohrnc = m, time = t, power_law_factor = a, p = p)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2, time = t, power_law_factor = a)]  
                                          + [0 for i in range(M - (len(list_with_zeros)+3))])
            local_array = local_array/(sum(local_array))
            m_matrix_list.append(local_array)
            local_array = None       
        if (m == M-1):
            local_array = np.array([0 for i in range(M - 2)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a) + 
                                               p * C_m_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)])
            local_array = local_array/(sum(local_array))
            m_matrix_list.append(local_array)
            local_array = None
    
    ### test:
    dimensions_equal_tot_no_cohr = [len(m_matrix_list) == M for i in range(len(m_matrix_list))]
    assert (np.all(dimensions_equal_tot_no_cohr and len(m_matrix_list) == M)),"Error: M-matrix is not square"
        
    if visualization:
        eigenvalues_list = linalg.eig(m_matrix_list)[0]
        plt.scatter([i+1 for i in range(len(eigenvalues_list))], np.array(eigenvalues_list), label = "M-matrix's eigenvalues")
        plt.matshow(m_matrix_list)
        
    if as_qutip_qobj:
        try: 
            m_matrix_list = qutip.Qobj(m_matrix_list)
        except NameError:
            print(NameError)
        
    return m_matrix_list

def choose_gen_func_M_matrix(parameters, init_configurations, timespan, 
                                          even_cohr_matrix_only = True):
    """
    This module construct the triangular-like matrix M,
    whose exponential yields the solution to the Markovian
    system of differential equations. It takes as parameters:
    
    ***. a dictionary of parameters, containing the total number 
           of coherences, the p-factor due to the unperturbed 
           Hamiltonian H_0, and the power-law factor.
    ***. An initial coherences' configuration,
    ***. A list of desired times 
    ***. A boolean option, not-implemented as of yet.
    """
    
    if even_cohr_matrix_only:
        try:
            m_matrix = gen_func_even_cohr_M_matrix(parameters, init_configurations, timespan, 
                                          closed_boundary_conditions = False,
                                          visualization = False,
                                          as_qutip_qobj = False)
        except NameError:
            print(NameError)
    else:
        try: 
            m_matrix = gen_func_complete_M_matrix(parameters, init_configurations, timespan, 
                                          closed_boundary_conditions = False)
        except NameError:
            print(NameError)
    
    return m_matrix

def M_matrix_test(total_no_cohrs, M_tensor):
    """
    This module tests the triangular-like structure for the 
    M-matrix. It takes as input:
    ***. the total number of coherences,
    ***. and the M-matrix. 
    
    It returns nothing but prints boolean result.
    """
    for i in range(len(M_tensor)):
        if (len(M_tensor[i]) != total_no_cohrs):
            print("Error on the ", i, "-th row. It has", len(M_tensor[i]), "elements")
        if (len(M_tensor[:,i]) != total_no_cohrs):
            print("Error: The ", i, "-th column. It has", len(M_tensor[:,i]), "elements")
    return None
