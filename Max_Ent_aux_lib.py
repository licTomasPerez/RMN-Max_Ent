# In [1]:

import qutip
import matplotlib.pyplot as plt 
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg
import math, cmath

# In [1]:

### como sería la norma esta???

def norm_at_timet(rho_M, time_t_minus_one):
    return sum()

def A_mmplustwo_matrix_elmt(cohrnc, time, power_law_factor = 1):
    """
    This module construct the weight of the m-th
    coherence's interaction with the (m+2)-th coherence, 
    according to the unperturbed Hamiltonian H0. 
    It takes the followings parameters as inputs:
    
    ***. the m-th coherence, labelled "cohrnc",
    ***. the time t,
    ***. and an optional real-valued power law factor for the 
         time. 
    
    It returns a real number. 
    """
    return np.e**(cohrnc * time**(-power_law_factor))

def B_mmminustwo_matrix_elmt(cohrnc, time, power_law_factor):
    """
    This module construct the weight of the m-th
    coherence's interaction with the (m-2)-th coherence, 
    according to the unperturbed Hamiltonian H0. 
    It takes the followings parameters as inputs:
    
    ***. the m-th coherence, labelled "cohrnc",
    ***. the time t,
    ***. and an optional real-valued power law factor for the 
         time. 
    
    It returns a real number. 
    """
    return np.e**(cohrnc * time**(-power_law_factor))

def C_m_matrix_elmt(cohrnc, time, power_law_factor):
    """
    This module construct the m-th coherence's self-
    interaction, associated with the perturbation Hamiltonian
    Sigma. It takes as parameters,
    
    ***. the m-th coherence, labelled "cohrnc",
    ***. the time t,
    ***. and an optional real-valued power law factor for the 
         time. 
    
    It returns a real number. 
    """
    return cohrnc * np.e**(cohrnc * time**(-power_law_factor))

def diag_mm_matrix_elmt(cohrnc, time, power_law_factor):
    """
    This module constructs the effective self-interaction for 
    the m-th coherence, due to both the unperturbed and the
    perturbation Hamiltonians. It takes as parameters:
    
    ***. the m-th coherence, labelled "cohrnc",
    ***. the time t,
    ***. and an optional real-valued power law factor for the 
         time. 
         
    It returns a real number.
    """
    return (-A_mmplustwo_matrix_elmt(cohrnc, time, power_law_factor)
            - B_mmminustwo_matrix_elmt(cohrnc, time, power_law_factor) 
            + p * C_m_matrix_elmt(cohrnc, time, power_law_factor))

def generating_function_M_matrix(parameters, cmt0_coeff_list, time, cbc = False):
    M = parameters["total_no_cohrs"]; p = parameters["p_factor"];  a = parameters["power_law_factor"]
    cm_list = cmt0_coeff_list
    m_matrix_list = []; t = time;
    
    for m in range(M):
        if m == 0:
            m_matrix_list.append(np.array([diag_mm_matrix_elmt(m,t,a)] + [0]
                                    + [A_mmplustwo_matrix_elmt(m+2,t,a)] + [0 for k in range(M-3)])) 
        if m == 1:
            local_list = [0] + [diag_mm_matrix_elmt(m,t,a)] + [0] + [A_mmplustwo_matrix_elmt(m+2, t, a)]
            local_length = len(np.array(local_list))
            m_matrix_list.append(local_list + [0 for k in range(M-local_length)])
        if (m > 1) and (m < M - 2):  
            local_list = [0 for j in range(m-2)]; local_length = len(local_list)
            local_list += ([B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] + [diag_mm_matrix_elmt(m,t,a)] + [0] 
                        + [A_mmplustwo_matrix_elmt(m+2,t,a)])
            local_list += [0 for j in range(M-local_length - 5)]
            m_matrix_list.append(np.array(local_list))
        if m == M - 2: 
            m_matrix_list.append(np.array([0 for i in range(m-2)] + [B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] 
                                 + [diag_mm_matrix_elmt(m,t,a)] + [0]))
        if m == M - 1:
            m_matrix_list.append(np.array([0 for i in range(m-2)] + [B_mmminustwo_matrix_elmt(-m-2,t,a)] + [0] 
                                 + [diag_mm_matrix_elmt(m,t,a)]))
            
    return qutip.Qobj(m_matrix_list)

def M_matrix_test(total_no_cohrs, M_tensor):
    """
    This module tests the triangular-like structure for the 
    M-matrix. It takes as input:
    ***. the total number of coherences,
    ***. and the M-matrix. 
    
    It returns a boolean result.
    """
    for i in range(len(M_tensor)):
        if (len(M_tensor[i]) != total_no_cohrs):
            print("Error on the ", i, "-th row. It has", len(M_tensor[i]), "elements")
        if (len(M_tensor[:,i]) != total_no_cohrs):
            print("Error: The ", i, "-th column. It has", len(M_tensor[:,i], "elements")
