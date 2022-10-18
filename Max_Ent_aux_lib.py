# In [1]:

import qutip, math, cmath
import matplotlib.pyplot as plt 
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg

# In [1]:

def norm_at_timet(rho_M, time_t_minus_one):
    return sum(1)

def A_mmplustwo_matrix_elmt(cohrnc, time, power_law_factor):
    return np.e**(cohrnc * time**(-power_law_factor))

def B_mmminustwo_matrix_elmt(cohrnc, time, power_law_factor):
    return np.e**(cohrnc * time**(-power_law_factor))

def C_m_matrix_elmt(cohrnc, time, power_law_factor):
    return cohrnc * np.e**(cohrnc * time**(-power_law_factor))

def diag_mm_matrix_elmt(cohr, time, power_law_factor):
    return (-A_mmplustwo_matrix_elmt(cohr, time, power_law_factor) - B_mmminustwo_matrix_elmt(cohr, time, power_law_factor) 
            + p * C_m_matrix_elmt(cohr, time, power_law_factor))

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

def M_matrix_test(total_no_spins, M_tensor):
    for i in range(len(M_tensor)):
        if (len(M_tensor[i]) != total_no_spins):
            print("Error on the ", i, "-th row. It has", len(M_tensor[i]), "elements")
        if (len(M_tensor[:,i]) != total_no_spins):
            print("Error: The ", i, "-th column. It has", len(M_tensor[:,i], "elements"))
