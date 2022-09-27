# In [1]:

import qutip
import matplotlib.pyplot as plt 
import numpy as np
import scipy.optimize as opt 
import scipy.linalg as linalg
import math, cmath

# In [2]:

def generating_function_M_matrix(total_no_spins, cmt0_coeff_list, param_list, cbc = False):
    
    cm_list = cmt0_coeff_list
    m_matrix_list = []
    M = total_no_spins
    
    p = param_list[0]
    
    if not cbc:  
        for m in range(M):
            if m == 0:
                m_matrix_list.append(([-np.e**(-m) - np.e**(-m) + p ** np.e**(-0)] + [0]
                                    + [np.e**(-m+2)] + [0 for k in range(M-3)])) 
            if m == 1:
                local_list = [0] + [-np.e**(-m) - np.e**(-m) + p ** np.e**(-0)] + [0] + [np.e**(-m+2)]; local_length = len(local_list)
                m_matrix_list.append(local_list + [0 for k in range(M-local_length)])
            if (m > 1) and (m < M - 2):  
                local_list = [0 for j in range(m-2)]; local_length = len(local_list)
                local_list += [np.e**(-m-2)] + [0] + [-np.e**(-m) - np.e**(-m) + p ** np.e**(-0)] + [0] + [np.e**(-m+2)]
                local_list += [0 for j in range(M-local_length - 5)]
                m_matrix_list.append(local_list)
            if m == total_no_spins - 2: 
                m_matrix_list.append([0 for i in range(m-2)] + [np.e**(-m-2)] + [0] + [-np.e**(-m) - np.e**(-m) + p ** np.e**(-0)] 
                                     + [0])
            if m == total_no_spins - 1:
                m_matrix_list.append([0 for i in range(m-2)] + [np.e**(-m-2)] + [0] + [-np.e**(-m) - np.e**(-m) + p ** np.e**(-0)])
            
    else:
        pass 
    
    return (m_matrix_list)

def M_matrix_test(total_no_spins, M_tensor):
    for i in range(len(M_tensor)):
        if (len(M_tensor[i]) != total_no_spins):
            print("Error on the ", i, "-th row. It has", len(M_tensor[i]), "elements")
        if (len(M_tensor[:,i]) != total_no_spins):
            print("Error: The ", i, "-th column. It has", len(M_tensor[:,i], "elements"))
