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
