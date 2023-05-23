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
    return np.e**(-cohrnc * time**(1+power_law_factor))

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
    return np.e**(-cohrnc * time**(1+power_law_factor))

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
    return cohrnc * np.e**(-cohrnc * time**(1+power_law_factor))

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

def normalization_factor(total_no_cohr, time, power_law_factor):
    """
    This module returs a normalization for the coherences
    """
    return sum(np.e**(-m*time**(1+power_law_factor)) for m in range(total_no_cohr))

# In [2]:

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
    if type(parameters) == dict:
        M = parameters["total_no_cohrs"]; p = parameters["p_factor"]; a = parameters["power_law_factor"]
    if type(parameters) == list:
        M = parameters[0]; p = parameters[1]; a = parameters[2]
        
    m_matrix_list = []; t = timet;
    normalization = normalization_factor(total_no_cohr = M,
                                         time = t,
                                         power_law_factor = a)
    
    for m in range(M):
        if m == 0:
            local_array = np.array([A_mmplustwo_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a) + 
                                           p * C_m_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, 
                                                                   power_law_factor = a)] 
                                          + [0 for i in range(M-2)])
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None
        if (m > 0 and m < M-1):
            list_with_zeros = [0 for i in range(m-1)]
            local_array = np.array(list_with_zeros
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)]
                                          + [diag_mm_matrix_elmt(cohrnc = m, time = t, power_law_factor = a, p = p)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, power_law_factor = a)]  
                                          + [0 for i in range(M - (len(list_with_zeros)+3))])
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None       
        if (m == M-1):
            local_array = np.array([0 for i in range(M - 2)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a) + 
                                               p * C_m_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)])
            local_array = local_array/normalization
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

# In [3]:

def Mtensor_2mx2m_dimensional_symplectic(parameters, init_configurations, timet, 
                                            closed_boundary_conditions = False,
                                            visualization = False,
                                            as_qutip_qobj = False):
    """
    This module constructs the skew-symmetric (2M by 2M)-dimensional M-matrix,
    in terms of which the system of (2M by 2M) quasi-coupled real-valued differential equations is written.
    This M-matrix takes the form of 
    
    M = [ 0^{M},  M^{1}
          -M^{1}   ,  0^{M}]
    
    wherein 0^{M} is a (M by M)-null matrix, and where M^{1} is the M-matrix written in equation (1).
    This module takes as input: 
        ***. 1. parameters: a dictionary of parameters
             **. where, in said dictionary, the following parameters are stored: 
                 a. the total number of coherences to be considered, 
                 b. the strength of the dipolar-dipolar Sigma-interaction Hamiltonian, labelled p,
                 c. and the power law factor,
        ***. 2. init_configurations: an initial configuration for the coherences at time t0,
        ***. 3. timet: some arbitrary time t>t0, for which the M-matrix is desired.
        ***. 4. closed_boundary_conditions: a boolean option, not implemented as of yet. 
        ***. 5. visualization: a boolean option for printing out a Hinton diagram of the M-matrix.
                               Only applicable when the total number of coherences is equal or less 
                               than 20.
        ***. 6. as_qutip_qobj: boolean option. When toggled, the output M-matrix will be instantiated 
                               as a QuTip quantum object (qutip.Qobj)
                               
                               
    ===> Returns: a (2M by 2M) np.array of real numbers with the desired skew-symmetric form.  
    
    Warnings: Only real values allowed for the parameters. 
     
    """
    if type(parameters) == dict:
        M = parameters["total_no_cohrs"]; p = parameters["p_factor"]; a = parameters["power_law_factor"]
    if type(parameters) == list:
        M = parameters[0]; p = parameters[1]; a = parameters[2]
        
    m_matrix_list = []; t = timet;
    m_dimensional_zero_array = [0 for m in range(M)]
    normalization = normalization_factor(total_no_cohr = M,
                                         time = t,
                                         power_law_factor = a)
    
    for m in range(M):
        if m == 0:
            local_array = np.array(m_dimensional_zero_array + [A_mmplustwo_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a) + 
                                           p * C_m_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, 
                                                                   power_law_factor = a)] 
                                          + [0 for i in range(M-2)])
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None
        if (m > 0 and m < M-1):
            list_with_zeros = [0 for i in range(m-1)]
            local_array = np.array(m_dimensional_zero_array + list_with_zeros
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)]
                                          + [diag_mm_matrix_elmt(cohrnc = m, time = t, power_law_factor = a, p = p)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, power_law_factor = a)]  
                                          + [0 for i in range(M - (len(list_with_zeros)+3))])
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None       
        if (m == M-1):
            local_array = np.array(m_dimensional_zero_array + [0 for i in range(M - 2)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a) + 
                                               p * C_m_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)])
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None
            
    for m in range(M):
        if m == 0:
            local_array = np.array([A_mmplustwo_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a) + 
                                           p * C_m_matrix_elmt(cohrnc = m, time = t, 
                                                                   power_law_factor = a)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, 
                                                                   power_law_factor = a)] 
                                          + [0 for i in range(M-2)]
                                + m_dimensional_zero_array)
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None
        if (m > 0 and m < M-1):
            list_with_zeros = [0 for i in range(m-1)]
            local_array = np.array(list_with_zeros
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)]
                                          + [diag_mm_matrix_elmt(cohrnc = m, time = t, power_law_factor = a, p = p)] 
                                          + [A_mmplustwo_matrix_elmt(cohrnc = m+2-2, time = t, power_law_factor = a)]  
                                          + [0 for i in range(M - (len(list_with_zeros)+3))]
                                + m_dimensional_zero_array)
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None       
        if (m == M-1):
            local_array = np.array([0 for i in range(M - 2)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)] 
                                          + [B_mmminustwo_matrix_elmt(cohrnc = m, time = t, power_law_factor = a) + 
                                               p * C_m_matrix_elmt(cohrnc = m, time = t, power_law_factor = a)]
                                + m_dimensional_zero_array)
            local_array = local_array/normalization
            m_matrix_list.append(local_array)
            local_array = None
    
    if visualization and M <= 20:
        qutip.Hinton(qutip.Qobj(m_matrix_list))
    
    if as_qutip_qobj:
        m_matrix_list = qutip.Qobj(m_matrix_list)
    else: 
        pass
    
    m_matrix_list = np.asarray(m_matrix_list)
    m_matrix_list = .5*(m_matrix_list + np.transpose(m_matrix_list)) 
    return m_matrix_list

# In [3]:

def complex_differential_system(cohr_complex, t, parameters):
    Mtensor = Mtensor_2mx2m_dimensional_symplectic(parameters = parameters, init_configurations = cohr_complex, 
                                                   timet = t)
    
    Mtensor_loc = np.asarray(Mtensor)
    Mtensor_loc = .5*(Mtensor + np.transpose(Mtensor)) 
    """
    This module sets up the system of (2M by 2M) real-valued coupled differential equations:
    M = [ 0^{M},  M^{1}
          -M^{1}   ,  0^{M}]
          
               [a^{M}(t)      [0^{M},  M^{1}      [a^{M}(t)
        i d/dt              =                   . 
                b^{M}(t)]      -M^{1},  0^{M}]     b^{M}(t)] 
           
    where a^{M} is a M-dimensional vector, corresponding to the real part of the coherences vector c(t),
          b^{M} is a M-dimensional vector, corresponding to the imaginary part of the coherences vector c(t),
          0^{M} is a null M-dimensional vector,
          and where M^{1} is the (M by M)-dimensional M-matrix written in equation (1).
    
    This module takes as input: 
    ***. 1. parameters: an initial configuration for the coherences at time t0,
             **. where, in said dictionary, the following parameters are stored: 
                 a. the total number of coherences to be considered, 
                 b. the strength of the dipolar-dipolar Sigma-interaction Hamiltonian, labelled p,
                 c. and the power law factor,
                 
        ***. 2. cohr_complex: an initial configuration for the complex-valued coherences at time t0,
        ***. 3. timet: some arbitrary time t>t0, for which the M-matrix is desired.
                               
    ===> Returns: a (2M)-dimensional np.array of complex-valued numbers, corresponding to the right-hand side of the
                     previous equation written in this module's documentation. 
    
    """
    return Mtensor_loc @ cohr_complex
