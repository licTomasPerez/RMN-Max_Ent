# In [1]:

import qutip
import numpy as np
import scipy.optimize as opt 
import pickle
import sys
import scipy.linalg as linalg

# In [2]:

def ev_checks(rho):
    a = bool; ev_list = linalg.eig(rho)[0]
    for i in range(len(ev_list)):s
        if (ev_list[i] > 0):
            a = True
        else:
            a = False
            print("Eigenvalues not positive")
    return a

def is_density_op(rho):
    return (qutip.isherm(rho) and (rho.tr() == 1 or (1 - rho.tr() < 10**-10)) and ev_checks(rho)) 

# In [3]: 

def one_body_spin_ops(N):
    loc_sx_list = []; loc_sy_list = []; loc_sz_list = []; loc_globalid_list = []
    id2 = qutip.qeye(2)
    sx = .5*qutip.sigmax()
    sy = .5*qutip.sigmay()
    sz = .5*qutip.sigmaz()
    loc_global_id = qutip.tensor([qutip.qeye(2) for k in range(N)])
    
    for n in range(N):
        operator_list = []
        for m in range(N):
            operator_list.append(id2)
        loc_globalid_list.append(loc_global_id)
        operator_list[n] = sx
        loc_sx_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sy
        loc_sy_list.append(qutip.tensor(operator_list))
        
        operator_list[n] = sz
        loc_sz_list.append(qutip.tensor(operator_list)) 
        
    return loc_globalid_list, loc_sx_list, loc_sy_list, loc_sz_list

# In [4]:

def all_two_body_spin_ops(big_list, N):
    loc_global_id_list, sx_list, sy_list, sz_list = big_list
      
    pauli_four_vec = [loc_global_id_list, sx_list, sy_list, sz_list];
        
    sxsa_list = []; sysa_list = []; szsa_list = []; two_body_s = [];
    
    sxsa_list = [sx_list[n] * pauli_four_vec[a][b] for n in range(N)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    sysa_list = [sy_list[n] * pauli_four_vec[a][b] for n in range(N)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    szsa_list = [sz_list[n] * pauli_four_vec[a][b] for n in range(N)
                                                   for a in range(len(pauli_four_vec))
                                                   for b in range(len(pauli_four_vec[a]))]
    
    two_body_s = [sxsa_list, sysa_list, szsa_list]
    return two_body_s

# In [5]: 

def two_body_spin_ops(big_list, N, build_all = False):
    loc_list = []
    if build_all:
        loc_list = all_two_body_spin_ops(big_list, N)
    else: 
        globalid_list, sx_list, sy_list, sz_list = big_list       
        loc_sxsx = []; loc_sysy = []; loc_szsz = [];
        
        loc_sxsx = [sx_list[n] * sx_list[m] for n in range(N)
                                            for m in range(N)]
        loc_sysy = [sy_list[n] * sy_list[m] for n in range(N)
                                            for m in range(N)]
        loc_szsz = [sz_list[n] * sz_list[m] for n in range(N)
                                            for m in range(N)]
        loc_list.append(loc_sxsx)
        loc_list.append(loc_sysy)
        loc_list.append(loc_szsz)
    return loc_list

# In [6]: 

def pines_et_al_hamiltonian(big_list, N, visualization, dmatrix):
    
    loc_global_id_list, sx_list, sy_list, sz_list = big_list
    
    H = 0
    
    # O(N^2), no es de lo más eficiente pero funciona. 
    # TODO con una list comprehension
    for i in range(N):
        for j in range(N):
            if i < j:
                #print(i+1)
                #print(j+1)
                H -= dmatrix[i][j] * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j])
    
    if visualization:
        qutip.hinton(H) 
         
    return H

# In [7]:

def Heisenberg_Hamiltonian(big_list, chain_type, N, visualization, Jx, Jy, Jz, h):
    spin_chain_type = ["XX", "XYZ", "XXZ", "XXX"]
    loc_globalid_list, sx_list, sy_list, sz_list = big_list       
          
    H = 0
    
    Jx = Jx * 2 * np.pi * np.ones(N)
    h = h * 2 * np.pi * np.ones(N)
    H += sum(-.5* h[n] * sz_list[n] for n in range(N))
    
    if (chain_type in spin_chain_type): 
        if chain_type == "XX":
            H += sum(-.5* Jx[n] *(sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]) for n in range(N-1))
            
        elif chain_type == "XXX":
            H += sum(-.5* Jx[n] * (sx_list[n]*sx_list[n+1] 
                                 + sy_list[n]*sy_list[n+1]
                                 + sz_list[n]*sz_list[n+1]) for n in range(N-1))
        
        elif chain_type == "XXZ":
            Jz = Jz * 2 * np.pi * np.ones(N)
            H += sum(-.5 * Jx[n] * (sx_list[n] * sx_list[n+1] + sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz[n] * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
        
        elif chain_type == "XYZ":
            Jy = Jy * 2 * np.pi * np.ones(N)
            Jz = Jz * 2 * np.pi * np.ones(N)
            H += sum(-.5 * Jx[n] * (sx_list[n] * sx_list[n+1])
                     -.5 * Jy[n] * (sy_list[n] * sy_list[n+1]) 
                     -.5 * Jz[n] * (sz_list[n] * sz_list[n+1]) for n in range(N-1))
    else:
        sys.exit("Currently not supported chain type")
              
    if visualization:
        qutip.hinton(H)
              
    return H
    
# In [8]:

def classical_ops(big_list, chain_type, N, Jx, Jy, Jz, h):
    H_H = Heisenberg_Hamiltonian(big_list, chain_type, N, False, Jx, Jy, Jz, h)
    sz_list = big_list[3]
        
    loc_x_op = sum((.5 + sz_list[a])*(a+1) for a in range(N))
    loc_p_op = 1j * (loc_x_op*H_H - H_H*loc_x_op)
    loc_comm_xp = .5*(loc_x_op*loc_p_op + loc_p_op*loc_x_op)
    loc_corr_xp = -1j*(loc_x_op*loc_p_op - loc_p_op*loc_x_op)
    loc_p_dot = 1j*(H_H * loc_p_op - loc_p_op * H_H)
    
    return loc_x_op, loc_p_op, loc_comm_xp, loc_corr_xp, loc_p_dot
   
# In [9]: 

natural = tuple('123456789')

def n_body_basis(big_list, gr, N):
    basis = []
    globalid_list, sx_list, sy_list, sz_list = big_list       
        
    if (isinstance(gr,int) and str(gr) in natural):
        try:
            if (gr == 1):
                basis = globalid_list + sx_list + sy_list + sz_list
            elif (gr > 1):
                basis = [op1*op2 for op1 in n_body_basis(big_list, gr-1, N) for op2 in n_body_basis(big_list, 1, N)]
        except Exception as ex:
            basis = None
            print(ex)
    return basis

# In [10]:

def n_body_max_ent_state(big_list, gr, N, coeffs = list, build_all = True, visualization = False):
    K = 0; rho_loc = 0;
    
    loc_globalid = qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    
    globalid_list, sx_list, sy_list, sz_list = big_list       
    
    pauli_vec = [sx_list, sy_list, sz_list];
    
    if (gr == 1):
        try:
            K += sum(coeffs[n][m] *  one_body_spin_ops(N)[n][m] 
                                    for n in range(len(one_body_spin_ops(N)))
                                    for m in range(len(one_body_spin_ops(N)[n]))
                   ) 
            K += 10**-6 * loc_globalid
        except Exception as exme1:
            print(exme1, "Max-Ent 1 Failure")
            raise exme1
    elif (gr == 2): 
        try:
            K += sum(coeffs[n][m] * two_body_spin_ops(big_list, N, build_all)[n][m] 
                    for n in range(len(two_body_spin_ops(big_list, N, build_all)))
                    for m in range(len(two_body_spin_ops(big_list, N, build_all)[n]))
                   )
            K += 10**-6 * loc_globalid
        except Exception as exme2:
            print(exme2, "Max-Ent 2 Failure")
            raise exme2
    else:
        print('gr must be either 1 or 2')
    
    rho_loc = K.expm()
    rho_loc = rho_loc/rho_loc.tr()
    
    if is_density_op(rho_loc):
        pass
    else:  
        rho_loc = None 
        raise Exception("The result is not a density operator")
        
    if visualization: 
        qutip.hinton(rho_loc)
        
    return rho_loc 

#In [11]:

def prod_basis(b1, b2):
    return [qutip.tensor(b,s) for b in b1 for s in b2]

def scalar_prod(op1, op2, rho0 = None, HS_prod_modified = True):
    if op1.dims[0][0]==op2.dims[0][0]:
        pass
    else:
        raise Exception("Incompatible Qobj dimensions")
    if rho0 is None:
        rho0 = qutip.qeye(op1.dims[0])/op1.dims[0][0]
    if (HS_prod_modified):  
        result = .5*(rho0*(op1*op2.dag()+op2.dag()*op1)).tr()
    else: 
        result = .5*(rho0*(op1.dag()*op2)).tr()
    return result.real

def base_orth(ops, rho0):
    if isinstance(ops[0], list):
        ops = [op for op1l in ops for op in op1l]
    dim = ops[0].dims[0][0]
    basis = []
    for i, op in enumerate(ops): 
        alpha = [scalar_prod(op2, op, rho0) for op2 in basis]
        op_mod = op - sum([c*op2 for c, op2, in zip(alpha, basis)])
        op_norm = np.sqrt(scalar_prod(op_mod,op_mod,rho0))
        if op_norm<1.e-12:
            continue
        op_mod = op_mod/(op_norm)
        basis.append(op_mod)
    return basis

def logM(rho):
    if ev_checks(rho):
        pass
    else:
        raise Exception("Singular input matrix")
    eigvals, eigvecs = rho.eigenstates()
    return sum([np.log(vl)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)])

def sqrtM(rho):
    if qutip.isherm(rho) and ev_checks(rho):
        pass
    else:
        raise Exception("Non-hermitian or singular input matrix")
    eigvals, eigvecs = rho.eigenstates()
    return sum([(vl**.5)*vc*vc.dag() for vl, vc in zip(eigvals, eigvecs)])

def proj_op(K, basis, rho0):
    return sum([scalar_prod(b, K,rho0) * b for b in basis])

def rel_entropy(rho, sigma):
    if (ev_checks(rho) and ev_checks(sigma)):
        pass
    else:
        raise Exception("Either rho or sigma non positive")
    
    val = (rho*(logM(rho)-logM(sigma))).tr()
                    
    if (abs(val.imag)>1.e-6):
        val = None
        raise Exception("Either rho or sigma not positive")
    return val.real
                
# In [12]:

def bures(rho, sigma):
    if is_density_op(rho) and is_density_op(sigma):
        val = abs((sqrtM(rho)*sqrtM(sigma)).tr())
        val = max(min(val,1.),-1.)
    return np.arccos(val)/np.pi
        
def maxent_rho(rho, basis):   
    def test(x, rho, basis):
        k = sum([-u*b for u,b in zip(x, basis)])        
        sigma = (.5*(k+k.dag())).expm()
        sigma = sigma/sigma.tr()
        return rel_entropy(rho, sigma)    
    res = opt.minimize(test,zeros(len(basis)),args=(rho,basis))
    k = sum([-u*b for u,b in zip(res.x, basis)])        
    sigma = (.5*(k+k.dag())).expm()
    sigma = sigma/sigma.tr()
    return sigma
 
def error_maxent_state(rho, basis, distance=bures):
    try:
        sigma = maxent_rho(rho, basis)
        return distance(rho,sigma)
    except:
        print("fail error max-ent state")
        return None
       
def error_proj_state(rho, rho0, basis, distance=bures):
    try:
        basis = base_orth(basis, rho0)
    except:
        print("orth error")
        raise
    try:
        sigma = proj_op(logM(rho), basis, rho0).expm()
        sigma = (sigma+sigma.dag())/(2.*sigma.tr())
    except:
        print("gram error")
    try:
        return distance(rho, sigma)
    except:
        print("fail error proj state")
        return None
    
# In [13]:

def spin_dephasing(big_list, N, gamma):
        loc_c_op_list = []; 
        loc_sz_list = big_list[3]
        
        collapse_weights = abs(gamma) * np.ones(N)
        loc_c_op_list = [np.sqrt(collapse_weights[n]) * loc_sz_list[n] for n in range(N)]
    
        return loc_c_op_list

# In [14]:

def initial_state(big_list, N = 1, gaussian = True, gr = 1, x = .5, coeffs = list, psi0 = qutip.Qobj,
                  build_all = False, visualization=False):
    loc_globalid = qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    if gaussian: 
        rho0 = n_body_max_ent_state(big_list, gr, N, coeffs, build_all, False)
    else:
        if (qutip.isket(psi0)):
            rho0 = psi0 * psi0.dag()
            rho0 = x * rho0 + (1-x)*loc_globalid * x/N
            rho0 = rho0/rho0.tr()
        else:
            print("Psi0 must be a ket")
    
    if is_density_op(rho0):
        pass
    else: 
        rho0 = None
        print("Output is not a density operador")
    
    if visualization:
            qutip.hinton(rho0)
    
    return rho0     

# In [15]: 

def commutator(A, B, visualization):
    if A.dims[0][0] == B.dims[0][0]:
        pass
    else:
        sys.exit("Currently not supported chain type")
    
    result = A*B-B*A
    
    if visualization:
        qutip.hinton(result)
    return result

def nmr_initial_state(N, freq, temp, visualization):
    
    sz_list = me.one_body_spin_ops(N)[3]
    rho = 0
    
    rho += qutip.tensor([qutip.qeye(2) for k in range(N)]) 
    rho += freq/temp * sum(sz_list[j] for j in range(N))
    
    if visualization:
        qutip.hinton(rho)
    return rho
