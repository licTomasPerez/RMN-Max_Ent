{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9c998351",
   "metadata": {},
   "outputs": [],
   "source": [
    "import qutip, pickle, sys\n",
    "import matplotlib.pyplot as plt \n",
    "import numpy as np\n",
    "import scipy.optimize as opt \n",
    "import scipy.linalg as linalg\n",
    "import time as time\n",
    "import math, cmath\n",
    "import Max_Ent_aux_lib as me\n",
    "#import proj_ev_library as projev\n",
    "#import max_entev library as meev\n",
    "from IPython.display import display, Math, Latex\n",
    "\n",
    "np.set_printoptions(threshold=1.e-3,linewidth=120,precision=1, formatter={\"float\":lambda x: str(.001*int(1000*x)) })"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6413887b",
   "metadata": {},
   "source": [
    "Tenemos la siguiente ecuación de movimiento:\n",
    "\n",
    "$$\n",
    "    \\dot{c}_M(t) I_{M}^{0}(t) = -i \\bigg\\{\\bigg(c_{M+2}(t) - c_M(t) \\bigg)\\langle \\rho_{M}^{0\\dagger}(t) [H_{0,-2}, \\rho_{M+2}^0] \\rangle + \\bigg(c_{M-2}(t) - c_M(t)\\bigg) \\langle \\rho_{M}^{0\\dagger}(t) [H_{0,2}, \\rho_{M-2}^0] \\rangle + p \\langle \\rho_{M}^{0\\dagger}(t) [\\Sigma, \\rho_{M}^0 (t)]\\rangle \\bigg\\}\n",
    "$$\n",
    "\n",
    "siendo $I_{M}^{0}(t) = \\langle \\rho_{M}^{0\\dagger}(t) \\rho_{M}^{0}(t)\\rangle = I_{-M}^{0}(t)$\n",
    "la cual puede reescribirse como \n",
    "\n",
    "$$\n",
    "\\dot{c}_M(t) I_{M}^{0}(t) = -i \\bigg\\{A_{\\rho^{M, M+2}} c_{M+2}(t) - \\bigg(A_{\\rho^{M, M+2}} + B_{\\rho^{M, M-2}} \\bigg) c_{M}(t) + B_{\\rho^{M, M-2}} c_{M-2}(t) \\bigg\\} + p C_{\\rho^M} \\\\\n",
    "= - i({\\bf{\\mathcal{M}}}(t) \\textbf{c}(t))_M\n",
    "$$\n",
    "\n",
    "donde \n",
    "\n",
    "<ol>\n",
    "\n",
    "<li> $A_{\\rho^{M, M+2}} = \\langle \\rho_{M}^{0\\dagger}(t) [H_{0,-2}, \\rho_{M+2}^0] \\rangle$ </li> \n",
    "<li> $B_{\\rho^{M, M-2}} = \\langle \\rho_{M}^{0\\dagger}(t) [H_{0,2}, \\rho_{M-2}^0] \\rangle$ </li>\n",
    "<li> $C_{\\rho^M} = \\langle \\rho_{M}^{0\\dagger}(t) [\\Sigma, \\rho_{M}^0 (t)]\\rangle$ </li>\n",
    "    \n",
    "</ol>    \n",
    "\n",
    "Entonces, si $f(t) = \\langle \\rho(t) \\rho^0(t) \\rangle = \\sum_{M} \\dot{c}_M(t) I_{M}^{0}(t)$\n",
    "\n",
    "$$\n",
    "    \\dot{f}(t) = \\sum_{M} \\bigg(\\dot{c}_M(t) I_{M}^{0}(t) + {c}_M(t) \\dot{I}_{M}^{0}(t) \\bigg)\n",
    "$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9492c01e",
   "metadata": {},
   "source": [
    "where\n",
    "\n",
    "1: $A=\\alpha B = N(t) \\exp(-m t^{1+a}) = I_M^{(0)}$ (a primer orden alpha = 1) \\\n",
    "2: $C = M N(t) exp(-m t^{1+a})$\n",
    "\n",
    "con $N(t) = 1/Tr(\\rho(t))$ la calculo con la definición del $\\rho_M (t)=  c_{M}(t) \\rho^0_M(t)$, with $c_0(t)$= 1. \n",
    "Note that the kernel $K(t,t') = e^{(-i M(t-t'))}$ is not a solution to the previous differential equation for the M-tensor is time-dependent\n",
    "    \n",
    "$H_ {0, 2}^\\dagger = H_{0, -2}$ \n",
    "\n",
    "$\\rho_{M}^\\dagger = \\rho_{-M}$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a641b494",
   "metadata": {},
   "source": [
    "## Step 1: Fix parameters and initial conditions for the coherences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ea71fc99",
   "metadata": {},
   "outputs": [],
   "source": [
    "p = .0009;    # strength of the Sigma Interaction Hamiltonian\n",
    "a = -.05;      # Power-law factor\n",
    "M = 3;        # Truncation/Total no. of coherences\n",
    "coherences_t0_pert0 = 1. # Unused for the time being\n",
    "param_list = {\"total_no_cohrs\": M, \"p_factor\": p, \"power_law_factor\": a} # dictionary containing the simulation's initial \n",
    "                                                                         # parameters \n",
    "\n",
    "cohr_complex_t0 = [coherences_t0_pert0 - np.random.rand() for i in range(param_list[\"total_no_cohrs\"])] \n",
    "    # initial configuration of complex-valued coherences, random numbers for the time being. Can be changed in the future\n",
    "\n",
    "cohr_complex_t0 += [0 for i in range(param_list[\"total_no_cohrs\"])]\n",
    "    # normalization of the initial vector, so that Tr c_0 = 1. \n",
    "cohr_t0_trace = sum(cohr_complex_t0)\n",
    "cohr_complex_t0 = [cohr/cohr_t0_trace for cohr in cohr_complex_t0]\n",
    "cohr_complex_t0 = np.array(cohr_complex_t0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10e67289",
   "metadata": {},
   "source": [
    "Since the coherences are complex-valued numbers, the previous system of $M \\times M$ coupled complex-valued differential equations can be rewritten as a system of $2M \\times 2M$ coupled real-valued differential equations, as follows:\n",
    "\n",
    "if $c_M(t) = a_M(t) + i b_M(t)$, then \n",
    "\n",
    "$$\n",
    "    \\dot{a}_M(t) + i \\dot{b}_M(t) = -i \\sum_{m'}{\\bf{\\mathcal{M}}}_{Mm'}(t) \\bigg(a_{m'}(t) + i b_{m'}(t)\\bigg),\n",
    "$$\n",
    "\n",
    "$$\n",
    "    \\dot{a}_M(t) + i \\dot{b}_M(t) = -i \\sum_{m'}{\\bf{\\mathcal{M}}}_{Mm'}(t) a_{m'}(t) + \\sum_{m'} {\\bf{\\mathcal{M}}}_{Mm'}(t) b_{m'}(t)\n",
    "$$\n",
    "\n",
    "$$\n",
    "    \\dot{{\\bf a}}(t) = {\\cal M}(t) \\cdot {\\bf b}(t), \\quad \\dot{{\\bf b}}(t) = -{\\cal M}(t) \\cdot {\\bf a}(t), \\qquad s.t. \\quad{\\bf c}(t) = {\\bf a}(t) + i {\\bf b}(t)\n",
    "$$\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "afc249cc",
   "metadata": {},
   "source": [
    "## Step 2: Setting up and Solving system of complex diff. eqs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "89f9074c",
   "metadata": {},
   "outputs": [],
   "source": [
    "param_list = {\"total_no_cohrs\": M, \"p_factor\": p, \"power_law_factor\": a} # dictionary containing the simulation's initial \n",
    "                                                                         # parameters \n",
    "\n",
    "cohr_complex_t0 = [coherences_t0_pert0 - np.random.rand() for i in range(param_list[\"total_no_cohrs\"])]\n",
    "cohr_complex_t0 += [0 for i in range(param_list[\"total_no_cohrs\"])]\n",
    "cohr_complex_t0 = np.array(cohr_complex_t0)\n",
    "param_list = [M, p, a]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ec08f283",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "B = me.complex_differential_system(parameters = param_list, cohr_complex = cohr_complex_t0, t=1)\n",
    "len(B)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "1bf86fb7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.20500000000000002, 0.01, 0.251, 0.0, 0.0, 0.0],\n",
       "       [0.20500000000000002, 0.01, 0.251, 0.0, 0.003, 0.001],\n",
       "       [0.20500000000000002, 0.01, 0.251, 0.0, 0.006, 0.002],\n",
       "       ...,\n",
       "       [0.325, 0.032, 0.34600000000000003, 0.04, 0.322, 0.132],\n",
       "       [0.328, 0.033, 0.34800000000000003, 0.041, 0.326, 0.133],\n",
       "       [0.331, 0.033, 0.35000000000000003, 0.042, 0.33, 0.135]])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from scipy.integrate import odeint\n",
    "ts = np.linspace(0.1, 1., 100)              ## times \n",
    "time_ev_ReIm_cohrs = odeint(func = me.complex_differential_system, \n",
    "                y0 = cohr_complex_t0, \n",
    "                t = ts,\n",
    "                args = ((param_list,)))\n",
    "#  tiempo vertical, primera linea es la config init: t  cohr\n",
    "time_ev_ReIm_cohrs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "70c4d158",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "100"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(time_evolved_real_imag_parts_cohrs) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "5781009a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(time_evolved_real_imag_parts_cohrs[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62cb6ca9",
   "metadata": {},
   "source": [
    "## Step 3: Obtaining the time-evolved complex-valued coherences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "377f28d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "cohr1_t = [time_ev_ReIm_cohrs[t][0] + 1j * time_ev_ReIm_cohrs[t][0 + M] for t in range(100)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b5536492",
   "metadata": {},
   "outputs": [],
   "source": [
    "vec_cohr_t = [[time_ev_ReIm_cohrs[t][m] + 1j * time_ev_ReIm_cohrs[t][m + M] for t in range(100)] for m in range(M)]\n",
    "# prime parámetro coherencia\n",
    "# segundo parámetro tiempo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "15848b77",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'fidelity_vs_t' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [56]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mfidelity_vs_t\u001b[49m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'fidelity_vs_t' is not defined"
     ]
    }
   ],
   "source": [
    "fidelity_t0 = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "0f0b1404",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'result' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [55]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m fidelity_vs_t \u001b[38;5;241m=\u001b[39m [[\u001b[38;5;28msum\u001b[39m(result[:, m][\u001b[38;5;28mint\u001b[39m(\u001b[38;5;28mlist\u001b[39m(ts)\u001b[38;5;241m.\u001b[39mindex(t)) \n\u001b[0;32m      2\u001b[0m                                    \u001b[38;5;241m*\u001b[39m \u001b[38;5;241m-\u001b[39mm \u001b[38;5;241m*\u001b[39m (a\u001b[38;5;241m+\u001b[39m\u001b[38;5;241m1\u001b[39m) \u001b[38;5;241m*\u001b[39m t\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39ma \u001b[38;5;241m/\u001b[39m (me\u001b[38;5;241m.\u001b[39mA_mmplustwo_matrix_elmt(m, t, a))] \u001b[38;5;28;01mfor\u001b[39;00m m \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(M))] \u001b[38;5;28;01mfor\u001b[39;00m t \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mlist\u001b[39m(ts)]\n",
      "Input \u001b[1;32mIn [55]\u001b[0m, in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[1;32m----> 1\u001b[0m fidelity_vs_t \u001b[38;5;241m=\u001b[39m [[\u001b[38;5;28;43msum\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mresult\u001b[49m\u001b[43m[\u001b[49m\u001b[43m:\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mm\u001b[49m\u001b[43m]\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;28;43mint\u001b[39;49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mlist\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mts\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mindex\u001b[49m\u001b[43m(\u001b[49m\u001b[43mt\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\u001b[43m \u001b[49m\n\u001b[0;32m      2\u001b[0m \u001b[43m                                   \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m \u001b[49m\u001b[38;5;241;43m-\u001b[39;49m\u001b[43mm\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[43ma\u001b[49m\u001b[38;5;241;43m+\u001b[39;49m\u001b[38;5;241;43m1\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m \u001b[49m\u001b[43mt\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43ma\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m/\u001b[39;49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[43mme\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mA_mmplustwo_matrix_elmt\u001b[49m\u001b[43m(\u001b[49m\u001b[43mm\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mt\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43ma\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\u001b[43m]\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mm\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;28;43mrange\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mM\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m] \u001b[38;5;28;01mfor\u001b[39;00m t \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mlist\u001b[39m(ts)]\n",
      "Input \u001b[1;32mIn [55]\u001b[0m, in \u001b[0;36m<genexpr>\u001b[1;34m(.0)\u001b[0m\n\u001b[1;32m----> 1\u001b[0m fidelity_vs_t \u001b[38;5;241m=\u001b[39m [[\u001b[38;5;28msum\u001b[39m(\u001b[43mresult\u001b[49m[:, m][\u001b[38;5;28mint\u001b[39m(\u001b[38;5;28mlist\u001b[39m(ts)\u001b[38;5;241m.\u001b[39mindex(t)) \n\u001b[0;32m      2\u001b[0m                                    \u001b[38;5;241m*\u001b[39m \u001b[38;5;241m-\u001b[39mm \u001b[38;5;241m*\u001b[39m (a\u001b[38;5;241m+\u001b[39m\u001b[38;5;241m1\u001b[39m) \u001b[38;5;241m*\u001b[39m t\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39ma \u001b[38;5;241m/\u001b[39m (me\u001b[38;5;241m.\u001b[39mA_mmplustwo_matrix_elmt(m, t, a))] \u001b[38;5;28;01mfor\u001b[39;00m m \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(M))] \u001b[38;5;28;01mfor\u001b[39;00m t \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mlist\u001b[39m(ts)]\n",
      "\u001b[1;31mNameError\u001b[0m: name 'result' is not defined"
     ]
    }
   ],
   "source": [
    "fidelity_vs_t = [[sum(result[:, m][int(list(ts).index(t)) \n",
    "                                   * -m * (a+1) * t**a / (me.A_mmplustwo_matrix_elmt(m, t, a))] for m in range(M))] for t in list(ts)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "ec4d586e",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "100"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(result[:, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "067ec940",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(result[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07d78449",
   "metadata": {},
   "source": [
    "# Tests y fallos :("
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "781be6e4",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (3991330021.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Input \u001b[1;32mIn [23]\u001b[1;36m\u001b[0m\n\u001b[1;33m    Preguntas:\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "Preguntas:\n",
    "    \n",
    "    1. La matrix M tiene que depender con el tiempo me parece, si A, B y C lo hacen: Check \n",
    "    2. Como tendría que implementar la norma???: Check\n",
    "    3. Empezar a jugar con los parámetros: Checkn't\n",
    "    \n",
    "m+2 vs m+2 : Check\n",
    "** reescribir la matriz en tèrminos de los c_pares. : Check"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
