
# coding: utf-8

# ## Heat Eq (without discretisation)
# 
# 
# $\mathcal{L}_{\bar{x}}^{\phi}u(\bar{x}) = \frac{\partial}{\partial t}u(\bar{x}) - \phi \frac{\partial^2}{\partial x^2}u(\bar{x}) = f(\bar{x})$, where $\bar{x} = (t, x) \in \mathbb{R}^2$
# 
# $u(x,t) = e^{-t}sin(2\pi x)$
# 
# $f(x,t) = e^{-t}(4\pi^2 - 1)sin(2\pi x)$
# 
# 
# #### Step 1: simulate data

# In[1]:

import time
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import pyGPs


def get_heat_eqn_data(n = 10):
    t = np.random.rand(n)
    x = np.random.rand(n)
    y_u = np.multiply(np.exp(-t), np.sin(2*np.pi*x))
    y_f = (4*np.pi**2 - 1) * np.multiply(np.exp(-t), np.sin(2*np.pi*x))
    return (x,y_u,y_f)


# In[3]:



# #### Step 2: evaluate kernels and covariance matrix
# 
# Declare symbols

# In[4]:

x_i, x_j, t_i, t_j, theta, phi = sp.symbols('x_i x_j t_i t_j theta phi')


# $k_{uu}(x_i, x_j, t_i, t_j; \theta) = \theta exp \left[ -\frac{1}{2} \left( (x_i-x_j)^2 + (t_i-t_j)^2 \right) \right]$

# In[5]:

kuu_sym = sp.exp(-theta*((x_i - x_j)**2 + (t_i - t_j)**2))
kuu_fn = sp.lambdify((x_i, x_j, t_i, t_j, theta), kuu_sym, "numpy")
def kuu(t, x, theta):
    k = np.zeros((t.size, t.size))
    for i in range(t.size):
        for j in range(t.size):
            k[i,j] = kuu_fn(x[i], x[j], t[i], t[j], theta)
    return k


# $k_{ff}(\bar{x}_i,\bar{x}_j;\theta,\phi) \\
# = \mathcal{L}_{\bar{x}_i}^\phi \mathcal{L}_{\bar{x}_j}^\phi k_{uu}(\bar{x}_i, \bar{x}_j; \theta) \\
# = \mathcal{L}_{\bar{x}_i}^\phi \left[ \frac{\partial}{\partial t_j}k_{uu} - \phi \frac{\partial^2}{\partial x_j^2} k_{uu} \right] \\
# = \frac{\partial}{\partial t_i}\frac{\partial}{\partial t_j}k_{uu} - \phi \left[ \frac{\partial}{\partial t_i}\frac{\partial^2}{\partial x_j^2}k_{uu} + \frac{\partial^2}{\partial x_i^2}\frac{\partial}{\partial t_j}k_{uu} \right] + \phi^2 \frac{\partial^2}{\partial x_i^2}\frac{\partial^2}{\partial x_j^2}k_{uu}$

# In[6]:

kff_sym = sp.diff(kuu_sym, t_j, t_i)         - phi*sp.diff(kuu_sym,x_j,x_j,t_i)         - phi*sp.diff(kuu_sym,t_j,x_i,x_i)         + phi**2*sp.diff(kuu_sym,x_j,x_j,x_i,x_i)
kff_fn = sp.lambdify((x_i, x_j, t_i, t_j, theta, phi), kff_sym, "numpy")
def kff(t, x, theta, p):
    k = np.zeros((t.size, t.size))
    for i in range(t.size):
        for j in range(t.size):
            k[i,j] = kff_fn(x[i], x[j], t[i], t[j], theta, p)
    return k


# $k_{fu}(\bar{x}_i,\bar{x}_j;\theta,\phi) \\
# = \mathcal{L}_{\bar{x}_i}^\phi k_{uu}(\bar{x}_i, \bar{x}_j; \theta) \\
# = \frac{\partial}{\partial t_i}k_{uu} - \phi \frac{\partial^2}{\partial x_i^2}k_{uu}$

# In[7]:

kfu_sym = sp.diff(kuu_sym,t_i) - phi*sp.diff(kuu_sym,x_i,x_i)
kfu_fn = sp.lambdify((x_i, x_j, t_i, t_j, theta, phi), kfu_sym, "numpy")
def kfu(t, x, theta, p):
    k = np.zeros((t.size, t.size))
    for i in range(t.size):
        for j in range(t.size):
            k[i,j] = kfu_fn(x[i], x[j], t[i], t[j], theta, p)
    return k


# In[8]:

def kuf(t, x, theta, p):
    return kfu(t, x, theta, p).T


# #### Step 3: create covariance matrix and NLML
# 
# ```
# params = [sig_u, l_u, phi]
# ```

# In[9]:

def nlml(params, t, x, y1, y2, s):
    params = np.exp(params)
    K = np.block([
        [
            kuu(t, x, params[0]) + s*np.identity(x.size),
            kuf(t, x, params[0], params[1])
        ],
        [
            kfu(t, x, params[0], params[1]),
            kff(t, x, params[0], params[1]) + s*np.identity(x.size)
        ]
    ])
    y = np.concatenate((y1, y2))
    val = 0.5*(np.log(abs(np.linalg.det(K))) + np.mat(y) * np.linalg.inv(K) * np.mat(y).T)
    return val.item(0)


# In[10]:

nlml((1,1), t, x, y_u, y_f, 1e-6)


# In[11]:

get_ipython().run_cell_magic('timeit', '', 'nlml_wp = lambda params: nlml(params, t, x, y_u, y_f, 1e-7)\nminimize(nlml_wp, np.random.rand(2), method="Nelder-Mead", options={\'maxiter\' : 5000, \'fatol\' : 0.001})')


# In[12]:

def minimize_restarts(t, x, y_u, y_f, n = 10):
    nlml_wp = lambda params: nlml(params, t, x, y_u, y_f, 1e-7)
    all_results = []
    for it in range(0,n):
        all_results.append(minimize(nlml_wp, np.random.rand(2), method="Nelder-Mead", options={'maxiter' : 5000, 'fatol' : 0.001}))
    filtered_results = [m for m in all_results if 0 == m.status]
    return min(filtered_results, key = lambda x: x.fun)


# In[13]:

m = minimize_restarts(t, x, y_u, y_f, 20)
m


# ##### Estimated value of $\alpha$

# In[14]:

np.exp(m.x[1])

