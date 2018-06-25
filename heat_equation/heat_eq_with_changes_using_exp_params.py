
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


# In[2]:


n1 = 5
n2 = 5
np.random.seed(int(time.time()))
t_u, x_u = np.random.rand(n1), np.random.rand(n1)
t_f, x_f = np.random.rand(n2), np.random.rand(n2)


# In[3]:


#y_u = 2*np.square(x_u) + np.multiply(x_u, t_u) + np.random.normal(0,1, x_u.size)
#y_f = x_f - 4*12.0 + np.random.normal(0,1, x_f.size)

y_u = np.multiply(np.exp(-t_u), np.sin(2*np.pi*x_u))
y_f = (4*np.pi**2 - 1) * np.multiply(np.exp(-t_f), np.sin(2*np.pi*x_f))


# #### Step 2: evaluate kernels and covariance matrix
# 
# Declare symbols

# In[4]:


x_i, x_j, t_i, t_j, sig_u, l_u, phi = sp.symbols('x_i x_j t_i t_j sig_u l_u phi')


# $k_{uu}(\bar{x}_i, \bar{x}_j; \theta) = \sigma_u^2 exp(-\frac{1}{2l_u}\left[(x_i-x_j)^2 +(t_i-t_j)^2\right])$

# In[5]:


k_uu_sym = sig_u**2*sp.exp(-1/(2*sp.exp(l_u)**2)*((x_i - x_j)**2 + (t_i - t_j)**2))
k_uu_fn = sp.lambdify((x_i, x_j, t_i, t_j, sig_u, l_u), k_uu_sym, "numpy")
def kuu(t, x, sigma, l):
    k = np.zeros((t.size, t.size))
    for i in range(t.size):
        for j in range(t.size):
            k[i,j] = k_uu_fn(x[i], x[j], t[i], t[j], sigma, l)
    return k


# $k_{ff}(\bar{x}_i,\bar{x}_j;\theta,\phi) \\
# = \mathcal{L}_{\bar{x}_i}^\phi \mathcal{L}_{\bar{x}_j}^\phi k_{uu}(\bar{x}_i, \bar{x}_j; \theta) \\
# = \mathcal{L}_{\bar{x}_i}^\phi \left[ \frac{\partial}{\partial t_j}k_{uu} - \phi \frac{\partial^2}{\partial x_j^2} k_{uu} \right] \\
# = \frac{\partial}{\partial t_i}\frac{\partial}{\partial t_j}k_{uu} - \phi \left[ \frac{\partial}{\partial t_i}\frac{\partial^2}{\partial x_j^2}k_{uu} + \frac{\partial^2}{\partial x_i^2}\frac{\partial}{\partial t_j}k_{uu} \right] + \phi^2 \frac{\partial^2}{\partial x_i^2}\frac{\partial^2}{\partial x_j^2}k_{uu}$

# In[6]:


k_ff_sym = sp.diff(k_uu_sym, t_j, t_i)         - phi*sp.diff(k_uu_sym,x_j,x_j,t_i)         - phi*sp.diff(k_uu_sym,t_j,x_i,x_i)         + phi**2*sp.diff(k_uu_sym,x_j,x_j,x_i,x_i)
k_ff_fn = sp.lambdify((x_i, x_j, t_i, t_j, sig_u, l_u, phi), k_ff_sym, "numpy")
def kff(t, x, sigma, l, p):
    k = np.zeros((t.size, t.size))
    for i in range(t.size):
        for j in range(t.size):
            k[i,j] = k_ff_fn(x[i], x[j], t[i], t[j], sigma, l, p)
    return k


# $k_{fu}(\bar{x}_i,\bar{x}_j;\theta,\phi) \\
# = \mathcal{L}_{\bar{x}_i}^\phi k_{uu}(\bar{x}_i, \bar{x}_j; \theta) \\
# = \frac{\partial}{\partial t_i}k_{uu} - \phi \frac{\partial^2}{\partial x_i^2}k_{uu}$

# In[7]:


k_fu_sym = sp.diff(k_uu_sym,t_i) - phi*sp.diff(k_uu_sym,x_i,x_i)
k_fu_fn = sp.lambdify((x_i, x_j, t_i, t_j, sig_u, l_u, phi), k_fu_sym, "numpy")
def kfu(t1, x1, t2, x2, sigma, l, p):
    k = np.zeros((t1.size, t2.size))
    for i in range(t1.size):
        for j in range(t2.size):
            k[i,j] = k_fu_fn(x1[i], x2[j], t1[i], t2[j], sigma, l, p)
    return k


# In[8]:


k_uf_sym = sp.diff(k_uu_sym,t_j) - phi*sp.diff(k_uu_sym,x_j,x_j)
k_uf_fn = sp.lambdify((x_i, x_j, t_i, t_j, sig_u, l_u, phi), k_uf_sym, "numpy")
def kuf(t1, x1, t2, x2, sigma, l, p):
    k = np.zeros((t2.size, t1.size))
    for i in range(t2.size):
        for j in range(t1.size):
            k[i,j] = k_uf_fn(x2[i], x1[j], t2[i], t1[j], sigma, l, p)
    return k


# #### Step 3: create covariance matrix and NLML
# 
# ```
# params = [sig_u, l_u, phi]
# ```

# In[9]:


def nlml(params, t1, x1, y1, t2, x2, y2, s):
    K = np.block([
        [
            kuu(t1, x1, params[0], params[1]) + s*np.identity(x1.size),
            kuf(t1, x1, t2, x2, params[0], params[1], params[2])
        ],
        [
            kfu(t1, x1, t2, x2, params[0], params[1], params[2]),
            kff(t2, x2, params[0], params[1], params[2]) + s*np.identity(x2.size)
        ]
    ])
    y = np.concatenate((y1, y2))
    val = 0.5*(np.log(abs(np.linalg.det(K))) + np.mat(y) * np.linalg.inv(K) * np.mat(y).T)
    return val.item(0)


# In[10]:


nlml((1,1,1), t_u, x_u, y_u, t_f, x_f, y_f, 1e-6)

res = minimize(nlml, (0.5,0.5,1), args=(t_u, x_u, y_u, t_f, x_f, y_f, 1e-6), method="Nelder-Mead")

# In[11]:

# sig_array = np.zeros(15)
# l_array = np.zeros(15)
# phi_array = np.zeros(15)

# sig_array = np.zeros(15) - 1
# l_array = np.zeros(15) - 1
# phi_array = np.zeros(15) - 1
# fun_array = np.zeros(15)

# for i in range(15):
#   try:
#     res = minimize(nlml, (0.5,0.5,1), args=(t_u, x_u, y_u, t_f, x_f, y_f, 1e-3), method="Nelder-Mead")
#     sig_array[i] = res.x[0]
#     l_array[i] = np.exp(res.x[1])
#     phi_array[i] = res.x[2]
#     fun_array[i] = res.fun
# except:
#   continue

# Return sigma and np.exp(l)!
# Harsha set the first parameter to be sigma!
print("Sigma equals %f", res.x[0])
print("l equals %f", np.exp(res.x[1]))
print("Phi equals %f", res.x[2])