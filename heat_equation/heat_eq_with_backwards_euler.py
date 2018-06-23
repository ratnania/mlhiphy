
# coding: utf-8

# ## Heat Equation
# #### Parameter estimation for the heat equation (no source) using Gaussian processes (Backward Euler scheme)
# 
# 
# #### Problem Setup
# 
# $u_t - \alpha u_{xx} = 0$
# 
# $u(x,t) = e^{-t}sin(\frac{x}{\sqrt{\alpha}})$
# 
# Using the backward Euler scheme, the equation can be re-written as:
# 
# $\frac{u_n - u_{n-1}}{\tau} - \alpha \frac{d^2}{dx^2}u_n = 0$
# 
# and so:
# 
# $u_n - \tau \alpha \frac{d^2}{dx^2}u_n = u_{n-1}$
# 
# 
# Consider $u_n$ to be a Gaussian processes.
# 
# $u_n \sim \mathcal{GP}(x_i, x_j, \theta)$
# 
# And the linear operator:
# 
# $\mathcal{L}_x^\alpha = \cdot - \tau \alpha \frac{d^2}{dx^2}\cdot$
# 
# so that
# 
# $\mathcal{L}_x^\alpha u_n = u_{n-1}$
# 
# Problem at hand: estimate $\alpha$.
# 
# For the sake of simplicity, take $u := u_n$ and $f := u_{n-1}$.
# 
# 
# #### step 1: Simulate data
# 
# Take data points at $t = 0$ for $(u_{n-1})$ and $t = \tau$ for $(u_n)$, where $\tau$ is the time step.
# 
# $\alpha = 1$ and $x \in (0,2\pi)$.

# In[1]:

import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[2]:

tau = 0.02
# x_u = np.linspace(0,2*np.pi,10)
x_u = np.random.rand(10)*2*np.pi
y_u = np.exp(-tau)*np.sin(x_u)
# x_f = np.linspace(0,2*np.pi, 10)
x_f = x_u
y_f = np.sin(x_f)


# In[3]:

# plt.plot(x_u, y_u)
# plt.show()


# In[4]:

# plt.plot(x_f, y_f)
# plt.show()


# #### Step 2:Evaluate kernels
# 
# $k_{nn}(x_i, x_j; \theta) = \theta exp(-\frac{1}{2}(x_i-x_j)^2)$

# In[5]:

x_i, x_j, theta, alpha = sp.symbols('x_i x_j theta alpha')
kuu_sym = theta**2*sp.exp(-1/(2)*((x_i - x_j)**2))
kuu_fn = sp.lambdify((x_i, x_j, theta), kuu_sym, "numpy")
def kuu(x, theta):
    k = np.zeros((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            k[i,j] = kuu_fn(x[i], x[j], theta)
    return k


# $k_{ff}(x_i,x_j;\theta,\phi) \\
# = \mathcal{L}_{x_i}^\phi \mathcal{L}_{x_j}^\phi k_{uu}(x_i, x_j; \theta) \\
# = \mathcal{L}_{x_i}^\phi \left( k_{uu} - \tau \alpha \frac{\partial^2}{\partial x_j^2}k_{uu} \right) \\
#  = k_{uu} - \tau \alpha \left( \frac{\partial^2}{\partial x_j^2} + \frac{\partial^2}{\partial x_i^2} \right)k_{uu} + \tau^2 \alpha^2 \frac{\partial^2}{\partial x_i^2}\frac{\partial^2}{\partial x_j^2}k_{uu}$

# In[6]:

kff_sym = kuu_sym         - tau*alpha*(sp.diff(kuu_sym, x_j, x_j)         + sp.diff(kuu_sym, x_i, x_i))         + tau**2*alpha**2*sp.diff(kuu_sym, x_j, x_j, x_i, x_i)
kff_fn = sp.lambdify((x_i, x_j, theta, alpha), kff_sym, "numpy")
def kff(x, theta, alpha):
    k = np.zeros((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            k[i,j] = kff_fn(x[i], x[j], theta, alpha)
    return k


# $k_{fu}(x_i,x_j;\theta,\phi) \\
# = \mathcal{L}_{x_i}^\alpha k_{uu}(x_i, x_j; \theta) \\
# = k_{uu} - \tau \alpha \frac{\partial^2}{\partial x_i^2}k_{uu} $

# In[7]:

kfu_sym = kuu_sym - tau*alpha*sp.diff(kuu_sym, x_i, x_i)
kfu_fn = sp.lambdify((x_i, x_j, theta, alpha), kfu_sym, "numpy")
def kfu(x1, x2, theta, alpha):
    k = np.zeros((x1.size, x2.size))
    for i in range(x1.size):
        for j in range(x2.size):
            k[i,j] = kfu_fn(x1[i], x2[j], theta, alpha)
    return k


# In[8]:

def kuf(x1, x2, theta, alpha):
    return kfu(x1,x2,theta,alpha).T


# #### Step 3: Compute NLML

# In[9]:

def nlml(params, x1, x2, y1, y2, s):
    K = np.block([
        [kuu(x1, params[0]) + s*np.identity(x1.size), kuf(x1, x2, params[0], params[1])],
        [kfu(x1, x2, params[0], params[1]), kff(x2, params[0], params[1]) + s*np.identity(x2.size)]
    ])
    y = np.concatenate((y1, y2))
    val = 0.5*(np.log(abs(np.linalg.det(K))) + np.mat(y) * np.linalg.inv(K) * np.mat(y).T)
    return val.item(0)


# In[10]:

nlml((1, 2), x_u, x_f, y_u, y_f, 1e-6)


# Callback function for more details in output
Nfeval = 1

def callbackF(params):
    global Nfeval
    print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, params[0], params[1], nlml(params, x_u, x_f, y_u, y_f, 1e-6))
    Nfeval += 1


# #### Step 4: Optimise hyperparameters

# In[14]:

print  '{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', 'theta', ' alpha', 'nlml(params)')

res = minimize(nlml, np.random.rand(2), args=(x_u, x_f, y_u, y_f, 1e-6), method="Nelder-Mead", callback=callbackF)
print res
print "alpha is", res.x[1]


# ### Some analysis [to-do]

# Predicted values of $\alpha$ against time steps.
