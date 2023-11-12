# General problem definition

![image.png](attachment:image.png)

Consider a 1D heat conduction problem
$$ - \frac{\partial}{\partial x} k(x,T,\mu)\frac{\partial T}{\partial x} = q(x)$$
where the conductivity is a function of temperature. 

As an example, we can use
$$k(x,T,\mu)=k(x,\mu)$$
Hence, we are solving:
$$- \frac{\partial}{\partial x}k(x,\mu) \frac{\partial T}{\partial x} = q(x)
$$

Here the function $k(x,\mu)$ can be defined using the piecewise function notation

$$
k(x,\mu) = 
\begin{cases} 
1+\mu & \text{if } 0 \leq x \leq 0.4  \\
10|\sin(\mu)| & \text{if } 0.4\leq x \leq 0.5
\end{cases}
$$

Where as for the nonlinear problem we define $k(x,\mu)$ in the following way:

$$
k(x,\mu) = 
\begin{cases} 
1.05*mu + 2150/(T-73.15) & \text{if } 0 \leq x \leq 0.4  \\
7.51 + 2.09\times10^{-2}T - 1.45\times10^{-5}T^2 + 7.67\times10^{-9}T^3 & \text{if } 0.4\leq x \leq 0.5
\end{cases}
$$ 

Here $\mu\in[0,10]$

**Boundary conditions**:
- Reflective at $x=0$: $\quad \left.\tfrac{\partial T}{\partial x}\right|_{x=0} = 0$

- Imposed temperature (Dirichlet bc) at $x=L=0.5$: $\quad T(L)=T_{bc}=573.15K$

# Finding the weak form of the PDE

To derive the weak form of the given PDE, let's follow these steps:

1) Multiply the PDE by a test function $ v(x) $ and integrate over the domain $ \Omega $:

$$
-\int_{\Omega} v(x) \frac{\partial}{\partial x} \left( k(x,T,\mu) \frac{\partial T}{\partial x} \right) \, dx = \int_{\Omega} v(x) q(x) \, dx
$$

2) Apply integration by parts:

$$ 
-\int_{\Omega} v(x) \frac{\partial}{\partial x} \left( k(x,T,\mu) \frac{\partial T}{\partial x} \right) \, dx = v(x) \left( k(x,T,\mu) \frac{\partial T}{\partial x} \right) \Bigg|_{\partial\Omega} - \int_{\Omega} \frac{\partial v}{\partial x} k(x,T,\mu) \frac{\partial T}{\partial x} \,dx
$$


On the boundary $ \Gamma_D $, the test function $ v(x) $ is zero due to the Dirichlet condition. The boundary term, therefore, only contributes on $ \Gamma_N $:

$$
-\v(x) \left( k(x,T,\mu) \frac{\partial T}{\partial x} \right) \Bigg|_{\Gamma_N} = v(x) g(x) \Bigg|_{\Gamma_N}
$$  

3) The weak form then becomes:

$$
-\int_{\Omega} \frac{\partial v}{\partial x} k(x,T,\mu) \frac{\partial T}{\partial x} \, dx - v(x) g(x) \Bigg|_{\Gamma_N} = \int_{\Omega} v(x) q(x) \, dx
$$

This gives us the weak form for the PDE with both Neumann and Dirichlet boundary conditions. For this case, $v(x) g(x) \Bigg|_{\Gamma_N} = 0$ 

$$
\int_{\Omega} \frac{\partial v}{\partial x} k(x,T,\mu) \frac{\partial T}{\partial x} \, dx - \int_{\Omega} v(x) q(x) \, dx = 0
$$

 # ECSW Hyper-reduction

 Total virtual work-done should be preserved to the extent possible:
![Screenshot%202023-09-08%20151208.png](attachment:Screenshot%202023-09-08%20151208.png)

Consider there are multiple snapshots corresponding to different parameter values: $\large \{\mathbf{u}^{s}_{N_h}\} ^{\mathbf{n}_s}_{s=1}$. We can write down the virtual work done for these different cases, in a compact fashion, using the following notation:
![Screenshot%202023-09-08%20154728.png](attachment:Screenshot%202023-09-08%20154728.png)

## Step 1: Perform SVD on the snapshots (calculate $\mathbb{V}(=\mathbb{W}$)):

## Step-2 Calculate virtual workdone $\mathbf{d}$ corresponding to $s^{th}$ snapshot $\mathbf{u}^{s}_{N_h}\in\mathbb{R}^{N_h\times 1}$:


$$
\mathbf{c}^{s}_e\in\mathbb{R}^{N_h\times 1} := \mathbb{L}_{e}^{T}{\cal K}_e(\mu)\mathbb{L}_{e}\mathbb{\Pi}_{\mathbb{V}}\mathbf{u}^{s}_{N_h}(\mu,x) - \mathbb{L}_{e}^{T}\mathbf{q}^s_e(x)
$$

And
$$
\mathbf{b}^s_n\in\mathbb{R}^{n\times 1} := \sum_{e\in{\cal E}}\mathbb{V}^{T}\mathbf{c}^s_e
$$
This the the total virtual work done in the direction of $n$ column vectors of $\mathbb{V}$

## Step-3 calculate $\xi^{*}$:

To find a reduced mesh, we want the following:
![image.png](attachment:image.png)
 
