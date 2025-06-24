import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import os
import json
from tqdm import tqdm


torch.manual_seed(0)
np.random.seed(0)

layers = [2, 50, 50, 50, 1]

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f"tanh_{i}", nn.Tanh())
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        return self.net(x)

def burgers_analytical_solution(xs, ts, nu=0.01, M=5, Ny=801):
    x_min, x_max = -1.0, 1.0

    # Precompute y-grid and weights
    ys = np.linspace(x_min, x_max, Ny, endpoint=False)
    dy = (x_max - x_min) / Ny

    # H(y) = ∫_{-1}^y u0(s) ds = (cos(pi y) + 1)/pi
    Hys = (np.cos(np.pi * ys) + 1.0) / np.pi

    # Weight N(y) = exp( -H(y)/(2 nu) )
    Nys = np.exp(-Hys / (2 * nu))

    # Allocate solution array
    U = np.zeros((len(ts), len(xs)))

    # Loop over time
    for it, t in tqdm(enumerate(ts), total=len(ts)):
        # accumulators for numerator and denominator integrals
        numer = np.zeros_like(xs)
        denom = np.zeros_like(xs)

        # Sum over periodic images
        for m in range(-M, M + 1):
            # Z_ij = x_i - y_j + 2m
            Z = xs[:, None] - ys[None, :] + 2 * m
            # heat kernel G(Z; t)
            G = np.exp(-Z**2 / (4 * nu * t)) / np.sqrt(4 * np.pi * nu * t)

            # accumulate weighted integrals
            # denom_i += ∑_j N(y_j) G_ij * dy
            denom += (Nys[None, :] * G).sum(axis=1) * dy
            # numer_i += ∑_j N(y_j) G_ij * Z_ij * dy
            numer += (Nys[None, :] * G * Z).sum(axis=1) * dy

        # u(x,t) = numer / (t * denom)
        U[it, :] = numer / (t * denom)

    return U

class HeatSolver:
    def __init__(self, u0, x, t, alpha=1.0):
        self.u0 = np.asarray(u0)

        self.x = x
        self.x0 = self.x[0]
        self.x1 = self.x[-1]
        self.nx = len(self.x)-1

        self.t = t
        self.nt = len(self.t)-1
        self.T = self.t[-1]-self.t[0]
        self.dt = self.T/self.nt
        
        self.alpha = alpha

        self.h = (self.x1 - self.x0) / self.nx
        self.nsteps = int(np.ceil(self.T / self.dt))

        self.M, self.K = self.assemble_matrices()
        self.A_fact = spla.factorized(self.M + self.alpha * self.dt * self.K)

        self.usol = None

    def assemble_matrices(self):
        main_M = 2/3 * self.h * np.ones(self.nx - 1)
        off_M = 1/6 * self.h * np.ones(self.nx - 2)
        M = sp.diags([off_M, main_M, off_M], offsets=[-1, 0, 1], format='csr')

        main_K = 2 / self.h * np.ones(self.nx - 1)
        off_K = -1 / self.h * np.ones(self.nx - 2)
        K = sp.diags([off_K, main_K, off_K], offsets=[-1, 0, 1], format='csr')

        return M, K

    def solve(self):
        usol = np.zeros((self.nsteps + 1, self.nx + 1))
        usol[0, :] = self.u0.copy()

        u_int = self.u0[1:-1].copy()

        for n in range(self.nsteps):
            rhs = self.M.dot(u_int)
            u_int = self.A_fact(rhs)
            u_full = np.zeros(self.nx + 1)
            u_full[1:-1] = u_int
            usol[n+1, :] = u_full

        self.usol = usol
        return usol
    
    def plot(self, usol=None, cmap='jet', figsize=(8,6)):
        if usol is None:
            usol = self.solve()
        X, Tm = np.meshgrid(self.x, self.t)
        plt.figure(figsize=figsize)
        plt.pcolormesh(X, Tm, usol, shading='auto', cmap=cmap)
        plt.colorbar(label='u')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Space-Time Diagram (alpha={self.alpha})')
        plt.tight_layout()
        plt.show()

class RusanovBurgersSolver:
    def __init__(self, u0, x, t, nu=0.01, bc_type='dirichlet'):
        self.x, self.t = np.asarray(x), np.asarray(t)
        self.nx, self.nt = len(self.x), len(self.t)
        self.L, self.T = abs(self.x[-1] - self.x[0]), abs(self.t[-1] - self.t[0])
        self.dx, self.dt = self.L/(self.nx-1), self.T/self.nt
        self.nu = nu

        # Storage
        self.u = np.zeros((self.nt, self.nx))
        self.u[0, :] = np.asarray(u0).flatten()

        # Boundary type
        self.bc_type = bc_type.lower()
        if self.bc_type not in ('dirichlet','neumann','periodic'):
            raise ValueError(f"Unknown bc_type '{bc_type}'")

    def flux(self, u):
        return 0.5*u**2

    def rusanov_flux(self, uL, uR):
        a = np.maximum(np.abs(uL), np.abs(uR))
        return 0.5*(self.flux(uL) + self.flux(uR)) - 0.5*a*(uR-uL)

    def solve(self):
        for n in range(self.nt-1):
            un = self.u[n].copy()
            F = np.zeros_like(un)

            # interior flux differences
            for i in range(1, self.nx-1):
                F[i] = ( self.rusanov_flux(un[i],   un[i+1])
                       - self.rusanov_flux(un[i-1], un[i]) )

            # update interior
            self.u[n+1, 1:-1] = (
                un[1:-1]
                - self.dt/self.dx * F[1:-1]
                + self.nu*self.dt/self.dx**2 * (un[2:] - 2*un[1:-1] + un[:-2])
            )

            # enforce boundaries
            if self.bc_type == 'dirichlet':
                # hold original boundary values
                self.u[n+1, 0]  = self.u[0, 0]
                self.u[n+1,-1] = self.u[0,-1]

            elif self.bc_type == 'neumann':
                # zero-gradient: mirror nearest interior
                self.u[n+1, 0]  = self.u[n+1, 1]
                self.u[n+1,-1] = self.u[n+1,-2]

            else:  # periodic
                # wrap-around
                self.u[n+1, 0]  = self.u[n+1,-2]
                self.u[n+1,-1] = self.u[n+1, 1]

        return self.u

    def plot(self, cmap='viridis', vmin=None, vmax=None):
        plt.figure(figsize=(12,6))
        im = plt.imshow(
            self.u,
            extent=[self.x[0], self.x[-1], self.t[0], self.t[-1]],
            aspect='auto', origin='lower',
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        plt.colorbar(im, label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title("Burgers' Equation Solution")
        plt.tight_layout()
        plt.show()

class PINN:
    def __init__(
        self, u0, x, t, layers,
        epochs=5000, nu=0.1, equation='burgers',
        N = (1000, 1000, 5000), lr = 1e-3, real_solution=None,
        bc_type='dirichlet', path=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.nu = nu
        self.equation = equation.lower()
        self.real_solution = real_solution
        self.bc_type = bc_type.lower()
        self.model = MLP(layers).to(self.device)
        self.path = path
        self.losses = []
        self.loaded = None
        if path is not None:
            try:
                self.model.load_state_dict(torch.load(path))
                self.losses = np.load(path.replace('.pth', '_losses.npy'))
                self.loaded = True
            except FileNotFoundError:
                self.loaded = False

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.x = np.array(x, dtype=float)
        self.t = np.array(t, dtype=float)
        self.x_min, self.x_max = float(self.x.min()), float(self.x.max())
        self.t_min, self.t_max = float(self.t.min()), float(self.t.max())

        self.u0_np = (u0.copy() if isinstance(u0, np.ndarray)
                      else (u0.detach().cpu().numpy() if torch.is_tensor(u0)
                            else np.array(u0))).reshape(-1)

        if self.bc_type == 'dirichlet':
            self.bc_left_val, self.bc_right_val = self.u0_np[0], self.u0_np[-1]
        elif self.bc_type == 'neumann':
            self.bc_left_val, self.bc_right_val = self.u0_np[0], self.u0_np[-1]
        elif self.bc_type == 'periodic':
            pass
        else:
            raise ValueError(f"Unknown bc_type '{bc_type}'")

        X, T = np.meshgrid(self.x, self.t, indexing='ij')
        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.grid = torch.tensor(XT, dtype=torch.float32, device=self.device)
        self.N_ic, self.N_bc, self.N_col = N

    def sample_collocation(self, N):
        x = torch.rand(N,1, device=self.device)*(self.x_max-self.x_min) + self.x_min
        t = torch.rand(N,1, device=self.device)*(self.t_max-self.t_min) + self.t_min
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t

    def sample_initial(self, N):
        x_ic = torch.rand(N,1, device=self.device)*(self.x_max-self.x_min) + self.x_min
        t_ic = torch.zeros_like(x_ic, device=self.device)

        u0_ic = np.interp(x_ic.cpu().numpy().flatten(), self.x, self.u0_np)
        u0_ic = torch.tensor(u0_ic, dtype=torch.float32, device=self.device).view(-1,1)
        return x_ic, t_ic, u0_ic

    def sample_boundary(self, N):
        t_bc = torch.rand(N,1, device=self.device)*(self.t_max-self.t_min) + self.t_min
        if self.bc_type == 'dirichlet':
            x_left = torch.full_like(t_bc, self.x_min)
            x_right = torch.full_like(t_bc, self.x_max)
            Xb = torch.cat([torch.cat([x_left, t_bc], dim=1),
                            torch.cat([x_right, t_bc], dim=1)], dim=0)
            u_b = torch.cat([
                torch.full_like(t_bc, self.bc_left_val),
                torch.full_like(t_bc, self.bc_right_val)
            ], dim=0).view(-1,1).to(self.device)
            return Xb, u_b

        elif self.bc_type == 'neumann':
            x_left = torch.full_like(t_bc, self.x_min)
            x_right = torch.full_like(t_bc, self.x_max)
            Xb = torch.cat([torch.cat([x_left, t_bc], dim=1),
                            torch.cat([x_right, t_bc], dim=1)], dim=0)
            Xb = Xb.clone().requires_grad_(True)
            dub = torch.cat([
                torch.full_like(t_bc, self.bc_left_val),
                torch.full_like(t_bc, self.bc_right_val)
            ], dim=0).view(-1,1).to(self.device)
            return Xb, dub

        else:  # periodic
            x_left = torch.full_like(t_bc, self.x_min)
            x_right = torch.full_like(t_bc, self.x_max)
            X_lb = torch.cat([x_left, t_bc], dim=1)
            X_rb = torch.cat([x_right, t_bc], dim=1)
            return X_lb, X_rb

    def pde_residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = self.model(torch.cat([x, t], dim=1))
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        if self.equation == 'heat':
            return u_t - self.nu * u_xx
        else:
            return u_t + u * u_x - self.nu * u_xx

    def compute_loss(self):
        x_ic, t_ic, u0_ic = self.sample_initial(self.N_ic)
        u_pred_ic = self.model(torch.cat([x_ic, t_ic], dim=1))
        loss_i = self.loss_fn(u_pred_ic, u0_ic)

        if self.bc_type in ['dirichlet', 'neumann']:
            Xb, u_b = self.sample_boundary(self.N_bc)
            if self.bc_type == 'dirichlet':
                u_pred_b = self.model(Xb)
                loss_b = self.loss_fn(u_pred_b, u_b)
            else:
                u_pred = self.model(Xb)
                x_b = Xb[:, :1]
                u_x = autograd.grad(u_pred, x_b, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
                loss_b = self.loss_fn(u_x, u_b)
        else:  # periodic
            X_lb, X_rb = self.sample_boundary(self.N_bc)
            u_lb = self.model(X_lb)
            u_rb = self.model(X_rb)
            loss_b = self.loss_fn(u_lb, u_rb)

        x_f, t_f = self.sample_collocation(self.N_col)
        f_pred = self.pde_residual(x_f, t_f)
        loss_f = self.loss_fn(f_pred, torch.zeros_like(f_pred))

        total_loss = loss_i + loss_b + loss_f
        return total_loss, (loss_i.item(), loss_b.item(), loss_f.item())

    def Train(self, save_checkpoints=False):
        self.model.train()
        for ep in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            loss, (li, lb, lf) = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            self.losses.append((li, lb, lf, loss.item()))
            if ep % (self.epochs // 10 or 1) == 0:
                print(f"PINN: [{ep:5d}/{self.epochs}] | Loss={loss.item():.3e} | PDE={lf:.3e} | BC={lb:.3e} | IC={li:.3e}")
        if self.loaded == False:
            torch.save(self.model.state_dict(), self.path)
            losses = np.array(self.losses)
            np.save(self.path.replace('.pth', '_losses.npy'), losses)

    def Pred(self):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(self.grid).cpu().numpy()
        Nx, Nt = len(self.x), len(self.t)
        self.U_pred = u_pred.reshape(Nx, Nt).T

    def plot(self, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(self.U_pred, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                       origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        label = 'α' if self.equation == 'heat' else 'ν'
        ax.set_title(rf"PINN $\hat{{u}}(x,t)$ with {label} = {self.nu:.3g}")
        ax.set_xlabel('x'); ax.set_ylabel('t')
        fig.colorbar(im, ax=ax, label=r'$\hat{u}$')
        plt.tight_layout()
        plt.show()


class wPINN:
    def __init__(self, u0, x, t, layers, epochs = 5000, N = (1000, 1000, 5000), lr=1e-3,
                 bc_type='dirichlet', lambda_bc=1.0, adv_steps=5, path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x = np.array(x, dtype=float)
        self.t = np.array(t, dtype=float)
        self.u0_np = np.array(u0, dtype=float).reshape(-1)
        self.u0 = torch.tensor(self.u0_np, dtype=torch.float32, device=self.device).view(-1,1)

        self.u0_x_np = np.gradient(self.u0_np, self.x)

        self.u0 = torch.tensor(self.u0_np,  dtype=torch.float32, device=self.device).view(-1,1)
        self.u0_x = torch.tensor(self.u0_x_np, dtype=torch.float32, device=self.device).view(-1,1)

        self.t_min, self.t_max = float(self.t[0]), float(self.t[-1])
        self.x_min, self.x_max = float(self.x[0]), float(self.x[-1])

        self.lambda_bc = lambda_bc
        self.adv_steps = adv_steps

        self.layers = layers
        self.u_net   = MLP(layers).to(self.device)
        if not path.endswith('.pth'):
            path += '.pth'
        self.path = path
        self.losses = []
        self.loaded = None
        if self.path is not None:
            try:
                self.u_net.load_state_dict(torch.load(self.path))
                self.losses = np.load(self.path.replace('.pth', '_losses.npy'))
                self.loaded = True
            except FileNotFoundError:
                self.loaded = False

        self.phi_net = MLP([2,50,50,50,1]).to(self.device)
        self.xi_net  = MLP([2,50,50,50,1]).to(self.device)

        self.opt_u   = torch.optim.Adam(self.u_net.parameters(),   lr=lr)
        self.opt_phi = torch.optim.Adam(self.phi_net.parameters(), lr=lr)
        self.opt_xi  = torch.optim.Adam(self.xi_net.parameters(),  lr=lr)

        T_mesh, X_mesh = np.meshgrid(self.t, self.x, indexing='ij')  # shapes (nt,nx)
        stacked = np.hstack((T_mesh.flatten()[:,None], X_mesh.flatten()[:,None]))  # (nt*nx,2)
        self.grid = torch.tensor(stacked, dtype=torch.float32, device=self.device)
        self.U_pred = None
        self.N_ic, self.N_bc, self.N_int = N
        self.epochs = epochs
        self.bc_type = bc_type.lower()
        # self.R = RusanovBurgersSolver(self.u0_np.flatten(), self.x, self.t, nu=0, bc_type=self.bc_type)
        # self.U_exact = self.R.solve()
    def sample_collocation(self, N):
        t = torch.rand(N,1, device=self.device)*(self.t_max-self.t_min) + self.t_min
        x = torch.rand(N,1, device=self.device)*(self.x_max-self.x_min) + self.x_min
        t.requires_grad_(True); x.requires_grad_(True)
        return t, x

    def flux(self, u): return 0.5*u**2

    def cutoff(self, x):
        center = (self.x_min + self.x_max)/2
        radius = (self.x_max - self.x_min)/2
        return 1 - ((x - center)/radius)**2

    def pde_pairing(self, t, x):
        inp = torch.cat([t,x], dim=1)
        u = self.u_net(inp)
        u_t = autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        f = self.flux(u)
        f_x = autograd.grad(f, x, torch.ones_like(f), retain_graph=True, create_graph=True)[0]

        phi_raw = self.phi_net(inp)
        phi = phi_raw * self.cutoff(x)
        phi_x = autograd.grad(phi, x, torch.ones_like(phi), retain_graph=True, create_graph=True)[0]

        return u_t*phi - f*phi_x, phi_x

    def entropy_pairing(self, t, x):
        inp = torch.cat([t,x], dim=1)
        u = self.u_net(inp)
        e = 0.5*u**2
        q = (1./3.)*u**3
        e_t = autograd.grad(e, t, torch.ones_like(e), retain_graph=True, create_graph=True)[0]
        q_x = autograd.grad(q, x, torch.ones_like(q), retain_graph=True, create_graph=True)[0]

        res_pos = torch.relu(e_t + q_x)
        xi_raw = self.xi_net(inp)
        xi = xi_raw * self.cutoff(x)
        xi_x = autograd.grad(xi, x, torch.ones_like(xi), retain_graph=True, create_graph=True)[0]

        return res_pos*xi, xi_x

    def compute_boundary_loss(self):
        t_bc = torch.rand(self.N_bc, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min

        t_bc_cat = torch.cat([t_bc, t_bc], dim=0)
        x_bc = torch.cat([torch.full_like(t_bc, self.x_min),
                          torch.full_like(t_bc, self.x_max)], dim=0)
        X_bc = torch.cat([t_bc_cat, x_bc], dim=1)

        u_pred = self.u_net(X_bc)

        if self.bc_type == 'dirichlet':
            u_left  = self.u0[0]  
            u_right = self.u0[-1]     
            u_true  = torch.cat([
                u_left .expand(self.N_bc,1),
                u_right.expand(self.N_bc,1)
            ], dim=0)
            return (u_pred - u_true).pow(2).mean()

        elif self.bc_type == 'neumann':
            Xb = X_bc.clone().requires_grad_(True)
            u_b = self.u_net(Xb)
            u_x = autograd.grad(u_b,
                                Xb[:,1:2],
                                grad_outputs=torch.ones_like(u_b),
                                create_graph=True)[0]

            ux_left  = self.u0_x[0]
            ux_right = self.u0_x[-1]
            g_true   = torch.cat([
                ux_left .expand(self.N_bc,1),
                ux_right.expand(self.N_bc,1)
            ], dim=0)
            return (u_x - g_true).pow(2).mean()

        elif self.bc_type == 'periodic':
            X_lb = torch.cat([t_bc, torch.full_like(t_bc, self.x_min)], dim=1)
            X_rb = torch.cat([t_bc, torch.full_like(t_bc, self.x_max)], dim=1)
            return (self.u_net(X_lb) - self.u_net(X_rb)).pow(2).mean()

        else:
            raise ValueError(f"Unknown bc_type {self.bc_type}")
        

    def Train(self, save_checkpoints = False, store_times = False):
        train_start = time.time()
        self.times_dict = {}
        if save_checkpoints: 
            checkpoints = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000]
        for epoch in range(1, self.epochs+1):
            for _ in range(self.adv_steps):
                t_int, x_int = self.sample_collocation(self.N_int)
                r_pde, phi_x = self.pde_pairing(t_int, x_int)
                Lu_PDE = (r_pde - 0.5*phi_x**2).mean()
                r_ent, xi_x = self.entropy_pairing(t_int, x_int)
                L_ent = (r_ent - 0.5*xi_x**2).mean()

                self.opt_phi.zero_grad(); (-Lu_PDE).backward(retain_graph=True); self.opt_phi.step()
                self.opt_xi.zero_grad(); (-L_ent).backward(retain_graph=True); self.opt_xi.step()

            t_int, x_int = self.sample_collocation(self.N_int)
            r_pde, phi_x = self.pde_pairing(t_int, x_int)
            Lu_PDE = (r_pde - 0.5*phi_x**2).mean()
            r_ent, xi_x = self.entropy_pairing(t_int, x_int)
            L_ent = (r_ent - 0.5*xi_x**2).mean()

            x_ic = torch.rand(self.N_ic,1, device=self.device)*(self.x_max-self.x_min)+self.x_min
            t_ic = torch.zeros_like(x_ic, device=self.device)
            u_ic = self.u_net(torch.cat([t_ic, x_ic], dim=1))
            u0_ic = np.interp(x_ic.cpu().numpy().flatten(), self.x, self.u0_np)
            u0_ic = torch.tensor(u0_ic, dtype=torch.float32, device=self.device).view(-1,1)
            L_ic = (u_ic - u0_ic).pow(2).mean()

            L_bc = self.compute_boundary_loss()
            loss = Lu_PDE + L_ent + self.lambda_bc*(L_ic + L_bc)
            self.opt_u.zero_grad(); loss.backward(); self.opt_u.step()

            self.losses.append((Lu_PDE.item(), L_ent.item(), L_ic.item(), L_bc.item(), loss.item()))

            if epoch % 200 == 0:
                print(f"wPINN: [{epoch}/{self.epochs}] | Loss={loss.item():.2e} | PDE={Lu_PDE.item():.2e} | Ent={L_ent.item():.2e} | BC={L_bc.item():.2e} | IC={L_ic.item():.2e}")
            
            '''
            if self.loaded == False:
                if save_checkpoints and epoch in checkpoints:
                    torch.save(self.u_net.state_dict(), self.path+f'_{epoch}.pth')
                    losses = np.array(self.losses)
                    np.save(self.path+f'_{epoch}'+'_losses.npy', losses)
                elif not save_checkpoints and (epoch == self.epochs):
                    torch.save(self.u_net.state_dict(), self.path)
                    losses = np.array(self.losses)
                    np.save(self.path.replace('.pth', '_losses.npy'), losses)
            '''
            if store_times and epoch % 200 == 0:
                elapsed = time.time() - train_start
                #self.Pred()
                #self.U_err = self.U_pred - self.U_exact
                # l2 = self.l2_error()
                self.times_dict[str((self.layers, epoch))] = elapsed/60
                # print(f"[Epoch {epoch}] L2 error = {l2:.3e}, elapsed = {elapsed:.1f}s")
        times_path = self.path + '_times.json'
        os.makedirs(os.path.dirname(times_path), exist_ok=True)
        with open(times_path, 'w') as f:
            json.dump(self.times_dict, f, indent=4)
        print(f"Saved timing info to {times_path}")


    def Pred(self):
        self.u_net.eval()
        with torch.no_grad():
            u_pred = self.u_net(self.grid).cpu().numpy()
        nt, nx = len(self.t), len(self.x)
        self.U_pred = u_pred.reshape(nt, nx)

    def plot(self, vmin=None, vmax=None):
        # Imposto i font a 20pt
        plt.rcParams.update({
            'font.size':       20,
            'axes.titlesize':  20,
            'axes.labelsize':  20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
        })

        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(
            self.U_pred,
            extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
            origin='lower', aspect='auto',
            cmap='jet', vmin=vmin, vmax=vmax
        )
        ax.set_title(r"wPINN $\hat{u}(x,t)$")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        cbar = fig.colorbar(im, ax=ax, label=r'$\hat{u}$')
        # Anche l'etichetta della colorbar a 20pt
        cbar.ax.yaxis.get_offset_text().set_fontsize(20)
        cbar.ax.yaxis.label.set_size(20)

        plt.tight_layout()
        plt.show()

    def l2_error(self):

        Nt, Nx = self.U_err.shape
        dx = (self.x_max - self.x_min) / (Nx - 1)
        dt = (self.t[-1] - self.t[0]) / (Nt - 1)

        l2_abs = np.sqrt(np.sum(self.U_err   ** 2) * dx * dt)

        return l2_abs


class ProblemSetUp:
    def __init__(self, u0, x, t, layers, lr=1e-3, 
                 epochs=5000, N=(1000,1000,5000), nu=0.1,
                 equation='burgers', real_solution=None,
                 bc_type='dirichlet', lambda_bc=1.0, 
                 adv_steps=5, model = 'PINN', 
                 auto_train = True, store_times = False, path=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.lower()
        self.nu = nu
        self.real_solution = real_solution
        self.equation = equation.lower()
        self.bc_type = bc_type.lower()
        self.auto_train = auto_train
        self.losses = []
        self.u0_np = (u0.copy() if isinstance(u0, np.ndarray)
                        else (u0.detach().cpu().numpy() if torch.is_tensor(u0)
                                else np.array(u0))).reshape(-1, 1)
        self.u0 = torch.tensor(self.u0_np, dtype=torch.float32, device=self.device)
        self.x = np.array(x, dtype=float)
        self.t = np.array(t, dtype=float)
        self.x_min, self.x_max = float(self.x.min()), float(self.x.max())
        self.t_min, self.t_max = float(self.t.min()), float(self.t.max())
        self.L = self.x_max - self.x_min
        self.epochs = epochs

        if self.model == 'pinn':
            self.pinn = PINN(u0, x, t, layers, epochs=epochs, nu=nu, equation=equation, 
                            N=N, lr=lr, real_solution=real_solution, bc_type=bc_type, path=path)
        elif self.model == 'wpinn':
            self.pinn = wPINN(u0, x, t, layers, epochs=epochs, N=N, lr=lr, bc_type=bc_type, 
                            lambda_bc=lambda_bc, adv_steps=adv_steps, path=path)
        
        if auto_train and not self.pinn.loaded:
            self.pinn.Train(save_checkpoints=True)

        self.pinn.Pred()
        self.U_pred = self.pinn.U_pred
        self.losses = self.pinn.losses

        if self.real_solution is not None:
            self.U_exact = real_solution
        else:
            if self.equation == 'burgers':
                self.R = RusanovBurgersSolver(self.u0_np.flatten(), self.x, self.t, self.nu, self.bc_type)
                self.U_exact = self.R.solve()
            elif self.equation == 'heat':
                self.H = HeatSolver(self.u0_np.flatten(), self.x, self.t, self.nu)
                self.U_exact = self.H.solve()

        self.U_err = self.U_pred - self.U_exact

    def plot_losses(self, window: int = 200, eps: float = 1e-8, figsize=(12,6)):
        # Imposto il font grande prima di plottare
        plt.rcParams.update({
            'font.size':           20,
            'axes.titlesize':      20,
            'axes.labelsize':      20,
            'xtick.labelsize':     20,
            'ytick.labelsize':     20,
            'legend.fontsize':     20,
            'legend.title_fontsize':20
        })

        if type(self.losses) == list:
            if not self.losses:
                print("No losses to plot.")
                return
        else:
            if self.losses.size == 0:
                print("No losses to plot.")
                return

        arr = np.array(self.losses, dtype=float)
        N, M = arr.shape

        if M == 4:
            labels = ['Loss IC', 'Loss BC', 'Loss PDE', 'Loss Totale']
        elif M == 5:
            labels = ['Loss PDE', 'Loss Entropia', 'Loss IC', 'Loss BC', 'Loss Totale']
        else:
            raise ValueError(f"Unexpected loss‐tuple length: {M}")

        color_map = {
            'Loss PDE':      'blue',
            'Loss IC':       'green',
            'Loss BC':       'red',
            'Loss Totale':   'purple',
            'Loss Entropia': 'orange',
        }

        cols = [np.clip(arr[:, i], eps, None) for i in range(M)]

        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='valid')

        smooth = [moving_average(c, window) for c in cols]

        epochs = np.arange(len(smooth[0])) + window//2 + 1

        fig, ax = plt.subplots(figsize=figsize)
        for lbl, series in zip(labels, smooth):
            lw = 2 if "Totale" in lbl else 1
            alpha = 0.9 if "Totale" in lbl else 0.7
            ax.semilogy(epochs, series,
                        label=lbl,
                        color=color_map.get(lbl, 'black'),
                        linewidth=lw,
                        alpha=alpha)
        
        title0 = "wPINN" if self.model == 'wpinn' else "PINN"

        ax.set_title(title0)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss (scala logaritmica)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(title="Componenti")
        plt.tight_layout()
        plt.show()

        return fig, ax


    def plot(self):
        # font grande
        plt.rcParams.update({
            'font.size':       20,
            'axes.titlesize':  20,
            'axes.labelsize':  20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
        })

        vmin = min(self.U_pred.min(), self.U_exact.min())
        vmax = max(self.U_pred.max(), self.U_exact.max())
        
        print("Predizione PINN")
        self.pinn.plot(vmin=vmin, vmax=vmax)
        
        print("Rusanov")
        self.R.plot(cmap='jet', vmin=vmin, vmax=vmax)

    def plot_comparison(self, num_slices=3, vmax=None, vmin=None):

        # dimensione font
        plt.rcParams.update({
            'font.size':       20,
            'axes.titlesize':  20,
            'axes.labelsize':  20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
        })

        # limiti dati
        vmin = vmin if vmin is not None else min(self.U_pred.min(), self.U_exact.min())
        vmax = vmax if vmax is not None else max(self.U_pred.max(), self.U_exact.max())
        verr = abs(self.U_err[:-2].max())

        # istanti temporali per i profili
        time_indices = np.linspace(1, len(self.t) - 1, num_slices, dtype=int)

        # figura 2×3 con constrained_layout
        fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        # --- RIGA 1: le tre mappe -----------------------------------------------
        # 1) PINN
        im_pred = axs[0, 0].pcolormesh(self.x, self.t, self.U_pred,
                                    shading='auto', cmap='jet',
                                    vmin=vmin, vmax=vmax)
        
        title0 = "wPINN" if self.model == 'wpinn' else "PINN"
        axs[0, 0].set_title(title0)
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel("t")

        # 2) Riferimento
        im_ref = axs[0, 1].pcolormesh(self.x, self.t, self.U_exact,
                                    shading='auto', cmap='jet',
                                    vmin=vmin, vmax=vmax)
        
        axs[0, 1].set_title("Riferimento")
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("t")

        # 3) Errore
        if verr > 0.25:
            verr/=3
        else:
            verr*=3
        im_err = axs[0, 2].pcolormesh(self.x, self.t, self.U_err,
                                    shading='auto', cmap='bwr',
                                    vmin=-verr, vmax=verr)
        axs[0, 2].set_title("Errore")
        axs[0, 2].set_xlabel("x")
        axs[0, 2].set_ylabel("t")

        # unica colorbar a sinistra per PINN+Riferimento
        cbar_left = fig.colorbar(
            im_pred,
            ax=[axs[0,0], axs[0,1]],
            location='left',
            pad=0.02,
        )
        cbar_left.ax.yaxis.set_tick_params(labelsize=20)
        cbar_left.ax.yaxis.label.set_size(20)

        # unica colorbar a destra per Errore
        cbar_right = fig.colorbar(
            im_err,
            ax=axs[0,2],
            location='right',
            pad=0.02,
        )
        cbar_right.ax.yaxis.set_tick_params(labelsize=20)
        cbar_right.ax.yaxis.label.set_size(20)

        # --- RIGA 2: profili ----------------------------------------------------
        # raccogliamo handle & label per una sola legenda
        legend_handles, legend_labels = None, None

        for j, idx in enumerate(time_indices):
            ax = axs[1, j]
            t_val = self.t[idx]
            u_ref  = self.U_exact[idx, :]
            u_pred = self.U_pred[idx, :]

            lbl_ref  = ("Esatto" if self.real_solution is not None
                        else ("Rusanov" if self.equation=="burgers" else "FEM"))
            lbl_pred = "wPINN" if self.model=="wpinn" else "PINN"

            h1, = ax.plot(self.x, u_ref,  'k-',  linewidth=2, label=lbl_ref)
            h2, = ax.plot(self.x, u_pred, 'r--', linewidth=2, label=lbl_pred)

            ax.set_title(f"t = {t_val:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x,t)")
            ax.set_ylim(vmin, vmax)
            ax.grid(True)

            if legend_handles is None:
                legend_handles = [h1, h2]
                legend_labels  = [lbl_ref, lbl_pred]

        # legenda unica in basso, con spazio extra
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', ncol=2,
            bbox_to_anchor=(0.5, -0.08)
        )

        plt.show()



    def l2_error(self):

        Nt, Nx = self.U_err.shape
        dx = (self.x_max - self.x_min) / (Nx - 1)
        dt = (self.t[-1] - self.t[0]) / (Nt - 1)

        l2_abs = np.sqrt(np.sum(self.U_err   ** 2) * dx * dt)
        l2_rel = np.sqrt(np.sum(self.U_err** 2))

        header = "Metriche di Errore L2"
        print(f"\n{header}\n{'-' * len(header)}")
        print(f"{'Errore L2 assoluto':<30}: {l2_abs:.6e}")
        print(f"{'Errore L2 relativo':<30}: {l2_rel:.6e}\n")

        return l2_abs