import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 1000)
print("Using default x and t values:")
print("x:", len(x), "t:", len(t))
torch.manual_seed(0)
np.random.seed(0)

layers = [2, 50, 50, 50, 1]

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f"tanh_{i}", nn.Tanh())
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        return self.net(x)

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
    def __init__(self, u0, x=x, t=t, nu=0.01):
        self.x = x
        self.t = t
        self.nx = len(x)
        self.nt = len(t)
        self.L = abs(x[-1]-x[0])
        self.T = abs(t[-1]-t[0])
        self.nu = nu

        self.dx = self.L / (self.nx - 1)
        self.dt = self.T / self.nt

        self.u = np.zeros((self.nt, self.nx))
        self.u[0, :] = u0

    def flux(self, u):
        return 0.5 * u**2

    def rusanov_flux(self, uL, uR):
        a = np.maximum(np.abs(uL), np.abs(uR))
        return 0.5 * (self.flux(uL) + self.flux(uR)) - 0.5 * a * (uR - uL)

    def solve(self):
        for n in range(self.nt - 1):
            un = self.u[n, :].copy()
            F = np.zeros_like(un)
            for i in range(1, self.nx - 1):
                F[i] = self.rusanov_flux(un[i], un[i+1]) - self.rusanov_flux(un[i-1], un[i])

            self.u[n+1, 1:-1] = un[1:-1] - self.dt / self.dx * F[1:-1] + \
                                self.nu * self.dt / self.dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])

            self.u[n+1, 0] = 0
            self.u[n+1, -1] = 0
        
        return self.u

    def plot(self, cmap='viridis', vmin=None, vmax=None):

        plt.figure(figsize=(12, 6))
        im = plt.imshow(
            self.u,
            extent=[self.x[0], self.x[-1], self.t[0], self.t[-1]],
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower'
        )
        plt.colorbar(im, label='u(x, t)')
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Burgers' Equation Solution Evolution")
        plt.tight_layout()
        plt.show()

class PINNSolver:
    def __init__(self, u0, x, t, layers, epochs = 5000, N_col = 10000, nu = 0.1, equation = 'burgers', real_solution = None, save_model = None, load_model = None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu
        self.epochs = epochs
        self.save_model = save_model
        self.load_model = load_model
        self.equation = equation.lower()
        self.real_solution = real_solution

        # Domain
        self.x = np.array(x)
        self.t = np.array(t)
        self.x_min, self.x_max = float(self.x.min()), float(self.x.max())
        self.L = self.x_max - self.x_min

        # Grid for collocation and plotting
        X, T = np.meshgrid(self.x, self.t, indexing="ij")
        self.X_flat = X.flatten()[:, None]
        self.T_flat = T.flatten()[:, None]
        self.grid = torch.tensor(
            np.hstack((self.X_flat, self.T_flat)), dtype=torch.float32, device=self.device
        )

        # Initial and boundary conditions
        self.u0_np = (
            u0.copy()
            if isinstance(u0, np.ndarray)
            else (u0.detach().cpu().numpy() if torch.is_tensor(u0) else np.array(u0))
        ).reshape(-1, 1)

        self.u0 = torch.tensor(self.u0_np, dtype=torch.float32, device=self.device)
        # IC points
        self.x0 = self.x[:, None]
        self.t0 = np.zeros_like(self.x0)
        # BC points
        self.tb = self.t[:, None]
        self.x_lb = self.x_min * np.ones_like(self.tb)
        self.x_rb = self.x_max * np.ones_like(self.tb)
        self.ub = np.zeros_like(self.tb)

        self.X_ic = np.hstack((self.x0, self.t0))
        self.X_icT = torch.tensor(self.X_ic, dtype=torch.float32, device=self.device)
        self.X_lb = np.hstack((self.x_lb, self.tb))
        self.X_rb = np.hstack((self.x_rb, self.tb))
        self.X_given = torch.tensor(
            np.vstack((self.X_lb, self.X_rb)), dtype=torch.float32, device=self.device
        )
        self.u_given = torch.tensor(
            np.vstack((self.ub, self.ub)), dtype=torch.float32, device=self.device
        )

        # Collocation points
        self.N_col = N_col
        x_col = np.random.uniform(self.x_min, self.x_max, (N_col, 1))
        t_col = np.random.uniform(self.t[0], self.t[-1], (N_col, 1))
        self.X_col = torch.tensor(
            np.hstack((x_col, t_col)), dtype=torch.float32, device=self.device
        )

        # PINN model
        self.model = PINN(layers).to(self.device)
        if load_model is not None:
            self.model.load_state_dict(torch.load(load_model))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        if self.real_solution is not None:
            self.U_exact = real_solution

        else:
            if self.equation == "burgers":
                self.R = RusanovBurgersSolver(self.u0_np.flatten(), self.x, self.t, self.nu)
                self.U_exact = self.R.solve()
            elif self.equation == "heat":
                self.H = HeatSolver(self.u0_np.flatten(),self.x, self.t, self.nu)
                self.U_exact = self.H.solve()

        self.U_pred = None
        self.U_err = None
        self.losses = []
        self.ada = 1.0

    def plot_initial_data(self):
        plt.plot(self.x, self.u0_np)
        plt.title('Initial Condition')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.grid(True)
        plt.show()

    def plot_training_points(self):
        plt.figure(figsize=(8, 6))

        vmin = np.min(self.u0_np)
        vmax = np.max(self.u0_np)

        sc0 = plt.scatter(self.X_ic[:, 0], self.X_ic[:, 1], c=self.u0_np, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)
        plt.scatter(self.X_lb[:, 0], self.X_lb[:, 1], c=self.ub, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)
        plt.scatter(self.X_rb[:, 0], self.X_rb[:, 1], c=self.ub, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar(sc0, label="u(x,t)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def pde_residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = self.model(torch.cat([x, t], dim=1))
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        if self.equation == "heat":
            return u_t - self.nu * u_xx
        else:
            return u_t + u * u_x - self.nu * u_xx

    def compute_initial_loss(self):
        u_pred_init = self.model(self.X_icT)
        loss_init = self.loss_fn(u_pred_init, self.u0)
        return loss_init

    def compute_boundary_loss(self):
        u_pred_bound = self.model(self.X_given)
        loss_bound = self.loss_fn(u_pred_bound, self.u_given)
        return loss_bound

    def compute_loss(self):
        loss_init = self.compute_initial_loss()
        loss_bound = self.compute_boundary_loss()

        x_f = self.X_col[:, 0:1]
        t_f = self.X_col[:, 1:2]
        f_pred = self.pde_residual(x_f, t_f)
        loss_phys = self.loss_fn(f_pred, torch.zeros_like(f_pred))

        self.losses.append((
            loss_init.item(), loss_bound.item(), loss_phys.item(),
            loss_init.item() + loss_bound.item() + loss_phys.item()
        ))

        return 5 * loss_init + loss_bound + loss_phys


    def Train(self):
        self.model.train()
        for ep in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            if ep % (self.epochs // 10) == 0:
                print(f"Epoch {ep:5d} | Loss = {loss.item():.3e}")
        if self.save_model is not None:
            torch.save(self.model.state_dict(), self.save_model)

    def Pred(self):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(self.grid).cpu().numpy()
        Nx, Nt = len(self.x), len(self.t)
        self.U_pred = u_pred.reshape(Nx, Nt).T
        self.U_err = self.U_pred - self.U_exact

    def plot(self):
        common_vmin = np.min(self.U_pred)
        common_vmax = np.max(self.U_pred)

        fig, axs = plt.subplots(1, 1, figsize=(8, 5))

        im = axs.imshow(self.U_pred, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                        origin='lower', aspect='auto', cmap='jet',
                        vmin=common_vmin, vmax=common_vmax)
        
        if self.equation == "burgers":
            axs.set_title(r"PINN $\hat{{u}}(x,t)$ con $\nu$ = {:.3g}".format(self.nu))
        else:
            axs.set_title(r"PINN $\hat{{u}}(x,t)$ con $\alpha$ = {:.3g}".format(self.nu))

        axs.set_xlabel("x")
        axs.set_ylabel("t")
        fig.colorbar(im, ax=axs, label=r"$\hat{{u}}$")

        plt.tight_layout()
        plt.show()


    def plot_losses(self):

        loss_data_vals = [loss[0] for loss in self.losses]
        loss_phys_vals = [loss[1] for loss in self.losses]
        total_loss_vals = [loss[2] for loss in self.losses]

        epochs = range(1, len(self.losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_data_vals, label='Data Loss')
        plt.plot(epochs, loss_phys_vals, label='Physics Loss')
        plt.plot(epochs, total_loss_vals, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_comparison(self):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        im0 = axs[0].pcolormesh(self.x, self.t, self.U_pred,
                                shading='auto', cmap='jet')
        axs[0].set_title(r"PINN $\hat{{u}}$")
        axs[0].set_xlabel("x"); axs[0].set_ylabel("t")
        fig.colorbar(im0, ax=axs[0], label=r"$u_{pred}$")

        im1 = axs[1].pcolormesh(self.x, self.t, self.U_exact,
                                shading='auto', cmap='jet')
        if self.real_solution is not None:
            title = r"Analytical $u$" 
        else:  
            title = r"Reference FEM $u$"
        axs[1].set_title(title)
        axs[1].set_xlabel("x"); axs[1].set_ylabel("t")
        fig.colorbar(im1, ax=axs[1], label=r"$u_{exact}$")

        im2 = axs[2].pcolormesh(self.x, self.t, self.U_err,
                                shading='auto', cmap='bwr')
        axs[2].set_title(r"Error $\hat{{u}} - u$")
        axs[2].set_xlabel("x"); axs[2].set_ylabel("t")
        fig.colorbar(im2, ax=axs[2], label="error")

        plt.show()

    def plot_slices(self, num_slices=6, vmax = None, vmin = None):
        time_indices = np.linspace(0, len(self.t) - 1, num_slices, dtype=int)

        if not vmin: vmin = min(self.U_pred.min(), self.U_exact.min())
        if not vmax: vmax = max(self.U_pred.max(), self.U_exact.max())

        plt.figure(figsize=(16, 9))
        for i, idx in enumerate(time_indices):
            t_val = self.t[idx]
            u_pred_slice = self.U_pred[int(idx), :]
            u_rusanov_slice = self.U_exact[int(idx), :]


            plt.subplot(2, (num_slices + 1) // 2, i + 1)
            if self.equation == "burgers":
                label = 'Rusanov'
            elif self.equation == "heat":
                if self.real_solution is not None:
                    label = 'Exact'
                else:
                    label = 'FEM'
            plt.plot(self.x, u_rusanov_slice, 'k-', linewidth=1.5, label=label)
            plt.plot(self.x, u_pred_slice, 'r--', linewidth=1.5, label='PINN')

            plt.title(f't = {t_val:.3f}')
            plt.xlabel('x')
            plt.ylabel('u(x,t)')
            plt.ylim(vmin, vmax)
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def compute_l2_error(self):
        Nt, Nx = self.U_err.shape
        dx = (self.x_max - self.x_min) / (Nx - 1)
        dt = (self.t[-1] - self.t[0]) / (Nt - 1)
        l2_abs = np.sqrt(np.sum(self.U_err**2) * dx * dt)
        l2_exact = np.sqrt(np.sum(self.U_exact**2) * dx * dt)
        return l2_abs, l2_abs / l2_exact