import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 1000)

print("Using default x and t values:")
print("x:", len(x), "t:", len(t))

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

def default_u_true_func(x):
    return -np.sin(np.pi * x)


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

            # Boundary conditions
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
    def __init__(self, u0, x=x, t=t, layers = [2, 50,50,50,1], epochs = 5000, N_col=100000, boundary_condition=None, nu=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu

        self.x = np.array(x)
        self.t = np.array(t)

        self.X, self.T = np.meshgrid(self.x, self.t, indexing='ij')
        self.X_flat = self.X.flatten()[:, None]
        self.T_flat = self.T.flatten()[:, None]

        self.grid = torch.tensor(np.hstack((self.X_flat, self.T_flat)), dtype=torch.float32).to(self.device)

        self.x0 = np.hstack((self.x[:, None], self.t[0] * np.ones_like(self.x)[:, None]))
        self.u0_np = u0.copy() if isinstance(u0, np.ndarray) else u0.detach().cpu().numpy().flatten()
        self.u0 = torch.tensor(u0).reshape(-1, 1)

        self.X_lb = np.hstack((self.x[0] * np.ones_like(self.t)[:, None], self.t[:, None]))
        self.X_rb = np.hstack((self.x[-1] * np.ones_like(self.t)[:, None], self.t[:, None]))
        if not boundary_condition or boundary_condition == 'Dirichlet':
            self.u_boundary = torch.zeros_like(torch.tensor(self.t)).reshape(-1, 1)

        self.X_given = torch.tensor(np.vstack((self.x0, self.X_lb, self.X_rb)), dtype=torch.float32).to(self.device)
        self.u_given = torch.tensor(np.vstack((self.u0, self.u_boundary, self.u_boundary)), dtype=torch.float32).to(self.device)

        self.N_col = N_col
        self.x_col = np.random.uniform(self.x[0], self.x[-1], (self.N_col, 1))
        self.t_col = np.random.uniform(self.t[0], self.t[-1], (self.N_col, 1))
        self.X_col = torch.tensor(np.hstack((self.x_col, self.t_col)), dtype=torch.float32).to(self.device)

        self.model = PINN(layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        self.u_pred = None

    def plot_initial_data(self):
        plt.plot(self.x, self.u0_np)
        plt.title('Initial Condition')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.grid(True)
        plt.show()

    def plot_training_points_1D(self):
        plt.figure(figsize=(8, 6))

        vmin = np.min(self.u0_np)
        vmax = np.max(self.u0_np)

        sc0 = plt.scatter(self.x0[:, 0], self.x0[:, 1], c=self.u0_np, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)
        plt.scatter(self.X_lb[:, 0], self.X_lb[:, 1], c=self.u_boundary, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)
        plt.scatter(self.X_rb[:, 0], self.X_rb[:, 1], c=self.u_boundary, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar(sc0, label="u(x,t)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def pde_residual(self, x, t):
        t.requires_grad = True
        x.requires_grad = True
        u = self.model(torch.cat([x, t], dim=1))
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx

    def compute_loss(self):
        u_pred = self.model(self.X_given)
        mse_u = self.loss_fn(u_pred, self.u_given)

        x_f, t_f = self.X_col[:, 0:1], self.X_col[:, 1:2]
        f_pred = self.pde_residual(x_f, t_f)
        mse_f = self.loss_fn(f_pred, torch.zeros_like(f_pred))

        return mse_u + mse_f

    def Train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        self.model.eval()
        with torch.no_grad():
            self.u_pred = self.model(self.grid).cpu().numpy()

        
    def plot_predicted_slices_grid(self, num_slices=6):

        u_pred_2d = self.u_pred.reshape(len(self.x), len(self.t))

        time_indices = np.linspace(0, len(self.t) - 1, num_slices, dtype=int)

        plt.figure(figsize=(14, 8))
        for i, idx in enumerate(time_indices):
            t_val = self.t[idx]
            u_pred_slice = u_pred_2d[:, idx]

            plt.subplot(2, (num_slices + 1) // 2, i + 1)
            plt.plot(self.x, u_pred_slice, 'r--', label='Predicted')
            plt.title(f't = {t_val:.2f}')
            plt.xlabel('x')
            plt.ylabel('$\hat{u}(x,t)$')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()



class ProblemSetUp:
    def __init__(self, u0, x=x, t=t, nu=0.01, layers=[2, 50, 50, 50, 1], boundaries=[], boundary_fn=None, auto_plot=False, auto_train=True):
        self.u0 = u0
        self.x = x
        self.t = t
        self.nu = nu
        self.boundaries = boundaries
        self.boundary_fn = boundary_fn

        self.PS = PINNSolver(
            u0=u0,
            x=x,
            t=t,
            layers=layers,
            epochs=5000,
            N_col=50000,
            boundary_condition=None,
            nu=nu
        )

        self.R = RusanovBurgersSolver(
            u0=u0,
            x=x,
            t=t,
            nu=nu,
        )
        self.u_R = self.R.solve()  
        # print(self.u_R.shape)
        if auto_plot:
            self.PS.plot_initial_data()
            self.PS.plot_training_points_1D()

        if auto_train:
            self.PS.Train()

    def plot_solution(self):
        u_pred_2d = self.PS.u_pred.reshape(len(self.x), len(self.t)).T  # Transpose to match (t, x)
        residual = u_pred_2d - self.u_R

        # Use common color scale for u_R and u_pred
        common_vmin = min(self.u_R.min(), u_pred_2d.min())
        common_vmax = max(self.u_R.max(), u_pred_2d.max())
        
        # Residual color scale (symmetric around 0)
        res_max = max(abs(residual.min()), abs(residual.max()))

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Rusanov
        im0 = axs[0].imshow(self.u_R, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                            origin='lower', aspect='auto', cmap='viridis',
                            vmin=common_vmin, vmax=common_vmax)
        axs[0].set_title(r"Rusanov $u(x,t)$ con $\nu$ = {:.3g}".format(self.nu))
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        fig.colorbar(im0, ax=axs[0], label=r"$u_{{R}}$")

        # PINN Prediction
        im1 = axs[1].imshow(u_pred_2d, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                            origin='lower', aspect='auto', cmap='viridis',
                            vmin=common_vmin, vmax=common_vmax)
        axs[1].set_title(r"PINN $\hat{{u}}(x,t)$ con $\nu$ = {:.3g}".format(self.nu))
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        fig.colorbar(im1, ax=axs[1], label=r"$\hat{{u}}$")

        # Residual
        im2 = axs[2].imshow(residual, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                            origin='lower', aspect='auto', cmap='seismic',
                            vmin=common_vmin, vmax=common_vmax)
        axs[2].set_title(r"Residuo: $\hat{{u}} - u_{{R}}$")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("t")
        fig.colorbar(im2, ax=axs[2], label="Residuo")

        plt.tight_layout()
        plt.show()

