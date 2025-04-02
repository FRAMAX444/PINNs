import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt


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

class PINNSolver:
    def __init__(self, layers, x, t, epochs = 5000, N_col=10000, boundary_condition=None, u_true_func=None, nu=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu

        self.u_true_func = u_true_func if u_true_func is not None else default_u_true_func

        self.x = np.array(x)
        self.t = np.array(t)

        self.X, self.T = np.meshgrid(self.x, self.t, indexing='ij')
        self.X_flat = self.X.flatten()[:, None]
        self.T_flat = self.T.flatten()[:, None]

        self.grid = torch.tensor(np.hstack((self.X_flat, self.T_flat)), dtype=torch.float32).to(self.device)

        self.x0 = np.hstack((self.x[:, None], self.t[0] * np.ones_like(self.x)[:, None]))
        self.u0 = self.u_true_func(self.x).reshape(-1, 1)

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
        plt.plot(self.x, self.u0)
        plt.title('Initial Condition')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.grid(True)
        plt.show()

    def plot_training_points_1D(self):
        plt.figure(figsize=(8, 6))

        vmin = np.min(self.u0)
        vmax = np.max(self.u0)

        # Initial condition (horizontal line): t = 0
        sc0 = plt.scatter(self.x0[:, 0], self.x0[:, 1], c=self.u0, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)

        # Left boundary (vertical line): x = x_min
        plt.scatter(self.X_lb[:, 0], self.X_lb[:, 1], c=self.u_boundary, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)

        # Right boundary (vertical line): x = x_max
        plt.scatter(self.X_rb[:, 0], self.X_rb[:, 1], c=self.u_boundary, cmap='coolwarm', vmin=vmin, vmax=vmax, s=25)

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar(sc0, label="u(x,t)")
        plt.legend()
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


    def plot_solution(self):

        # Reshape prediction to 2D and transpose to swap axes
        u_pred_2d = self.u_pred.reshape(len(self.x), len(self.t)).T  # Transpose to switch axes

        # Plotting
        plt.figure(figsize=(6, 5))
        plt.imshow(u_pred_2d, extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()],
                origin='lower', aspect='auto', cmap='viridis')
        plt.title("$\\hat{u}(x,t)$")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar(label="$\\hat{u}$")
        plt.tight_layout()
        plt.show()

    def plot_predicted_slices_grid(self, num_slices=6):
        # Convert u_pred to 2D: shape (len(x), len(t))
        u_pred_2d = self.u_pred.reshape(len(self.x), len(self.t))

        # Choose evenly spaced time indices
        time_indices = np.linspace(0, len(self.t) - 1, num_slices, dtype=int)

        plt.figure(figsize=(14, 8))
        for i, idx in enumerate(time_indices):
            t_val = self.t[idx]
            u_pred_slice = u_pred_2d[:, idx]

            plt.subplot(2, (num_slices + 1) // 2, i + 1)
            plt.plot(self.x, u_pred_slice, 'r--', label='Predicted')
            plt.title(f't = {t_val:.2f}')
            plt.xlabel('x')
            plt.ylabel('$\\hat{u}(x,t)$')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()