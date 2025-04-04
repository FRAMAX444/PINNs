import scipy.io
from scipy.interpolate import griddata
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
from tqdm import tqdm


############################################# 1st MODEL DEFINITION #################################################

class SmallNetwork(nn.Module):
    def __init__(self):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SmallPINN(nn.Module):
    def __init__(self, 
                 X, # (x,t) points
                 u, # labels (soluzioni della PDE nei punti (x,t))
                 lb, # lower bound
                 ub, # upper bound
                 physics = True,
                 nu = 0.01,
                 original_shape = (100, 256)):
        super(SmallPINN, self).__init__()
        self.lb = lb.clone().detach().float()
        self.ub = ub.clone().detach().float()
        self.x = X[:, 0:1].clone().detach().requires_grad_(True).float()
        self.t = X[:, 1:2].clone().detach().requires_grad_(True).float()

        self.u = torch.tensor(u).float()
        self.nu = nu
        self.physics = physics
        self.lossTracker = []
        self.original_shape = original_shape
        self.network = SmallNetwork()
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
    def forward(self, x,t):
        X = torch.cat([x,t],1)
        return self.network(X)
    
    def grad(self, u, x, out):
        grad_u = torch.autograd.grad(u, x, out, create_graph=True)[0]
        return grad_u
    
    def residual(self, x, t):
        
        u = self.forward(x, t)
        u_t = self.grad(u, t, torch.ones_like(u))
        u_x = self.grad(u, x, torch.ones_like(u))
        u_xx = self.grad(u_x, x, torch.ones_like(u_x))
        
        return u_t + u*u_x - (self.nu/np.pi)*u_xx
    
    def train(self, epochs):
        self.network.train()
        for epoch in range(epochs):
            u_pred = self.forward(self.x, self.t)
            residual_pred = self.residual(self.x, self.t)
            loss = torch.mean((self.u - u_pred)**2)
            if self.physics == True:
                loss += torch.mean(residual_pred**2)
            self.lossTracker.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % (epochs // 10) == 0:
                print("Epoch: ", epoch, "| Loss: ", loss.item())

        plt.figure(figsize=(10, 6))
        plt.plot(self.lossTracker, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

        return self.lossTracker
            
    def predict(self, x, t): 
        self.network.eval()

        x = torch.tensor(x, requires_grad=True).float()
        t = torch.tensor(t, requires_grad=True).float()

        u = self.forward(x, t)  # Get predicted solution
        res = self.residual(x, t)   # Compute PDE residual

        return u.detach().numpy().reshape(self.original_shape), res.detach().numpy().reshape(self.original_shape)


############################################# 2nd MODEL DEFINITION #################################################

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

class BigNetwork(nn.Module):
    def __init__(self):
        super(BigNetwork, self).__init__()
        self.fourier = FourierFeatures(2, num_frequencies=6)
        self.layers = nn.Sequential(
            nn.Linear(26, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.fourier(x)
        return self.layers(x)

# Adaptive-weight PINN
class AdaPINN(nn.Module):
    def __init__(self, X, u, lb, ub, physics, nu = 0.01, original_shape = (100, 256)):
        super(AdaPINN, self).__init__()
        self.original_shape = original_shape
        self.lb = lb.clone().detach().float()
        self.ub = ub.clone().detach().float()
        self.x = X[:, 0:1].clone().detach().requires_grad_(True).float()
        self.t = X[:, 1:2].clone().detach().requires_grad_(True).float()
        self.u = torch.tensor(u).float()
        self.nu = nu
        self.network = BigNetwork()
        self.boundary_conditions = []
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.lambda_phys = 1.0
        self.lossTracker = []


    def forward(self, x, t):
        X = torch.cat([x, t], 1)
        return self.network(X)

    def grad(self, u, x, out):
        grad_u = torch.autograd.grad(u, x, out, create_graph=True)[0]
        return grad_u
    
    def residual(self, x, t):
        u = self.forward(x, t)
        u_t = self.grad(u, t, torch.ones_like(u))
        u_x = self.grad(u, x, torch.ones_like(u))
        u_xx = self.grad(u_x, x, torch.ones_like(u_x))
        return u_t + u * u_x - (self.nu / np.pi) * u_xx

    def train(self, epochs, collocation_points=10000):
        for epoch in range(epochs):
            # Sample collocation points
            x_f = (self.lb[0] + (self.ub[0] - self.lb[0]) * torch.rand((collocation_points, 1))).requires_grad_(True)
            t_f = (self.lb[1] + (self.ub[1] - self.lb[1]) * torch.rand((collocation_points, 1))).requires_grad_(True)

            residual_pred = self.residual(x_f, t_f)
            physics_loss = torch.mean(residual_pred ** 2)

            boundary_loss = 0
            for bc_func in self.boundary_conditions:
                boundary_loss += bc_func(self)

            loss = physics_loss + boundary_loss

            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch} | Physics loss: {physics_loss.item()} | Boundary loss: {boundary_loss}")

            self.lossTracker.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        plt.figure(figsize=(10, 6))
        plt.plot(self.lossTracker, label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN Training Loss")
        plt.grid()
        plt.show()

    
    
    '''
    def train(self, epochs):
        lossTracker = []
        self.network.train()
        for idx in range(epochs):
            u_pred = self.forward(self.x, self.t)
            residual_pred = self.residual(self.x, self.t)
            # data_loss = torch.mean((self.u - u_pred) ** 2)
            loss = torch.mean(residual_pred ** 2)

            # Adaptive weight
            #loss = data_loss + self.lambda_phys * physics_loss

            if idx % (epochs//10) == 0:  
                #if idx > 0:
                #    if float(physics_loss.item()) > (5 * float(data_loss.item())):
                #        self.lambda_phys *= 0.9
                #    elif float(data_loss.item()) > (5 * float(physics_loss.item())):
                #        self.lambda_phys *= 1.1
                print("Epoch: ", idx, "| Loss: ", loss.item())
                       #",| Physics Loss: ", physics_loss.item(), "| Data Loss: ", data_loss.item(), "| Lambda: ", self.lambda_phys)
            self.lossTracker.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        plt.figure(figsize=(10, 6))
        plt.plot(self.lossTracker, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
        return lossTracker
    '''
    def predict(self, x, t):
        self.network.eval()

        x = torch.tensor(x, requires_grad=True).float()
        t = torch.tensor(t, requires_grad=True).float()

        u = self.forward(x, t)
        res = self.residual(x, t)

        return u.detach().numpy().reshape(self.original_shape), res.detach().numpy().reshape(self.original_shape)


############################################# Useful Plots #################################################


def view_training_points(X_u_train, u_train, x, t, resolution=200, cmap="viridis", save_path=None):

    grid_x, grid_t = np.meshgrid(np.linspace(x.min(), x.max(), resolution),
                                 np.linspace(t.min(), t.max(), resolution))

    if isinstance(X_u_train, torch.Tensor):
        X_u_train_np = X_u_train.detach().cpu().numpy()
    else:
        X_u_train_np = X_u_train

    if isinstance(u_train, torch.Tensor):
        u_train_np = u_train.detach().cpu().numpy()
    else:
        u_train_np = u_train

    u_interp = griddata(X_u_train_np, u_train_np[:, 0], (grid_x, grid_t), method='linear')

    plt.figure(figsize=(12, 6))
    c = plt.contourf(grid_x, grid_t, u_interp, levels=200, cmap=cmap, alpha=0.8)
    plt.colorbar(c, label="Interpolated u value")

    plt.scatter(X_u_train_np[:, 0], X_u_train_np[:, 1], c=u_train_np[:, 0],
                cmap=cmap, edgecolor="black", s=30, marker='o', label="Punti di Training")

    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Campo di u ottenuto interpolando i dati di training")
    plt.legend()
    plt.grid(True)

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()

    return grid_x, grid_t, u_interp


def true_vs_predicted(x, t, usol, model, usol_shape=(100, 256)):

    X, T = np.meshgrid(x, t)
    U_pred, _ = model.predict(X.flatten()[:, None], T.flatten()[:, None])
    U_pred = U_pred.reshape(usol_shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c1 = axes[0].contourf(X, T, usol, levels=100, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("True Solution")

    c2 = axes[1].contourf(X, T, U_pred, levels=100, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("PINN Predicted Solution")

    plt.show()

def plot_time_slices_with_prediction(x, t, usol, model, usol_shape=(100, 256), interval=0.2):

    X, T = np.meshgrid(x, t)
    U_pred, _ = model.predict(X.flatten()[:, None], T.flatten()[:, None])
    U_pred = U_pred.reshape(usol_shape)

    t_points = len(t)
    time_indices = np.arange(0, t_points, int(interval * t_points))

    for idx in time_indices:
        plt.figure(figsize=(5, 4))
        
        plt.plot(x, usol[idx, :], 'b-', label="True Solution", linewidth=2)
        plt.plot(x, U_pred[idx, :], 'r--', label="wPINN Prediction", linewidth=2)
        
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"True vs PINN Prediction at t = {t[idx][0]:.2f}")
        plt.legend()
        plt.grid()
        plt.show()
