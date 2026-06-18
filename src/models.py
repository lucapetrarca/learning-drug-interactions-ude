import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torch.nn.functional as F

#La red neuronal InteractionNN aproxima la tasa no lineal de interacción/eliminación competitiva de las drogas
class InteractionNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  #Input: [A, B]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)    #Output: escalar
        )
        
    def forward(self, u):
        return self.net(u)


#La clase UDEField define el campo vectorial f(u, t, theta)
class UDEField(nn.Module):
    def __init__(self, hidden_dim=16, init_kB=0.5):
        super().__init__()
        self.nn_interaction = InteractionNN(hidden_dim)
        self.k_B = nn.Parameter(torch.tensor([init_kB], dtype=torch.float32))
        
    def forward(self, t, u):
        A = u[..., 0:1]
        B = u[..., 1:2]
        
        #Lo conocido
        dB_dt = - self.k_B * B
        
        #Lo aprendido
        nn_input = torch.cat([F.relu(A), F.relu(B)], dim=-1)
        nn_output = self.nn_interaction(nn_input)
        
        dA_dt = - torch.nn.functional.softplus(nn_output) * A
        
        return torch.cat([dA_dt, dB_dt], dim=-1)


#Se establece el modelo híbrido
class PharmacokineticUDE(nn.Module):
    def __init__(self, hidden_dim=16, init_kB=0.5):
        super().__init__()
        self.ude_field = UDEField(hidden_dim, init_kB)
        
    def forward(self, u0, t_span):
        #Se le pasa self.ude_field definido previamente
        pred = odeint(self.ude_field, u0, t_span, method='dopri5', rtol=1e-3, atol=1e-3, options={'step_size': 0.1})
        return pred