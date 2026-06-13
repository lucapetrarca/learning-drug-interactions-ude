import torch
import nn as nn
from torchdiffeq import odeint_adjoint as odeint

class InteractionNN(nn.Module):
    """
    Aproxima la tasa no lineal de eliminación competitiva.
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        #Usamos Tanh para asegurar suavidad -> campo vectorial
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, u):
        return self.net(u)


class PharmacokineticUDE(nn.Module):
    """
    Combina la parte conocida (Droga B) y la red neuronal (Droga A).
    """
    def __init__(self, hidden_dim=16, init_kB=0.5):
        super().__init__()
        self.nn_interaction = InteractionNN(hidden_dim)
        
        #Registramos k_B como un parámetro físico optimizable. 
        self.k_B = nn.Parameter(torch.tensor([init_kB], dtype=torch.float32))
        
    def dstate_dt(self, t, u):
        """
        Definimos el campo vectorial completo f(u, t, theta).
        """
        A = u[..., 0:1]
        B = u[..., 1:2]
        
        dB_dt = - self.k_B * B
        
        nn_input = torch.cat([A, B], dim=-1)
        nn_output = self.nn_interaction(nn_input)
        
        #Aplicamos Softplus para forzar que el término sea estrictamente positivo. Le ponemos un menos delante
        dA_dt = - torch.nn.functional.softplus(nn_output) * A
        
        return torch.cat([dA_dt, dB_dt], dim=-1)

    def forward(self, u0, t_span, method='lsoda'):
        #Usamos Método del Adjunto para O(1) en memoria
        options = {}
        if method == 'lsoda':
            pass
            
        pred = odeint(self.dstate_dt, u0, t_span, method='dopri5', rtol=1e-6, atol=1e-6)
        return pred