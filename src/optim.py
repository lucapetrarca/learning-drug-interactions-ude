import torch
import torch.optim as optim
import torch.nn.functional as F

def heteroscedastic_nll_loss(pred, target, noise_params):
    pred_A, pred_B = pred[..., 0], pred[..., 1]
    targ_A, targ_B = target[..., 0], target[..., 1]
    
    #Todo estrictamente positivo
    s_add_A = F.softplus(noise_params['add_A'])
    s_prop_A = F.softplus(noise_params['prop_A'])
    s_add_B = F.softplus(noise_params['add_B'])
    s_prop_B = F.softplus(noise_params['prop_B'])

    #Varianza aditiva + proporcional
    var_A = s_add_A**2 + (s_prop_A * pred_A)**2
    var_B = s_add_B**2 + (s_prop_B * pred_B)**2
    
    #Log-Verosimilitud Negativa
    loss_A = 0.5 * (((targ_A - pred_A)**2 / var_A) + torch.log(var_A))
    loss_B = 0.5 * (((targ_B - pred_B)**2 / var_B) + torch.log(var_B))
    
    #Promedio de la pérdida total
    return torch.mean(loss_A + loss_B)


#Se entrena la UDE con ADAM + L-BFGS
def train_ude_alternating(model, t_span, u0, target_data, epochs_adam=300, epochs_lbfgs=50):
    #Se establece el ruido como tensores optimizables
    noise_params = {
        'add_A': torch.tensor([-1.0], requires_grad=True),
        'prop_A': torch.tensor([-2.0], requires_grad=True),
        'add_B': torch.tensor([-1.0], requires_grad=True),
        'prop_B': torch.tensor([-2.0], requires_grad=True)
    }
    print("\nFASE 1: ADAM")
    #Optimización alternada (para parámetros y varianza)
    adam_theta = optim.Adam(model.parameters(), lr=1e-2)
    adam_sigma = optim.Adam(noise_params.values(), lr=5e-2)
    
    for epoch in range(epochs_adam):
        #Se optimizan los pesos
        adam_theta.zero_grad()
        pred = model(u0, t_span) #Se usa Adjoint Method al hacer backward
        loss_theta = heteroscedastic_nll_loss(pred, target_data, noise_params)
        loss_theta.backward()
        adam_theta.step()
        
        #Se optimizan los parámetros de ruido
        adam_sigma.zero_grad()
        pred_detached = model(u0, t_span).detach()
        loss_sigma = heteroscedastic_nll_loss(pred_detached, target_data, noise_params)
        loss_sigma.backward()
        adam_sigma.step()
        
        if epoch % 1000 == 0:
            print(f"Época {epoch:03d} | NLL Total: {loss_theta.item():.4f}")
            
    print("\nFASE 2: L-BFGS")
    #Se fija Sigma -asumiendo convergencia- y optimizamos Theta
    lbfgs_theta = optim.LBFGS(model.parameters(), lr=0.01, max_iter=10, line_search_fn="strong_wolfe")
    
    def closure():
        lbfgs_theta.zero_grad()
        pred = model(u0, t_span)
        loss = heteroscedastic_nll_loss(pred, target_data, noise_params)
        loss.backward()
        return loss

    for epoch in range(epochs_lbfgs):
        loss_val = lbfgs_theta.step(closure)
        if epoch % 100 == 0:
            print(f"L-BFGS Paso {epoch:02d} | NLL: {loss_val.item():.6f}")

    #Se imprimen los valores finales del ruido
    print("\n[Entrenamiento Completado]")
    print("Ruido estimado (Droga A): Add=", F.softplus(noise_params['add_A']).item(), 
          "Prop=", F.softplus(noise_params['prop_A']).item())
    
    return model, noise_params