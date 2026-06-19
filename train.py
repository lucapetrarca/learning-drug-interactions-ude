import os
import torch
import numpy as np
import pandas as pd

#Se importa la arquitectura creada en src
from src.models import PharmacokineticUDE
from src.optim import train_ude_alternating

def main():
    data_path = 'data/synthetic/ddi_synthetic_data.csv'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    #Se cargan y se arman los tensores de los datos
    print(f"[*] Cargando datos desde {data_path}")
    df = pd.read_csv(data_path)
    
    t_span = torch.tensor(df['time'].values, dtype=torch.float32)
    obs_A = df['obs_A'].values
    obs_B = df['obs_B'].values
    target_data = torch.tensor(np.stack([obs_A, obs_B], axis=-1), dtype=torch.float32)
    u0 = target_data[0]

    #Se inicializa el modelo UDE híbrido
    print("[*] Inicializando UDE híbrida")
    #Se fija semilla para reproducibilidad
    torch.manual_seed(42) 
    #Se establecen 8 neuronas, más chico para forzar a la red a no memorizar el ruido
    model = PharmacokineticUDE(hidden_dim=8)

    #Se entrena con ADAM + L-BFGS
    print("[*] Comenzando entrenamiento (Adam + L-BFGS)")
    #Se usan 2500 épocas para Adam y 50 para L-BFGS para evitar overfitting
    model_trained, noise_params_trained = train_ude_alternating(
        model=model,
        t_span=t_span,
        u0=u0,
        target_data=target_data,
        epochs_adam=2500,
        epochs_lbfgs=50
    )

    #Se guardan los parámetros a disco
    print("\n[*] Guardando pesos del modelo")
    model_save_path = os.path.join(checkpoint_dir, 'ude_model_final.pth')
    noise_save_path = os.path.join(checkpoint_dir, 'noise_params_final.pth')
    
    #Se guardan los pesos de la red neuronal
    torch.save(model_trained.state_dict(), model_save_path)

    #Se guardan los tensores del ruido descubierto
    torch.save(noise_params_trained, noise_save_path)
    
    print(f"[*]Entrenamiento finalizado y guardado en '{checkpoint_dir}/'")

if __name__ == "__main__":
    main()