import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

#Se definen los parámetros verdaderos, que la red Neuronal no va a conocer

TRUE_PARAMS = {
    'k_B': 0.2,       #Tasa de eliminación de B (1/h)
    'V_max': 10.0,    #Velocidad máxima de eliminación de A (mg/h)
    'K_m': 50.0,      #Concentración de A a la cual la velocidad es V_max/2 (mg)
    'K_i': 20.0       #Constante de inhibición de B sobre A (mg)
}

#Sistema de ODEs que representa la inhibición competitiva de Michaelis-Menten
def ddi_system(t, y):
    A, B = y
    k_B = TRUE_PARAMS['k_B']
    V_max = TRUE_PARAMS['V_max']
    K_m = TRUE_PARAMS['K_m']
    K_i = TRUE_PARAMS['K_i']
    
    #Ecuación lineal de decaimiento de Droga B
    dB_dt = -k_B * B
    
    #Ecuación de inhibición competitiva de Droga A por B
    dA_dt = - (V_max * A) / (K_m * (1.0 + B / K_i) + A)
    
    return [dA_dt, dB_dt]

def generate_timepoints(t_end=24.0, num_points=25):
    t_log = np.geomspace(0.1, t_end, num_points - 1)
    t_eval = np.concatenate(([0.0], t_log))
    return t_eval

#Se agrega ruido heterocedástico (aditivo y proporcional)
def add_combined_noise(clean_data, sigma_add=1.0, sigma_prop=0.1):
    #Se genera ruido con distribución N(0, 1)
    epsilon = np.random.normal(0, 1, size=clean_data.shape)
    
    #Se calcula la desviación estándar heterocedástica para cada punto para la suma de std aditivo y proporcional
    std_dev = np.sqrt(sigma_add**2 + (sigma_prop * clean_data)**2)
    
    #Se aplica el ruido a los datos
    noisy_data = clean_data + epsilon * std_dev
    
    return np.maximum(noisy_data, 0.0)

def generate_dataset(A0=100.0, B0=150.0, save_to_csv=True):
    t_eval = generate_timepoints(t_end=24.0, num_points=30)
    y0 = [A0, B0]
    
    #Se resuelve el sistema limpio con el método LSODA
    sol = solve_ivp(ddi_system, [0, t_eval[-1]], y0, t_eval=t_eval, method='LSODA')
    
    clean_A = sol.y[0]
    clean_B = sol.y[1]
    
    #Se agrega ruido a los datos
    noisy_A = add_combined_noise(clean_A, sigma_add=2.0, sigma_prop=0.15)
    noisy_B = add_combined_noise(clean_B, sigma_add=1.0, sigma_prop=0.05)
    
    df = pd.DataFrame({
        'time': sol.t,
        'true_A': clean_A,
        'true_B': clean_B,
        'obs_A': noisy_A,
        'obs_B': noisy_B
    })
    
    if save_to_csv:
        os.makedirs('../data/synthetic', exist_ok=True)
        path = '../data/synthetic/ddi_synthetic_data.csv'
        df.to_csv(path, index=False)
        print(f"Dataset sintético generado y guardado en {path}")
        
    return df

if __name__ == "__main__":
    generate_dataset()