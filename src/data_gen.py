import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

#Definimos los parámetros verdaderos, que la red Neuronal no va a conocer

TRUE_PARAMS = {
    'k_B': 0.2,       #Tasa de eliminación de B (1/h)
    'V_max': 10.0,    #Velocidad máxima de eliminación de A (mg/h)
    'K_m': 50.0,      #Concentración de A a la cual la velocidad es V_max/2 (mg)
    'K_i': 20.0       #Constante de inhibición de B sobre A (mg)
}

def ddi_system(t, y):
    """
    Sistema de ODEs.
    Representa la inhibición competitiva de Michaelis-Menten.
    """
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
    """
    Muestreo Log-spaced.
    Simula extracciones de sangre frecuentes al principio y espaciadas después.
    """
    #Espaciado logarítmico desde 0.1 hs hasta t_end hs
    t_log = np.geomspace(0.1, t_end, num_points - 1)
    #Agregamos el tiempo t=0 explícitamente
    t_eval = np.concatenate(([0.0], t_log))
    return t_eval

def add_combined_noise(clean_data, sigma_add=1.0, sigma_prop=0.1):
    """
    Agrega ruido aditivo y proporcional (Heterocedástico).
    """
    #Generamos ruido con distribución N(0, 1)
    epsilon = np.random.normal(0, 1, size=clean_data.shape)
    
    #Calculamos la desviación estándar heterocedástica para cada punto->suma de std aditivo y proporcional
    std_dev = np.sqrt(sigma_add**2 + (sigma_prop * clean_data)**2)
    
    #Aplicamos el ruido a los datos
    noisy_data = clean_data + epsilon * std_dev
    
    #La concentración no puede ser negativa en la vida real
    return np.maximum(noisy_data, 0.0)

def generate_dataset(A0=100.0, B0=150.0, save_to_csv=True):
    """
    Simulación y guardado del dataset.
    """
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