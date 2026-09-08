import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carica il CSV
print("Caricamento CSV in corso...")
# Aggiunti header=None per non perdere dati e .values per estrarre la matrice NumPy
ondulazioni_geoide = pd.read_csv('CSV/geoide_risoluzione_alta_cuda.csv', header=None).values

# Estrazione esplicita di righe (latitudini) e colonne (longitudini)
n_lat, n_lon = ondulazioni_geoide.shape
print(f"Griglia effettiva caricata: {n_lat} latitudini x {n_lon} longitudini.")

R_base = 6378137.0
esagerazione_anomalie = 15000      
esagerazione_schiacciamento = 300   

# Creazione della griglia basata sulle dimensioni reali della matrice
longitudini = np.linspace(0, 2 * np.pi, n_lon)
colatitudini = np.linspace(0, np.pi, n_lat)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

# 2. Ricostruzione puramente matematica del Quadrupolo (Schiacciamento polare)
C20 = -4.84165217061e-04  
t = np.cos(Colat)
P20 = np.sqrt(5) * (1.5 * t**2 - 0.5)
schiacciamento_metri = R_base * P20 * C20

# 3. Somma pesata delle due geometrie
R_perturbato = (ondulazioni_geoide * esagerazione_anomalie) + (schiacciamento_metri * esagerazione_schiacciamento)
R_totale = np.maximum(R_base + R_perturbato, R_base * 0.1)

# Conversione sferiche -> cartesiane
X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

# --- Impostazione Rendering Grafico ---
fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

cmap_geoide = plt.get_cmap('jet')

vmin_assoluto = -106.0 
vmax_assoluto = 85.0
Altezza_norm = np.clip((ondulazioni_geoide - vmin_assoluto) / (vmax_assoluto - vmin_assoluto), 0, 1)
colori = cmap_geoide(Altezza_norm)

# Stride adattato alla dimensione maggiore della griglia
# stride_visivo = max(1, max(n_lat, n_lon) // 100)

# surf = ax.plot_surface(X, Y, Z, facecolors=colori, rstride=stride_visivo, cstride=stride_visivo, antialiased=False, shade=True)
surf = ax.plot_surface(X, Y, Z, facecolors=colori, antialiased=True, shade=True)

surf.set_edgecolors('none')
ax.view_init(elev=20, azim=45)
ax.axis('off')

# Forza la bounding box a mantenere le vere proporzioni spaziali 
ax.set_box_aspect([1, 1, 1])

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig("terra_con_quadrupolo.png", facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
