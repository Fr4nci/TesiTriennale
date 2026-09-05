import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

# Impostazioni base dell'ellissoide
R_base = 6378137.0
l_max = 720  # Sotto il grado 20 per evitare stalli computazionali in Python puro

# Creazione della griglia 3D
n_punti = 5000
longitudini = np.linspace(0, 2 * np.pi, n_punti)
colatitudini = np.linspace(0, np.pi, n_punti)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

# Inizializza la matrice delle ondulazioni del geoide a zero
ondulazioni_geoide_metri = np.zeros_like(Lon)

# Generazione di coefficienti fittizi decadenti (per simulare il calo di gravità ai gradi alti)
# In uno scenario reale, dovresti parsare un file di testo di 500 MB linea per linea per estrarre C_nm e S_nm
np.random.seed(42)

# --- IL MOTORE MATEMATICO ESATTO (Doppia sommatoria) ---
for n in range(2, l_max + 1):
    for m in range(0, n + 1):
        # Simuliamo coefficienti reali (sempre più piccoli all'aumentare del grado n)
        C_nm = np.random.normal(0, 1e-4 / (n**2)) 
        S_nm = np.random.normal(0, 1e-4 / (n**2))

        # sph_harm(m, n, theta, phi) calcola l'armonica complessa.
        # Estraiamo le componenti reali equivalenti ai polinomi di Legendre per il calcolo del geoide
        Y_nm = sph_harm_y(m, n, Lon, Colat)
        
        armonica = np.real(Y_nm) * C_nm + np.imag(Y_nm) * S_nm
        ondulazioni_geoide_metri += R_base * armonica

# Applica l'esagerazione per rendere visibili i lobi gravitazionali
esagerazione = 10000
R_totale = R_base + (ondulazioni_geoide_metri * esagerazione)

# Conversione sferiche -> cartesiane
X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

# --- Grafica 3D stile EIGEN-6C4 ---
fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

cmap_geoide = plt.get_cmap('jet') # La mappa colori standard usata nell'immagine fornita
Altezza_norm = (ondulazioni_geoide_metri - ondulazioni_geoide_metri.min()) / (ondulazioni_geoide_metri.max() - ondulazioni_geoide_metri.min())
colori = cmap_geoide(Altezza_norm)

surf = ax.plot_surface(X, Y, Z, facecolors=colori, rstride=1, cstride=1, antialiased=False, shade=True)
surf.set_edgecolors('none')
ax.view_init(elev=20, azim=45) 
ax.axis('off') 

# Legenda accurata
mappable = plt.cm.ScalarMappable(cmap=cmap_geoide)
mappable.set_array(ondulazioni_geoide_metri)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.05)
cbar.set_label('Ondulazione del Geoide (metri)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
