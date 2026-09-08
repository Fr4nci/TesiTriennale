import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ondulazioni_geoide = np.loadtxt('geoide_risoluzione_alta.csv', delimiter=',')
n_punti = ondulazioni_geoide.shape[0]
R_base = 6378137.0
esagerazione = 15000

# Calcoliamo quanto stiamo rimpicciolendo la matrice
stride_visivo = max(1, n_punti // 2500)

# 1. FILTRO ANTI-ALIASING
# Applichiamo un filtro gaussiano calcolato in base allo stride. 
# Questo smussa i picchi ad altissima frequenza che diventerebbero cubi durante il taglio.
#if stride_visivo > 1:
#    ondulazioni_smussate = gaussian_filter(ondulazioni_geoide, sigma=stride_visivo / 2.0)
#else:
#    ondulazioni_smussate = ondulazioni_geoide
ondulazioni_smussate = ondulazioni_geoide
longitudini = np.linspace(0, 2 * np.pi, n_punti)
colatitudini = np.linspace(0, np.pi, n_punti)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

# Usiamo i dati smussati per calcolare la geometria
R_perturbato = ondulazioni_smussate * esagerazione
R_totale = np.maximum(R_base + R_perturbato, R_base * 0.1)

X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

# 2. SLICING STRUTTURALE
X_vis = X[::stride_visivo, ::stride_visivo]
Y_vis = Y[::stride_visivo, ::stride_visivo]
Z_vis = Z[::stride_visivo, ::stride_visivo]

# 3. MAPPATURA COLORI (calcolata sui dati filtrati e poi tagliata)
cmap_geoide = plt.get_cmap('jet')
vmin_assoluto = -106.0 
vmax_assoluto = 85.0
Altezza_norm = np.clip((ondulazioni_smussate - vmin_assoluto) / (vmax_assoluto - vmin_assoluto), 0, 1)
colori = cmap_geoide(Altezza_norm)
colori_vis = colori[::stride_visivo, ::stride_visivo, :]

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# 4. RENDERING FORZATO
# rstride=1 e cstride=1 impediscono a Matplotlib di fare ulteriori danni ai nostri array.
# linewidth=0 e antialiased=False eliminano il reticolato nero sui bordi dei poligoni.
surf = ax.plot_surface(X_vis, Y_vis, Z_vis, facecolors=colori_vis, 
                       rstride=1, cstride=1, 
                       linewidth=0, antialiased=False, shade=True)
surf.set_edgecolors('none')
ax.view_init(elev=20, azim=45)
ax.axis('off')

ax.set_box_aspect([1, 1, 1])

max_range = np.array([X_vis.max()-X_vis.min(), Y_vis.max()-Y_vis.min(), Z_vis.max()-Z_vis.min()]).max() / 2.0
mid_x, mid_y, mid_z = (X_vis.max()+X_vis.min())*0.5, (Y_vis.max()+Y_vis.min())*0.5, (Z_vis.max()+Z_vis.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig("terra_fluida.png", facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=2500)
