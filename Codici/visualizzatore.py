import numpy as np
import matplotlib.pyplot as plt

ondulazioni_geoide = np.loadtxt('geoide_risoluzione_alta.csv', delimiter=',')

n_punti = ondulazioni_geoide.shape[0]
R_base = 6378137.0
esagerazione = 800

longitudini = np.linspace(0, 2 * np.pi, n_punti)
colatitudini = np.linspace(0, np.pi, n_punti)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

# CORREZIONE: rimossa la moltiplicazione ridondante per R_base
R_totale = R_base + (ondulazioni_geoide * esagerazione)

X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

cmap_geoide = plt.get_cmap('jet')

# Normalizzazione basata sui percentili per evitare che i picchi polari appiattiscano i colori
vmin, vmax = np.percentile(ondulazioni_geoide, [2, 98])
Altezza_norm = np.clip((ondulazioni_geoide - vmin) / (vmax - vmin), 0, 1)
colori = cmap_geoide(Altezza_norm)

surf = ax.plot_surface(X, Y, Z, facecolors=colori, rstride=1, cstride=1, antialiased=False, shade=True)
surf.set_edgecolors('none')
ax.view_init(elev=20, azim=45)
ax.axis('off')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
