import numpy as np
import matplotlib.pyplot as plt

ondulazioni_geoide = np.loadtxt('geoide_risoluzione_alta.csv', delimiter=',')
n_punti = ondulazioni_geoide.shape[0]
R_base = 6378137.0
esagerazione = 15000

longitudini = np.linspace(0, 2 * np.pi, n_punti)
colatitudini = np.linspace(0, np.pi, n_punti)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

# Prevenzione per evitare che anomalie estreme invertano il raggio bucando la mesh
R_perturbato = ondulazioni_geoide * esagerazione
R_totale = np.maximum(R_base + R_perturbato, R_base * 0.1)

X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

cmap_geoide = plt.get_cmap('jet')

# Mappatura cromatica assoluta sui limiti fisici reali del geoide terrestre (in metri)
vmin_assoluto = -106.0 
vmax_assoluto = 85.0
Altezza_norm = np.clip((ondulazioni_geoide - vmin_assoluto) / (vmax_assoluto - vmin_assoluto), 0, 1)
colori = cmap_geoide(Altezza_norm)
# Campionamento dinamico per impedire il blocco del rendering su griglie 5000x5000
stride_visivo = max(1, n_punti // 300)

surf = ax.plot_surface(X, Y, Z, facecolors=colori, rstride=stride_visivo, cstride=stride_visivo, antialiased=False, shade=True)
surf.set_edgecolors('none')
ax.view_init(elev=20, azim=45)
ax.axis('off')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Forza le proporzioni degli assi 3D a essere un cubo perfetto
ax.set_box_aspect([1, 1, 1])

# Salvataggio con ereditarietà del facecolor per mantenere lo sfondo nero
plt.savefig("terra.png", facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
