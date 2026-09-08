import pyvista as pv
import numpy as np
import pandas as pd

print("Caricamento CSV in corso...")
ondulazioni_geoide = pd.read_csv('geoide_risoluzione_alta.csv', header=None).values

# 1. CHIUSURA DELLA SFERA: Copiamo la prima colonna (0°) e la aggiungiamo alla fine (360°)
# In questo modo i vertici combaciano al millimetro chiudendo la mesh senza tagli fisici.
ondulazioni_geoide = np.column_stack((ondulazioni_geoide, ondulazioni_geoide[:, 0]))

n_lat, n_lon = ondulazioni_geoide.shape
print(f"Griglia {n_lat}x{n_lon} caricata in RAM. Costruzione geometria...")

R_base = 6378137.0
esagerazione = 15000

longitudini = np.linspace(0, 2 * np.pi, n_lon)
colatitudini = np.linspace(0, np.pi, n_lat)
Lon, Colat = np.meshgrid(longitudini, colatitudini)

R_totale = R_base + (ondulazioni_geoide * esagerazione)

X = R_totale * np.sin(Colat) * np.cos(Lon)
Y = R_totale * np.sin(Colat) * np.sin(Lon)
Z = R_totale * np.cos(Colat)

mesh = pv.StructuredGrid(X, Y, Z)

# 2. FIX MEMORIA (order="F"): I colori si allineano perfettamente alla geometria 3D
mesh["Anomalie Geoidiche"] = ondulazioni_geoide.ravel(order="F")

print("Avvio motore grafico...")
plotter = pv.Plotter(window_size=[1920,1080], off_screen=True)
plotter.add_mesh(
    mesh, 
    scalars="Anomalie Geoidiche", 
    cmap="jet", 
    show_edges=False, 
    smooth_shading=True,
    interpolate_before_map=True # Risolve i salti di colore sfumando al pixel
)
plotter.set_background("black")
plotter.screenshot("terra.png")
plotter.show()

