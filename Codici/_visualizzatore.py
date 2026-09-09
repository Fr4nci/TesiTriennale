import pyvista as pv
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import cartopy.feature as cfeature

print("Caricamento CSV in corso...")
ondulazioni_geoide = pd.read_csv('geoide_risoluzione_alta.csv', header=None).values

# Chiusura della sfera
ondulazioni_geoide = np.column_stack((ondulazioni_geoide, ondulazioni_geoide[:, 0]))
n_lat, n_lon = ondulazioni_geoide.shape

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
mesh["Anomalie Geoidiche"] = ondulazioni_geoide.ravel(order="F")

plotter = pv.Plotter()
# --- 1. GESTIONE COLORI E SCALA ---
impostazioni_barra = dict(title="Variazione (m)", color="white", vertical=True, fmt="%.1f")

plotter.add_mesh(
    mesh, 
    scalars="Anomalie Geoidiche", 
    cmap="turbo",                 # Più luminoso e continuo di jet
    clim=[-130, 85],              # Sbilanciamento del minimo per restringere il blu scuro
    show_edges=False, 
    smooth_shading=True,
    interpolate_before_map=True,
    scalar_bar_args=impostazioni_barra    
)

# --- 2. GESTIONE CONTINENTI IN 3D ---
print("Estrazione coste e calcolo altitudini sul geoide deformato...")

# Creiamo un interpolatore per calcolare l'anomalia esatta in mezzo alla tua griglia
interpolatore = RegularGridInterpolator(
    (colatitudini, longitudini), 
    ondulazioni_geoide, 
    bounds_error=False, 
    fill_value=None
)

# Estraiamo le geometrie standard delle coste terrestri (110m o 50m resolution)
for feature in cfeature.COASTLINE.geometries():
    linee = feature.geoms if hasattr(feature, 'geoms') else [feature]
    
    for linea in linee:
        lon_costa, lat_costa = np.array(linea.xy)
        
        # Mappatura coordinate geografiche -> angoli del nostro sistema
        colat_costa = np.pi/2 - np.radians(lat_costa)
        lon_costa_rad = np.radians(lon_costa) % (2 * np.pi)
        
        # Interroga l'interpolatore per sapere l'altezza esatta del geoide sotto questa linea
        punti_interp = np.column_stack((colat_costa, lon_costa_rad))
        anomalia_costa = interpolatore(punti_interp)
        
        # Calcola il raggio finale e aggiungi lo 0.3% per far galleggiare le linee sopra i poligoni della mesh
        R_costa = (R_base + (anomalia_costa * esagerazione)) * 1.003 
        
        # Converti le coste in X, Y, Z cartesiane
        x_c = R_costa * np.sin(colat_costa) * np.cos(lon_costa_rad)
        y_c = R_costa * np.sin(colat_costa) * np.sin(lon_costa_rad)
        z_c = R_costa * np.cos(colat_costa)
        
        punti_3d = np.column_stack((x_c, y_c, z_c))
        n_punti = len(punti_3d)
        
        if n_punti > 1:
            # 1. Definisci l'array della linea PRIMA di creare l'oggetto
            linea_celle = np.insert(np.arange(n_punti), 0, n_punti)
            
            # 2. Crea il PolyData passando direttamente le celle, così non genera i vertici
            poly = pv.PolyData(punti_3d, lines=linea_celle)
            
            # 3. Renderizza la linea pulita
            plotter.add_mesh(poly, color="black", line_width=1)

print("Avvio rendering...")
plotter.set_background("black")
plotter.reset_camera()
plotter.camera.zoom(1.5)
plotter.show(screenshot="terra.png")
