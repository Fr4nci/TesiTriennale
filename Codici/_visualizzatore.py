import pyvista as pv
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

colori_eigen = [
    "#000055",  # Blu scuro profondo (minimi negativi)
    "#0044cc",  # Blu intermedio
    "#00c8ff",  # Ciano chiaro
    "#00cc44",  # Verde (~0 m)
    "#ffea00",  # Giallo vivo
    "#ff4400",  # Arancione / Rosso chiaro
    "#cc0033",  # Rosso intenso
]

cmap_magenta_peak = LinearSegmentedColormap.from_list("eigen_magenta", colori_eigen)
print("Caricamento CSV in corso...")
ondulazioni_geoide = pd.read_csv('geoide_risoluzione_alta.csv', header=None).values

# Chiusura della sfera
ondulazioni_geoide = np.column_stack((ondulazioni_geoide, ondulazioni_geoide[:, 0]))
n_lat, n_lon = ondulazioni_geoide.shape

R_base = 6378137.0
esagerazione = 15000

longitudini = np.linspace(0, 2 * np.pi, n_lon)
colatitudini = np.linspace(0, np.pi, n_lat)
R_totale = R_base + (ondulazioni_geoide * esagerazione)
# Creiamo le matrici 2D come sempre
# 1. Meshgrid con indexing='ij' garantisce che:
#    Colat_grid abbia shape (n_lat, n_lon)
#    Lon_grid   abbia shape (n_lat, n_lon)
Colat_grid, Lon_grid = np.meshgrid(colatitudini, longitudini, indexing='ij')

# 2. Coordinate Cartesiane coerenti con la matrice dei dati
X = R_totale * np.sin(Colat_grid) * np.cos(Lon_grid)
Y = R_totale * np.sin(Colat_grid) * np.sin(Lon_grid)
Z = R_totale * np.cos(Colat_grid)

# 3. Costruzione PyVista esplicita:
#    VTK structured grid ordina i punti con l'indice 0 (X/colonne) che scorre per primo.
#    Quindi dimensions = [n_lon, n_lat, 1] richiede che l'array sia ordinato per (lat, lon).
mesh = pv.StructuredGrid()
mesh.dimensions = np.array([n_lon, n_lat, 1])

# Con l'appiattimento C standard (riga per riga), lon scorre veloce e lat scorre lento:
mesh.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
mesh["Anomalie Geoidiche"] = ondulazioni_geoide.ravel()
# Inizializziamo una griglia strutturata vuota per bypassare il costruttore automatico
mesh = pv.StructuredGrid()

# Dichiariamo esplicitamente la topologia (X=longitudini, Y=latitudini, Z=1 livello)
mesh.dimensions = np.array([n_lon, n_lat, 1])

# Costruiamo i vertici appiattendo le coordinate esplicitamente (il default di ravel è order="C")
mesh.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Assegniamo le anomalie usando l'identico appiattimento. 
# Ora geometria e valori combaciano al singolo pixel.
mesh["Anomalie Geoidiche"] = ondulazioni_geoide.ravel()
plotter = pv.Plotter()
# --- 1. GESTIONE COLORI E SCALA ---
impostazioni_barra = dict(title="Variazione (m)", color="white", vertical=True, fmt="%.1f")

plotter.add_mesh(
    mesh, 
    scalars="Anomalie Geoidiche", 
    cmap=cmap_magenta_peak,                 # Più luminoso e continuo di jet
    clim=[-130, 100],              # Sbilanciamento del minimo per restringere il blu scuro
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
