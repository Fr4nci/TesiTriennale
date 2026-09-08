#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double GLOBAL_SCALE = 1e-280;

struct GravityModel {
    std::string modelname;
    double earth_gravity_constant = 0.0;
    double radius = 0.0;
    int max_degree = 0;
    std::vector<std::vector<double>> C;
    std::vector<std::vector<double>> S;
};

// Parser EIGEN-6C4
GravityModel parse_gfc(const std::string& filepath, int target_max_degree) {
    GravityModel model;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire il file " << filepath << std::endl;
        return model;
    }

    std::string line;
    bool header_ended = false;

    while (std::getline(file, line)) {
        if (line.find("end_of_head") != std::string::npos) {
            header_ended = true;
            break;
        }
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "modelname") {
            iss >> model.modelname;
        } else if (key == "earth_gravity_constant") {
            iss >> model.earth_gravity_constant;
        } else if (key == "radius") {
            iss >> model.radius;
        } else if (key == "max_degree") {
            iss >> model.max_degree;
            int alloc_size = std::min(model.max_degree, target_max_degree) + 1;
            model.C.assign(alloc_size, std::vector<double>(alloc_size, 0.0));
            model.S.assign(alloc_size, std::vector<double>(alloc_size, 0.0));
        }
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // FIX: da grado 371 in poi EIGEN-6C4.gfc usa la notazione Fortran (D anziche' E)
        // per l'esponente, es. "0.983749337450D-11" invece di "9.83749337450e-12".
        // istringstream >> double NON riconosce 'D' come marcatore di esponente: legge
        // silenziosamente solo la mantissa (0.9837...) e scarta l'esponente, con un
        // errore di un fattore ~1e11 su ogni coefficiente dal grado 371 al 2190.
        for (char& c : line) {
            if (c == 'D' || c == 'd') c = 'E';
        }

        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "gfc") {
            int L, M;
            double C_val, S_val;
            iss >> L >> M >> C_val >> S_val;
            
            // Azzeramento forzato del quadrupolo se L=2, M=0 per esaltare le anomalie continentali
            if (L == 2 && M == 0) {
                C_val = 0.0; 
                S_val = 0.0;
            }

            if (L <= target_max_degree && M <= L) {
                model.C[L][M] = C_val;
                model.S[L][M] = S_val;
            }
        }
    }

    file.close();
    return model;
}

// Motore Matematico Stabile: Modified Forward Column
class ModifiedForwardColumn {
private:
    int M;
    double t; // cos(theta)
    double u; // sin(theta)
    std::vector<std::vector<double>> P_scaled;

    double a_nm(int n, int m) {
        double num = (2.0 * n - 1.0) * (2.0 * n + 1.0);
        double den = (n - m) * (n + m);
        return std::sqrt(num / den);
    }

    double b_nm(int n, int m) {
        double num = (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0);
        double den = (n - m) * (n + m) * (2.0 * n - 3.0);
        return std::sqrt(num / den);
    }

public:
    ModifiedForwardColumn(int max_degree, double colatitude_rad) : M(max_degree) {
        t = std::cos(colatitude_rad);
        u = std::sin(colatitude_rad);
        if (std::abs(u) < 1e-15) u = 1e-15; 
        P_scaled.assign(M + 1, std::vector<double>(M + 1, 0.0));
    }

    void compute() {
        P_scaled[1][1] = std::sqrt(3.0) * GLOBAL_SCALE; 
        for (int m = 2; m <= M; ++m) {
            double coeff = std::sqrt(static_cast<double>(2 * m + 1) / (2.0 * m));
            P_scaled[m][m] = coeff * P_scaled[m - 1][m - 1];
        }
        P_scaled[0][0] = 1.0 * GLOBAL_SCALE;

        for (int m = 0; m <= M; ++m) {
            for (int n = m + 1; n <= M; ++n) {
                double a = a_nm(n, m);
                double b = (n > m + 1) ? b_nm(n, m) : 0.0;

                if (n == m + 1) {
                    P_scaled[n][m] = a * t * P_scaled[n - 1][m];
                } else {
                    P_scaled[n][m] = a * t * P_scaled[n - 1][m] - b * P_scaled[n - 2][m];
                }
            }
        }
    }

    double get_scaled_P(int n, int m) const {
        return P_scaled[n][m];
    }
};

int main() {
    const int l_max_calcolo = 2190; // Può essere spinto fino a 2190 con questo algoritmo
    const int n_punti = 5000; // Risoluzione griglia (alzata a 300 per maggiore dettaglio)
    
    std::cout << "Avvio parser per EIGEN-6C4.gfc..." << std::endl;
    GravityModel eigen6c4 = parse_gfc("../EIGEN-6C4.gfc", l_max_calcolo);
    
    if (eigen6c4.radius == 0.0) {
        std::cerr << "Caricamento fallito. Assicurati che il file 'EIGEN-6C4.gfc' sia presente." << std::endl;
        return 1;
    }
    
    std::cout << "Modello caricato. Calcolo parallelo fino al grado " << l_max_calcolo << " in corso..." << std::endl;

    std::vector<double> lat(n_punti);
    std::vector<double> lon(n_punti);
    for(int i = 0; i < n_punti; ++i) {
        lon[i] = 2.0 * M_PI * i / (n_punti - 1);
        lat[i] = M_PI * i / (n_punti - 1); 
    }

    std::vector<std::vector<double>> ondulazioni(n_punti, std::vector<double>(n_punti, 0.0));

    // Parallelizzazione esterna sulle latitudini per evitare colli di bottiglia e race conditions
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_punti; ++i) {
        double theta = lat[i];
        double u = std::sin(theta);
        if (std::abs(u) < 1e-15) u = 1e-15;

        // Inizializza e calcola i coefficienti di Legendre SOLO una volta per latitudine
        ModifiedForwardColumn mfc(l_max_calcolo, theta);
        mfc.compute();
        
        std::vector<double> omega_scaled(l_max_calcolo + 1);

        for (int j = 0; j < n_punti; ++j) {
            double phi = lon[j];
            std::fill(omega_scaled.begin(), omega_scaled.end(), 0.0);

            for (int m = 0; m <= l_max_calcolo; ++m) {
                double cos_m = std::cos(m * phi);
                double sin_m = std::sin(m * phi);
                for (int n = std::max(2, m); n <= l_max_calcolo; ++n) {
                    omega_scaled[m] += mfc.get_scaled_P(n, m) * (eigen6c4.C[n][m] * cos_m + eigen6c4.S[n][m] * sin_m);
                }
            }
            
            // Schema di Horner per reintegrare le potenze del seno (u) omesse dalla ricorsione
            double acc = omega_scaled[l_max_calcolo];
            for (int m = l_max_calcolo - 1; m >= 1; --m) {
                acc = acc * u + omega_scaled[m];
            }
            acc = acc * u + omega_scaled[0];
            
            // Scalatura finale per invertire il fattore 1e-280
            ondulazioni[i][j] = eigen6c4.radius * (acc / GLOBAL_SCALE);
        }
    }

    std::ofstream file("geoide_risoluzione_alta.csv");
    for (int i = 0; i < n_punti; ++i) {
        for (int j = 0; j < n_punti; ++j) {
            file << std::fixed << std::setprecision(5) << ondulazioni[i][j];
            if (j < n_punti - 1) file << ",";
        }
        file << "\n";
    }
    file.close();

    std::cout << "Calcolo completato. Matrice esportata in 'geoide_risoluzione_alta.csv'." << std::endl;
    return 0;
}
