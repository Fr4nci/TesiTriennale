// simulatore_geoide_cuda.cu
//
// Versione GPU (CUDA) del simulatore di geoide da armoniche sferiche EIGEN-6C4.
// Stessa fisica e stessa matematica della versione CPU gia' validata:
//  - fix D->E per la notazione Fortran del file .gfc (grado 371-2190)
//  - Modified Forward Column per le funzioni di Legendre normalizzate
//  - stesso azzeramento di C(2,0) dell'originale (non l'ho toccato: chiedimi se
//    vuoi che lo sistemi nel modo standard)
//
// COMPILAZIONE:
//   nvcc -O3 -arch=native -o simulatore_geoide_cuda simulatore_geoide_cuda.cu
//   (se "-arch=native" non e' supportato dalla tua versione di nvcc, sostituiscilo
//   con la compute capability della tua scheda, es. -arch=sm_86 per una RTX 30xx,
//   -arch=sm_89 per una RTX 40xx: "nvidia-smi --query-gpu=compute_cap --format=csv"
//   te la dice)
//
// NON L'HO POTUTO COMPILARE NE' ESEGUIRE: nel mio ambiente non c'e' una GPU e
// installare il toolkit CUDA per la sola verifica sintattica non e' andato a
// buon fine (download che si interrompe). L'ho scritta e riverificata a mano,
// istruzione per istruzione, contro la versione CPU che invece ho validato
// numericamente end-to-end. Prima di fidartene, valida tu: vedi la nota in
// fondo al file su come confrontarla con l'output della CPU a grado ridotto.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// constexpr (non solo "const"): garantisce che sia utilizzabile sia da codice
// host che da codice device senza bisogno di __constant__/cudaMemcpyToSymbol.
constexpr double GLOBAL_SCALE = 1e-280;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "Errore CUDA: " << cudaGetErrorString(err__) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

struct GravityModel {
    std::string modelname;
    double earth_gravity_constant = 0.0;
    double radius = 0.0;
    int max_degree = 0;
    int dim = 0; // = min(max_degree, target)+1 ; indicizzazione piatta: n*dim+m
    std::vector<double> C;
    std::vector<double> S;
};

// Parser EIGEN-6C4 - identico alla versione CPU gia' validata (array piatti + fix D->E)
GravityModel parse_gfc(const std::string& filepath, int target_max_degree) {
    GravityModel model;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire il file " << filepath << std::endl;
        return model;
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.find("end_of_head") != std::string::npos) {
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
            model.dim = alloc_size;
            model.C.assign((size_t)alloc_size * alloc_size, 0.0);
            model.S.assign((size_t)alloc_size * alloc_size, 0.0);
        }
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // FIX: da grado 371 in poi EIGEN-6C4.gfc usa la notazione Fortran (D anziche' E)
        // per l'esponente. istringstream >> double non la riconosce e legge solo la
        // mantissa, con un errore di un fattore ~1e11 su ogni coefficiente.
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

            // Azzeramento forzato del quadrupolo se L=2, M=0 (invariato rispetto all'originale)
            if (L == 2 && M == 0) {
                C_val = 0.0;
                S_val = 0.0;
            }

            if (L <= target_max_degree && M <= L) {
                model.C[(size_t)L * model.dim + M] = C_val;
                model.S[(size_t)L * model.dim + M] = S_val;
            }
        }
    }

    file.close();
    return model;
}

// ---------------------------------------------------------------------------
// Coefficienti di ricorrenza (Modified Forward Column, Holmes & Featherstone 2002),
// identici alla versione CPU. __forceinline__: sono chiamate nell'hot loop.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double a_nm_dev(int n, int m) {
    double num = (2.0 * n - 1.0) * (2.0 * n + 1.0);
    double den = (double)(n - m) * (double)(n + m);
    return sqrt(num / den);
}

__device__ __forceinline__ double b_nm_dev(int n, int m) {
    double num = (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0);
    double den = (double)(n - m) * (double)(n + m) * (2.0 * n - 3.0);
    return sqrt(num / den);
}

// ---------------------------------------------------------------------------
// Kernel 1 - per una riga (colatitudine fissata, t=cos(theta)): calcola l'intera
// tabella P_scaled(n,m). Un thread per ordine m, che scorre in sequenza la
// propria colonna n=m..L (la ricorsione in n e' sequenziale per m fissato, ma
// le colonne m sono indipendenti tra loro: e' li' che sta il parallelismo).
//
// P_diag contiene il termine settoriale P_scaled(m,m). Si dimostra dalla
// ricorsione stessa (si veda la spiegazione data in chat) che questo termine
// NON dipende da theta: e' quindi precalcolato una sola volta sull'host e
// riusato identico per tutte le 5000 righe, invece di essere ricalcolato ogni
// volta come faceva la versione CPU originale.
// ---------------------------------------------------------------------------
__global__ void kernel_legendre_column(double* P_scaled, const double* P_diag,
                                        double t, int L) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m > L) return;

    const int dim = L + 1;
    double Pnm1 = P_diag[m];       // P_scaled(m,m)
    P_scaled[(size_t)m * dim + m] = Pnm1;
    double Pnm2 = 0.0;              // non usato finche' n < m+2

    for (int n = m + 1; n <= L; ++n) {
        double a = a_nm_dev(n, m);
        double val;
        if (n == m + 1) {
            val = a * t * Pnm1;
        } else {
            double b = b_nm_dev(n, m);
            val = a * t * Pnm1 - b * Pnm2;
        }
        P_scaled[(size_t)n * dim + m] = val;
        Pnm2 = Pnm1;
        Pnm1 = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 - per una riga: calcola l'ondulazione per ogni longitudine j (un
// thread per punto). Rispetto alla versione CPU, la somma e' riorganizzata per
// evitare di dover tenere in vita l'intero array omega_scaled[0..L] (troppo
// grande per i registri di un thread GPU, finirebbe in local memory): si
// scende da m=L a m=0 calcolando al volo sia la somma interna su n sia lo
// schema di Horner, con un solo accumulatore scalare "acc". E' lo stesso
// calcolo, solo riordinato: risultato numericamente identico alla versione a
// due passate della CPU (l'ho verificato passo-passo a mano, si veda la
// spiegazione data in chat).
// ---------------------------------------------------------------------------
__global__ void kernel_synthesis_row(double* out_row, const double* P_scaled,
                                      const double* C, const double* S,
                                      int L, int dim, double u, double radius,
                                      int n_punti) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_punti) return;

    // Niente doppione dell'estremo (0 e 2*pi sarebbero lo stesso meridiano):
    // spaziatura 2*pi/n_punti, non 2*pi/(n_punti-1).
    double phi = 2.0 * M_PI * (double)j / (double)n_punti;

    // Punto di partenza della ricorrenza trigonometrica DISCENDENTE: cos(L*phi),
    // sin(L*phi). L*phi arriva fino a ~L*2*pi (~13760 rad per L=2190): le
    // funzioni cos/sin di CUDA fanno una riduzione d'angolo corretta anche per
    // argomenti grandi, quindi restano accurate in doppia precisione.
    double cos_m = cos((double)L * phi);
    double sin_m = sin((double)L * phi);
    double cos_dphi = cos(phi), sin_dphi = sin(phi);

    double acc = 0.0;
    for (int m = L; m >= 0; --m) {
        int n0 = (m < 2) ? 2 : m;
        size_t idx = (size_t)n0 * dim + m;
        double inner = 0.0;
        for (int n = n0; n <= L; ++n) {
            inner += P_scaled[idx] * (C[idx] * cos_m + S[idx] * sin_m);
            idx += dim;
        }
        acc = acc * u + inner;

        if (m > 0) {
            // ricorrenza discendente: cos((m-1)*phi) = cos(m*phi)cos(phi)+sin(m*phi)sin(phi)
            //                          sin((m-1)*phi) = sin(m*phi)cos(phi)-cos(m*phi)sin(phi)
            double new_cos = cos_m * cos_dphi + sin_m * sin_dphi;
            double new_sin = sin_m * cos_dphi - cos_m * sin_dphi;
            cos_m = new_cos;
            sin_m = new_sin;
        }
    }

    out_row[j] = radius * (acc / GLOBAL_SCALE);
}

int main() {
    const int l_max_calcolo = 2190; // grado massimo di EIGEN-6C4
    const int n_punti = 10000;       // >= minimo Nyquist-Driscoll-Healy per grado 2190 (~4382)

    std::cout << "Avvio parser per EIGEN-6C4.gfc..." << std::endl;
    GravityModel eigen6c4 = parse_gfc("EIGEN-6C4.gfc", l_max_calcolo);

    if (eigen6c4.radius == 0.0) {
        std::cerr << "Caricamento fallito. Assicurati che il file 'EIGEN-6C4.gfc' sia presente." << std::endl;
        return 1;
    }
    const int dim = eigen6c4.dim;
    std::cout << "Modello caricato (dim=" << dim << "). Calcolo su GPU fino al grado "
              << l_max_calcolo << ", griglia " << n_punti << "x" << n_punti << "..." << std::endl;

    // --- Termine settoriale P_scaled(m,m): indipendente da theta (vedi sopra),
    // calcolato UNA SOLA volta sull'host (ricorsione O(L), costo trascurabile)
    // e caricato una volta sola in GPU, riusato per tutte le righe.
    std::vector<double> P_diag(dim);
    P_diag[0] = 1.0 * GLOBAL_SCALE;
    if (dim > 1) P_diag[1] = std::sqrt(3.0) * GLOBAL_SCALE;
    for (int m = 2; m < dim; ++m) {
        double coeff = std::sqrt(static_cast<double>(2 * m + 1) / (2.0 * m));
        P_diag[m] = coeff * P_diag[m - 1];
    }

    // --- Allocazioni GPU ---
    double *d_C = nullptr, *d_S = nullptr, *d_P_diag = nullptr;
    double *d_P_scaled = nullptr, *d_out_row = nullptr;
    size_t coef_bytes = (size_t)dim * dim * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_C, coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_S, coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_P_diag, (size_t)dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P_scaled, coef_bytes));           // riusato ad ogni riga
    CUDA_CHECK(cudaMalloc(&d_out_row, (size_t)n_punti * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_C, eigen6c4.C.data(), coef_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, eigen6c4.S.data(), coef_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P_diag, P_diag.data(), (size_t)dim * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> h_out_row(n_punti);
    std::ofstream file("geoide_risoluzione_alta_cuda.csv");

    const int threads1 = 256;
    const int blocks1 = (dim + threads1 - 1) / threads1;
    const int threads2 = 256;
    const int blocks2 = (n_punti + threads2 - 1) / threads2;

    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n_punti; ++i) {
        // Colatitudine: 0=polo nord, pi=polo sud. Sono punti distinti (non un
        // doppione, a differenza della longitudine), quindi qui manteniamo
        // n_punti-1 come nell'originale.
        double theta = M_PI * i / (n_punti - 1);
        double t = std::cos(theta);
        double u = std::sin(theta);
        if (std::abs(u) < 1e-15) u = 1e-15;

        kernel_legendre_column<<<blocks1, threads1>>>(d_P_scaled, d_P_diag, t, l_max_calcolo);
        CUDA_CHECK(cudaGetLastError());

        kernel_synthesis_row<<<blocks2, threads2>>>(d_out_row, d_P_scaled, d_C, d_S,
                                                      l_max_calcolo, dim, u, eigen6c4.radius,
                                                      n_punti);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_out_row.data(), d_out_row, (size_t)n_punti * sizeof(double),
                               cudaMemcpyDeviceToHost));

        for (int j = 0; j < n_punti; ++j) {
            file << std::fixed << std::setprecision(5) << h_out_row[j];
            if (j < n_punti - 1) file << ",";
        }
        file << "\n";

        if (i % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            std::cout << "riga " << i << "/" << n_punti
                      << "  (" << std::fixed << std::setprecision(1) << elapsed << " s trascorsi)"
                      << std::endl;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    file.close();

    cudaFree(d_C);
    cudaFree(d_S);
    cudaFree(d_P_diag);
    cudaFree(d_P_scaled);
    cudaFree(d_out_row);

    std::cout << "Calcolo completato in " << total_s << " s. Matrice esportata in 'geoide_risoluzione_alta.csv'." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// COME VALIDARLA (non avendo potuto testarla io):
// Prima di lanciarla alla risoluzione piena, verifica che dia gli stessi
// risultati della versione CPU (simulatore_geoide_stabile_flat.cpp) a grado e
// risoluzione ridotti, es. l_max_calcolo=100, n_punti=20 in entrambi i file
// (stesso schema che ho usato io per isolare il bug del parsing): calcola con
// entrambe, confronta i CSV con qualcosa tipo
//   python3 -c "import numpy as np; a=np.loadtxt('cpu.csv',delimiter=','); \
//               b=np.loadtxt('gpu.csv',delimiter=','); print(np.abs(a-b).max())"
// Una differenza dell'ordine di 1e-9/1e-10 o meno e' normale (arrotondamenti,
// ordine delle operazioni leggermente diverso); una differenza grande indica
// un errore da qualche parte (indicizzazione, segno nella ricorrenza discendente
// dei coseni/seni, ecc.) da isolare nello stesso modo in cui abbiamo isolato il
// bug del parsing: bisecare su l_max_calcolo/n_punti finche' non si trova dove
// le due versioni divergono.
// ---------------------------------------------------------------------------
