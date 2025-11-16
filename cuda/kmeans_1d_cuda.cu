/* kmeans_1d_cuda.cu
 * K-means 1D - Versão CUDA (GPU)
 * Etapa 2 do Projeto
 *
 * Compilação: nvcc -O2 -std=c++11 cuda/kmeans_1d_cuda.cu -o bin/kmeans_1d_cuda
 * Uso: ./bin/kmeans_1d_cuda dados.csv centroides.csv [block_size] [max_iter] [eps] [assign.csv] [centroids.csv]
 */

// Headers C Padrão
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Header CUDA
#include <cuda_runtime.h>

#define MAX_LINE 8192

// Macro para verificação de erros CUDA
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "ERRO CUDA: %s - %s\n", msg, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ================================================================== */
/* =================== FUNÇÕES UTILITÁRIAS (CPU) ==================== */
/* ================================================================== */

static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERRO: Não foi possível abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[MAX_LINE];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R <= 0) {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }
    char line[MAX_LINE];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ERRO: Nao foi possivel abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ERRO: Nao foi possivel abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ================================================================== */
/* ======================= KERNEL CUDA (GPU) ======================== */
/* ================================================================== */

__global__ void assignment_kernel(const double *X, const double *C, int *assign, double *sse_errors, int N, int K) {
    // 1 thread por ponto
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    int best = -1;
    double bestd = 1e300;

    // Encontra o centróide mais próximo
    for (int c = 0; c < K; c++) {
        double diff = X[i] - C[c];
        double d = diff * diff;
        if (d < bestd) {
            bestd = d;
            best = c;
        }
    }
    
    assign[i] = best;
    sse_errors[i] = bestd; // Salva o erro deste ponto
}

/* ================================================================== */
/* =================== FUNÇÕES K-MEANS (HOST/CPU) =================== */
/* ================================================================== */

/* * Etapa de Assignment (executada na GPU)
 * Esta função host chama o kernel e reduz o SSE na CPU
 */
static double run_assignment_step_cuda(
    const double *d_X, 
    const double *d_C, 
    int *d_assign, 
    double *d_sse_errors, 
    double *h_sse_errors, // Buffer no host para copiar erros
    int N, 
    int K, 
    int block_size) 
{
    int grid_size = (N + block_size - 1) / block_size;

    // Lançar kernel
    assignment_kernel<<<grid_size, block_size>>>(d_X, d_C, d_assign, d_sse_errors, N, K);
    cudaCheckErrors("Lancamento do Kernel Assignment");

    // Copiar erros da GPU para CPU (D2H)
    cudaMemcpy(h_sse_errors, d_sse_errors, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy D2H sse_errors");

    // Reduzir (somar) SSE na CPU
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        sse += h_sse_errors[i];
    }
    return sse;
}

/* * Etapa de Update (executada na CPU/Host)
 * "Opção A" do projeto: reutiliza a lógica serial
 */
static void update_step_1d_host(const double *X, double *C, const int *assign, int N, int K) {
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if (!sum || !cnt) {
        fprintf(stderr, "ERRO: Sem memoria no update_host\n");
        exit(1);
    }

    // Acumula somas e contagens
    for (int i = 0; i < N; i++) {
        int a = assign[i];
        if (a >= 0 && a < K) {
            sum[a] += X[i];
            cnt[a] += 1;
        }
    }

    // Calcula novas médias (centróides)
    for (int c = 0; c < K; c++) {
        if (cnt[c] > 0) {
            C[c] = sum[c] / (double)cnt[c];
        } else {
            // Estratégia "naive" para cluster vazio
            C[c] = X[0]; 
        }
    }

    free(sum);
    free(cnt);
}

/* ================================================================== */
/* ========================= FUNÇÃO PRINCIPAL ======================= */
/* ================================================================== */

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Uso: %s dados.csv centroides.csv [block_size] [max_iter=100] [eps=1e-6] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("     [block_size] é obrigatório (ex: 128, 256, 512)\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int block_size = atoi(argv[3]);
    int max_iter = (argc > 4) ? atoi(argv[4]) : 100;
    double eps = (argc > 5) ? atof(argv[5]) : 1e-6;
    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    if (block_size <= 0 || max_iter <= 0 || eps <= 0.0) {
        fprintf(stderr, "Parametros invalidos: block_size>0, max_iter>0 e eps>0\n");
        return 1;
    }

    int N = 0, K = 0;
    
    // Alocar e ler dados do HOST (CPU)
    double *h_X = read_csv_1col(pathX, &N);
    double *h_C = read_csv_1col(pathC, &K);
    int *h_assign = (int*)malloc((size_t)N * sizeof(int));
    double *h_sse_errors = (double*)malloc((size_t)N * sizeof(double)); // Host buffer p/ erros
    double *sse_history = (double*)malloc((size_t)max_iter * sizeof(double));
    
    if (!h_assign || !h_sse_errors || !sse_history) {
        fprintf(stderr, "Sem memoria para buffers do host\n");
        exit(1);
    }

    // Alocar memória no DEVICE (GPU)
    double *d_X, *d_C, *d_sse_errors;
    int *d_assign;
    
    cudaMalloc((void**)&d_X, (size_t)N * sizeof(double));
    cudaCheckErrors("Malloc d_X");
    cudaMalloc((void**)&d_C, (size_t)K * sizeof(double));
    cudaCheckErrors("Malloc d_C");
    cudaMalloc((void**)&d_assign, (size_t)N * sizeof(int));
    cudaCheckErrors("Malloc d_assign");
    cudaMalloc((void**)&d_sse_errors, (size_t)N * sizeof(double));
    cudaCheckErrors("Malloc d_sse_errors");

    // Copiar dados iniciais para GPU (H2D)
    cudaMemcpy(d_X, h_X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy H2D h_X");
    cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy H2D h_C");

    // Criar eventos CUDA para medição de tempo
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_iter, stop_iter;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_iter);
    cudaEventCreate(&stop_iter);

    float ms_total = 0;
    float ms_kernel_total = 0;
    float ms_h2d_total = 0;
    float ms_d2h_total = 0;
    float ms_host_total = 0; // Tempo de update na CPU
    
    double prev_sse = 1e300;
    double sse = 0.0;
    int iters = 0;

    // Iniciar timer principal
    cudaEventRecord(start_total, 0);

    for (iters = 0; iters < max_iter; iters++) {
        
        // --- 1. ETAPA ASSIGNMENT (GPU) ---
        cudaEventRecord(start_iter, 0);
        sse = run_assignment_step_cuda(d_X, d_C, d_assign, d_sse_errors, h_sse_errors, N, K, block_size);
        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        float ms_kernel_iter = 0;
        cudaEventElapsedTime(&ms_kernel_iter, start_iter, stop_iter);
        ms_kernel_total += ms_kernel_iter;

        sse_history[iters] = sse;

        // --- 2. VERIFICAR CONVERGÊNCIA (CPU) ---
        double rel_err = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel_err < eps && iters > 0) {
            iters++; // Conta a última iteração
            break;
        }
        prev_sse = sse;

        // --- 3. ETAPA UPDATE (CPU) ---
        
        // Copiar assign (D2H) para o host
        cudaEventRecord(start_iter, 0);
        cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Memcpy D2H d_assign");
        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        float ms_d2h_iter = 0;
        cudaEventElapsedTime(&ms_d2h_iter, start_iter, stop_iter);
        ms_d2h_total += ms_d2h_iter;

        // Calcular update na CPU
        cudaEventRecord(start_iter, 0);
        update_step_1d_host(h_X, h_C, h_assign, N, K); // Atualiza h_C
        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        float ms_host_iter = 0;
        cudaEventElapsedTime(&ms_host_iter, start_iter, stop_iter);
        ms_host_total += ms_host_iter;
        
        // Copiar centróides (H2D) de volta para a GPU
        cudaEventRecord(start_iter, 0);
        cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);
        cudaCheckErrors("Memcpy H2D h_C");
        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        float ms_h2d_iter = 0;
        cudaEventElapsedTime(&ms_h2d_iter, start_iter, stop_iter);
        ms_h2d_total += ms_h2d_iter;
    }

    // Parar timer principal
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&ms_total, start_total, stop_total);

    // Copiar resultados finais da GPU para o Host para salvar
    cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Exibir resultados
    printf("========================================\n");
    printf("K-means 1D - Versão CUDA (GPU)\n");
    printf("========================================\n");
    printf("Parâmetros:\n");
    printf("  N = %d pontos\n", N);
    printf("  K = %d clusters\n", K);
    printf("  Block Size = %d\n", block_size);
    printf("  max_iter = %d\n", max_iter);
    printf("  eps = %g\n", eps);
    printf("\nResultados:\n");
    printf("  Iterações: %d\n", iters);
    printf("  SSE final: %.6f\n", sse);
    printf("\n--- Tempos de Execução (ms) ---\n");
    printf("  Tempo Total: %.2f ms\n", ms_total);
    printf("  Throughput: %.2f pontos/s\n", (double)(N * iters) / (ms_total / 1000.0));
    printf("\n--- Detalhamento dos Tempos (ms) ---\n");
    printf("  1. Kernel (Assignment): %.2f ms (%.1f%%)\n", ms_kernel_total, (ms_kernel_total/ms_total)*100.0);
    printf("  2. Update (Host):       %.2f ms (%.1f%%)\n", ms_host_total, (ms_host_total/ms_total)*100.0);
    printf("  3. Transfer H2D (C):    %.2f ms (%.1f%%)\n", ms_h2d_total, (ms_h2d_total/ms_total)*100.0);
    printf("  4. Transfer D2H (assign): %.2f ms (%.1f%%)\n", ms_d2h_total, (ms_d2h_total/ms_total)*100.0);
    float ms_comm = ms_h2d_total + ms_d2h_total;
    float ms_comp = ms_kernel_total + ms_host_total;
    printf("  --------------------------------------\n");
    printf("  Total Computação (Kernel + Host): %.2f ms (%.1f%%)\n", ms_comp, (ms_comp/ms_total)*100.0);
    printf("  Total Comunicação (H2D + D2H):  %.2f ms (%.1f%%)\n", ms_comm, (ms_comm/ms_total)*100.0);
    
    printf("\nSSE por iteração (10 primeiras):\n");
    for (int i = 0; i < iters && i < 10; i++) {
        printf("  [%3d] SSE = %.6f", i, sse_history[i]);
        if (i > 0) {
            double diff = sse_history[i] - sse_history[i-1];
            printf(" (Δ = %.6f)", diff);
        }
        printf("\n");
    }

    // Salvar arquivos de saída
    write_assign_csv(outAssign, h_assign, N);
    write_centroids_csv(outCentroid, h_C, K);
    if(outAssign) printf("\nAssignments salvos em: %s\n", outAssign);
    if(outCentroid) printf("Centróides salvos em: %s\n", outCentroid);

    // Limpeza
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_assign);
    cudaFree(d_sse_errors);
    
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_iter);
    cudaEventDestroy(stop_iter);
    
    free(h_X);
    free(h_C);
    free(h_assign);
    free(h_sse_errors);
    free(sse_history);

    return 0;
}