/* kmeans_1d_serial.c
 * K-means 1D - Versão Sequencial (Baseline)
 * Etapa 0 do Projeto
 * 
 * Compilação: mpicc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial -lm
 * Uso: ./kmeans_1d_serial dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define MAX_LINE 8192

/* ========== Utilitários CSV ========== */

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
        fprintf(stderr, "ERRO: Arquivo vazio: %s\n", path);
        exit(1);
    }
    
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) {
        fprintf(stderr, "ERRO: Sem memória para %d linhas\n", R);
        exit(1);
    }
    
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERRO: Não foi possível abrir %s\n", path);
        free(A);
        exit(1);
    }
    
    char line[MAX_LINE];
    int r = 0;
    
    while (fgets(line, sizeof(line), f) && r < R) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws) continue;
        
        const char *delim = ",; \t\n\r";
        char *tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "ERRO: Linha %d sem valor em %s\n", r+1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        
        A[r] = atof(tok);
        r++;
    }
    
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N) {
    if (!path) return;
    
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ERRO: Não foi possível abrir %s para escrita\n", path);
        return;
    }
    
    for (int i = 0; i < N; i++) {
        fprintf(f, "%d\n", assign[i]);
    }
    
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K) {
    if (!path) return;
    
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ERRO: Não foi possível abrir %s para escrita\n", path);
        return;
    }
    
    for (int c = 0; c < K; c++) {
        fprintf(f, "%.6f\n", C[c]);
    }
    
    fclose(f);
}

/* ========== K-means 1D ========== */

/* Assignment: Para cada ponto X[i], encontra cluster c com menor distância */
static double assignment_step_1d(const double *X, const double *C, int *assign,
                                 int N, int K) {
    double sse = 0.0;
    
    for (int i = 0; i < N; i++) {
        int best = -1;
        double bestd = 1e300;
        
        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        
        assign[i] = best;
        sse += bestd;
    }
    
    return sse;
}

/* Update: Recalcula centróides como média dos pontos de cada cluster */
static void update_step_1d(const double *X, double *C, const int *assign,
                           int N, int K) {
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    
    if (!sum || !cnt) {
        fprintf(stderr, "ERRO: Sem memória no update\n");
        exit(1);
    }
    
    for (int i = 0; i < N; i++) {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    /* kmeans_1d_serial.c
     * K-means 1D - Versão Sequencial (Baseline)
     * Etapa 0 do Projeto
     * 
     * Compilação: gcc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial -lm
     * Uso: ./kmeans_1d_serial dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
     */

    #define _POSIX_C_SOURCE 199309L

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <time.h>

    #define MAX_LINE 8192

    /* ========== Utilitários CSV ========== */

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
            fprintf(stderr, "ERRO: Arquivo vazio: %s\n", path);
            exit(1);
        }
    
        double *A = (double*)malloc((size_t)R * sizeof(double));
        if (!A) {
            fprintf(stderr, "ERRO: Sem memória para %d linhas\n", R);
            exit(1);
        }
    
        FILE *f = fopen(path, "r");
        if (!f) {
            fprintf(stderr, "ERRO: Não foi possível abrir %s\n", path);
            free(A);
            exit(1);
        }
    
        char line[MAX_LINE];
        int r = 0;
    
        while (fgets(line, sizeof(line), f) && r < R) {
            int only_ws = 1;
            for (char *p = line; *p; p++) {
                if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                    only_ws = 0;
                    break;
                }
            }
            if (only_ws) continue;
        
            const char *delim = ",; \t\n\r";
            char *tok = strtok(line, delim);
            if (!tok) {
                fprintf(stderr, "ERRO: Linha %d sem valor em %s\n", r+1, path);
                free(A);
                fclose(f);
                exit(1);
            }
        
            A[r] = atof(tok);
            r++;
        }
    
        fclose(f);
        *n_out = R;
        return A;
    }

    static void write_assign_csv(const char *path, const int *assign, int N) {
        if (!path) return;
    
        FILE *f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "ERRO: Não foi possível abrir %s para escrita\n", path);
            return;
        }
    
        for (int i = 0; i < N; i++) {
            fprintf(f, "%d\n", assign[i]);
        }
    
        fclose(f);
    }

    static void write_centroids_csv(const char *path, const double *C, int K) {
        if (!path) return;
    
        FILE *f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "ERRO: Não foi possível abrir %s para escrita\n", path);
            return;
        }
    
        for (int c = 0; c < K; c++) {
            fprintf(f, "%.6f\n", C[c]);
        }
    
        fclose(f);
    }

    /* ========== K-means 1D ========== */

    /* Assignment: Para cada ponto X[i], encontra cluster c com menor distância */
    static double assignment_step_1d(const double *X, const double *C, int *assign,
                                     int N, int K) {
        double sse = 0.0;
    
        for (int i = 0; i < N; i++) {
            int best = -1;
            double bestd = 1e300;
        
            for (int c = 0; c < K; c++) {
                double diff = X[i] - C[c];
                double d = diff * diff;
                if (d < bestd) {
                    bestd = d;
                    best = c;
                }
            }
        
            assign[i] = best;
            sse += bestd;
        }
    
        return sse;
    }

    /* Update: Recalcula centróides como média dos pontos de cada cluster */
    static void update_step_1d(const double *X, double *C, const int *assign,
                               int N, int K) {
        double *sum = (double*)calloc((size_t)K, sizeof(double));
        int *cnt = (int*)calloc((size_t)K, sizeof(int));
    
        if (!sum || !cnt) {
            fprintf(stderr, "ERRO: Sem memória no update\n");
            exit(1);
        }
    
        for (int i = 0; i < N; i++) {
            int a = assign[i];
            cnt[a] += 1;
            sum[a] += X[i];
        }
    
        for (int c = 0; c < K; c++) {
            if (cnt[c] > 0) {
                C[c] = sum[c] / (double)cnt[c];
            } else {
                // Cluster vazio: mantém o centróide ou usa X[0]
                C[c] = X[0];
            }
        }
    
        free(sum);
        free(cnt);
    }

    /* K-means: loop principal */
    static void kmeans_1d(const double *X, double *C, int *assign,
                          int N, int K, int max_iter, double eps,
                          int *iters_out, double *sse_out, double **sse_history) {
        double prev_sse = 1e300;
        double sse = 0.0;
        int it;
    
        // Alocar histórico de SSE
        *sse_history = (double*)malloc((size_t)(max_iter + 1) * sizeof(double));
        if (!*sse_history) {
            fprintf(stderr, "ERRO: Sem memória para histórico de SSE\n");
            exit(1);
        }
    
        for (it = 0; it < max_iter; it++) {
            sse = assignment_step_1d(X, C, assign, N, K);
            (*sse_history)[it] = sse;
        
            // Critério de parada: variação relativa do SSE
            double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        
            if (rel < eps) {
                it++;
                break;
            }
        
            update_step_1d(X, C, assign, N, K);
            prev_sse = sse;
        }
    
        *iters_out = it;
        *sse_out = sse;
    }

    /* ========== MAIN ========== */

    int main(int argc, char **argv) {
        if (argc < 3) {
            printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=100] [eps=1e-6] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho\n");
            return 1;
        }
    
        const char *pathX = argv[1];
        const char *pathC = argv[2];
        int max_iter = (argc > 3) ? atoi(argv[3]) : 100;
        double eps = (argc > 4) ? atof(argv[4]) : 1e-6;
        const char *outAssign = (argc > 5) ? argv[5] : NULL;
        const char *outCentroid = (argc > 6) ? argv[6] : NULL;
    
        if (max_iter <= 0 || eps <= 0.0) {
            fprintf(stderr, "ERRO: Parâmetros inválidos (max_iter>0 e eps>0)\n");
            return 1;
        }
    
        // Ler dados
        int N = 0, K = 0;
        double *X = read_csv_1col(pathX, &N);
        double *C = read_csv_1col(pathC, &K);
    
        int *assign = (int*)malloc((size_t)N * sizeof(int));
        if (!assign) {
            fprintf(stderr, "ERRO: Sem memória para assign\n");
            free(X);
            free(C);
            return 1;
        }
    
        // Executar K-means e medir tempo
        double *sse_history = NULL;
        int iters = 0;
        double sse = 0.0;
    
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
    
        kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse, &sse_history);
    
        clock_gettime(CLOCK_MONOTONIC, &end);
    
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_nsec - start.tv_nsec) / 1e6;
    
        // Exibir resultados
        printf("========================================\n");
        printf("K-means 1D - Versão SERIAL (Baseline)\n");
        printf("========================================\n");
        printf("Parâmetros:\n");
        printf("  N = %d pontos\n", N);
        printf("  K = %d clusters\n", K);
        printf("  max_iter = %d\n", max_iter);
        printf("  eps = %g\n", eps);
        printf("\nResultados:\n");
        printf("  Iterações: %d\n", iters);
        printf("  SSE final: %.6f\n", sse);
        printf("  Tempo: %.2f ms\n", time_ms);
        printf("  Throughput: %.2f pontos/ms\n", N / time_ms);
        printf("\nSSE por iteração:\n");
    
        for (int i = 0; i < iters; i++) {
            printf("  [%3d] SSE = %.6f", i, sse_history[i]);
            if (i > 0) {
                double diff = sse_history[i] - sse_history[i-1];
                printf(" (Δ = %.6f)", diff);
            }
            printf("\n");
        }
    
        // Salvar resultados
        if (outAssign) {
            write_assign_csv(outAssign, assign, N);
            printf("\nAssignments salvos em: %s\n", outAssign);
        }
    
        if (outCentroid) {
            write_centroids_csv(outCentroid, C, K);
            printf("Centróides salvos em: %s\n", outCentroid);
        }
    
        // Liberar memória
        free(sse_history);
        free(assign);
        free(X);
        free(C);
    
        return 0;
    }