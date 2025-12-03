/* kmeans_1d_mpi.c
 * K-means 1D - Versão MPI (inicial)
 *
 * Compilação: mpicc -O2 -std=c99 mpi/kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
 * Uso: mpirun -np <P> ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
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

/* ========== K-means 1D (serial helpers reused) ========== */

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
            C[c] = X[0];
        }
    }

    free(sum);
    free(cnt);
}

/* K-means paralelizado com MPI
 * Strategy per iteration:
 * 1) Broadcast current centroids C to all processes
 * 2) Each process performs assignment on its local_X and computes local_sse,
 *    local_sum[K] and local_count[K]
 * 3) MPI_Reduce local_sse -> global_sse at rank 0
 *    MPI_Allreduce local_sum -> global_sum and local_count -> global_count
 * 4) All processes update C = global_sum / global_count
 * Stop criterion: rank 0 checks relative change in global_sse and broadcasts a
 * converged flag to all ranks.
 */
static void kmeans_1d_mpi(const double *local_X, int local_N, double *C, int K,
                         int max_iter, double eps,
                         int *iters_out, double *sse_out, double **sse_history,
                         int rank, int size, MPI_Comm comm,
                         int *local_assign) {
    double prev_sse = 1e300;
    double local_sse = 0.0;
    double global_sse = 0.0;
    int it;

    // sse_history stored only on rank 0
    if (rank == 0) {
        *sse_history = (double*)malloc((size_t)(max_iter + 1) * sizeof(double));
        if (!*sse_history) {
            fprintf(stderr, "ERRO: Sem memória para histórico de SSE\n");
            MPI_Abort(comm, 1);
        }
    } else {
        *sse_history = NULL;
    }

    double *local_sum = (double*)calloc((size_t)K, sizeof(double));
    double *global_sum = (double*)calloc((size_t)K, sizeof(double));
    int *local_count = (int*)calloc((size_t)K, sizeof(int));
    int *global_count = (int*)calloc((size_t)K, sizeof(int));

    if (!local_sum || !global_sum || !local_count || !global_count) {
        fprintf(stderr, "ERRO: Sem memória para buffers de soma/contagem\n");
        MPI_Abort(comm, 1);
    }

    double allreduce_time = 0.0;
    for (it = 0; it < max_iter; it++) {
        // 1) Garantir que todos tenham os centróides atuais
        MPI_Bcast(C, K, MPI_DOUBLE, 0, comm);

        // reset local accumulators
        for (int c = 0; c < K; c++) { local_sum[c] = 0.0; local_count[c] = 0; }
        local_sse = 0.0;

        // 2) Assignment local
        for (int i = 0; i < local_N; i++) {
            int best = -1;
            double bestd = 1e300;
            for (int c = 0; c < K; c++) {
                double diff = local_X[i] - C[c];
                double d = diff * diff;
                if (d < bestd) { bestd = d; best = c; }
            }
            local_assign[i] = best;
            local_sum[best] += local_X[i];
            local_count[best] += 1;
            local_sse += bestd;
        }

        // 3a) Reduce local_sse -> global_sse at rank 0
        MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

        // 3b) Allreduce sums and counts so everyone has totals
        double t0 = MPI_Wtime();
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_count, global_count, K, MPI_INT, MPI_SUM, comm);
        double t1 = MPI_Wtime();
        allreduce_time += (t1 - t0);

        // 4) Update centroids locally using global sums and counts
        for (int c = 0; c < K; c++) {
            if (global_count[c] > 0) {
                C[c] = global_sum[c] / (double)global_count[c];
            } else {
                // keep previous C[c]
            }
        }

        // Rank 0 checks convergence using global_sse
        int converged = 0;
        if (rank == 0) {
            (*sse_history)[it] = global_sse;
            double rel = fabs(global_sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
            if (rel < eps) converged = 1;
            prev_sse = global_sse;
        }

        // Broadcast converged flag to all ranks
        MPI_Bcast(&converged, 1, MPI_INT, 0, comm);

        if (converged) { it++; break; }
    }

    // After loop, ensure all processes know iters and final sse
    int iters = (rank == 0) ? it : 0;
    MPI_Bcast(&iters, 1, MPI_INT, 0, comm);

    double final_sse = 0.0;
    if (rank == 0) final_sse = prev_sse;
    MPI_Bcast(&final_sse, 1, MPI_DOUBLE, 0, comm);

    *iters_out = iters;
    *sse_out = final_sse;

    // Report allreduce cost: reduce local accumulated times to total
    double total_allreduce_time = 0.0;
    MPI_Reduce(&allreduce_time, &total_allreduce_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        double avg_allreduce_time = total_allreduce_time / (double)size;
        // convert to milliseconds
        fprintf(stdout, "MPI_Allreduce total time (sum over ranks): %.6f ms\n", total_allreduce_time * 1000.0);
        fprintf(stdout, "MPI_Allreduce average time per rank: %.6f ms\n", avg_allreduce_time * 1000.0);
    }

    free(local_sum); free(global_sum); free(local_count); free(global_count);
}

/* ========== MAIN MPI ========== */

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

    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, K = 0;
    double *X = NULL;
    double *C = NULL;

    if (rank == 0) {
        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0 || K <= 0) {
        if (rank == 0) fprintf(stderr, "ERRO: N ou K inválidos (N=%d K=%d)\n", N, K);
        MPI_Finalize();
        return 1;
    }

    int *sendcounts = (int*)malloc((size_t)size * sizeof(int));
    int *displs = (int*)malloc((size_t)size * sizeof(int));
    if (!sendcounts || !displs) {
        if (rank == 0) fprintf(stderr, "ERRO: Sem memória para sendcounts/displs\n");
        MPI_Finalize();
        return 1;
    }

    int base = N / size;
    int rem = N % size;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) displs[i] = displs[i-1] + sendcounts[i-1];

    int local_N = sendcounts[rank];
    double *local_X = (double*)malloc((size_t)local_N * sizeof(double));
    if (!local_X) {
        if (rank == 0) fprintf(stderr, "ERRO: Sem memória para local_X\n");
        free(sendcounts); free(displs);
        MPI_Finalize();
        return 1;
    }

    MPI_Scatterv(X, sendcounts, displs, MPI_DOUBLE,
                 local_X, local_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Não encerrar ranks não-zero: todos participam da iteração MPI
    // Garantir que todos tenham espaço para os centróides C
    if (rank != 0) {
        C = (double*)malloc((size_t)K * sizeof(double));
        if (!C) {
            fprintf(stderr, "ERRO: Sem memória para C no rank %d\n", rank);
            free(local_X); free(sendcounts); free(displs);
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast inicial de C para sincronizar (poderia ser feito dentro da função)
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *assign = (int*)malloc((size_t)local_N * sizeof(int));
    if (!assign) {
        fprintf(stderr, "ERRO: Sem memória para assign local\n");
        free(X); free(C); free(local_X); free(sendcounts); free(displs);
        MPI_Finalize();
        return 1;
    }

    double *sse_history = NULL;
    int iters = 0;
    double sse = 0.0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Chamar a versão MPI do kmeans, que trabalha com local_X/local_N
    kmeans_1d_mpi(local_X, local_N, C, K, max_iter, eps,
                  &iters, &sse, &sse_history, rank, size, MPI_COMM_WORLD, assign);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1e6;

    // Reunir assignments locais em Rank 0 para salvar (se solicitado)
    int *recv_assign = NULL;
    if (outAssign) {
        if (rank == 0) recv_assign = (int*)malloc((size_t)N * sizeof(int));
        MPI_Gatherv(assign, local_N, MPI_INT,
                    recv_assign, sendcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("========================================\n");
        printf("K-means 1D - Versão MPI\n");
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
    }

    if (outAssign && rank == 0) {
        write_assign_csv(outAssign, recv_assign, N);
        printf("\nAssignments salvos em: %s\n", outAssign);
        free(recv_assign);
    }

    if (outCentroid && rank == 0) {
        write_centroids_csv(outCentroid, C, K);
        printf("Centróides salvos em: %s\n", outCentroid);
    }

    if (rank == 0) free(sse_history);
    free(assign);
    free(X);
    free(C);

    free(local_X);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
