#!/bin/bash

# Script de execução e benchmark para K-means 1D
# Etapas 0 e 1: Serial e OpenMP

echo "============================================="
echo "K-means 1D - Benchmark Completo"
echo "Etapa 0: Versão Serial (Baseline)"
echo "Etapa 1: Versão OpenMP (Paralelizada)"
echo "============================================="

# Criar diretórios para resultados
mkdir -p results/serial
mkdir -p results/openmp
mkdir -p results/benchmarks

# Configurações de teste
MAX_ITER=100
EPS=1e-6

# Verificar se binários existem
if [ ! -f "./bin/kmeans_1d_serial" ]; then
    echo "ERRO: Binário serial não encontrado!"
    echo "Execute './compile.sh' primeiro."
    exit 1
fi

if [ ! -f "./bin/kmeans_1d_omp" ]; then
    echo "ERRO: Binário OpenMP não encontrado!"
    echo "Execute './compile.sh' primeiro."
    exit 1
fi

# Verificar se dados existem
if [ ! -f "data/dados_teste.csv" ]; then
    echo "ERRO: Dados não encontrados!"
    echo "Execute './generate_data.sh' primeiro."
    exit 1
fi

# Função para executar e extrair métricas
run_serial() {
    local dataset=$1
    local centroids=$2
    local output_prefix=$3
    
    # Executar e salvar saída completa
    ./bin/kmeans_1d_serial "$dataset" "$centroids" $MAX_ITER $EPS \
        "results/serial/${output_prefix}_assign.csv" \
        "results/serial/${output_prefix}_centroids.csv" \
        > "results/serial/${output_prefix}_output.txt"
        
    if [ $? -ne 0 ]; then
        echo "ERRO na execução serial do dataset $output_prefix"
        exit 1
    fi
    
    # Extrair tempo e sse
    local time_ms=$(grep "Tempo:" "results/serial/${output_prefix}_output.txt" | awk '{print $2}')
    echo $time_ms
}

# Função para executar e extrair métricas do OpenMP
run_openmp() {
    local dataset=$1
    local centroids=$2
    local threads=$3
    local output_prefix=$4
    
    # Executar e salvar saída completa
    ./bin/kmeans_1d_omp "$dataset" "$centroids" $threads $MAX_ITER $EPS \
        "results/openmp/${output_prefix}_t${threads}_assign.csv" \
        "results/openmp/${output_prefix}_t${threads}_centroids.csv" \
        > "results/openmp/${output_prefix}_t${threads}_output.txt"

    if [ $? -ne 0 ]; then
        echo "ERRO na execução OpenMP (T=$threads) do dataset $output_prefix"
        exit 1
    fi

    # Extrair tempo, sse e iters
    local metrics=$(grep -E "Tempo:|SSE final:|Iterações:" "results/openmp/${output_prefix}_t${threads}_output.txt" | awk '{print $NF}')
    echo $metrics
}

# Helper para extrair apenas o tempo (para média)
extract_time_omp() {
    grep "Tempo:" "results/openmp/${1}_output.txt" | awk '{print $2}'
}


# ============================================
# ETAPA 0: Versão Serial (Baseline)
# ============================================

echo ""
echo ""
echo "============================================="
echo "ETAPA 0: Executando Versão Serial (Baseline)"
echo "============================================="

# Teste (validação)
echo "--- Dataset TESTE (20 pontos, 4 clusters) ---"
run_serial "data/dados_teste.csv" "data/centroides_teste.csv" "teste"

# Pequeno
echo "--- Dataset PEQUENO (10k pontos, 4 clusters) ---"
serial_time_small=$(run_serial "data/dados_pequeno.csv" "data/centroides_pequeno.csv" "pequeno")
echo "Tempo Serial (Pequeno): $serial_time_small ms"

# Médio
echo "--- Dataset MÉDIO (100k pontos, 8 clusters) ---"
serial_time_medium=$(run_serial "data/dados_medio.csv" "data/centroides_medio.csv" "medio")
echo "Tempo Serial (Médio): $serial_time_medium ms"

# Grande
echo "--- Dataset GRANDE (1M pontos, 16 clusters) ---"
serial_time_large=$(run_serial "data/dados_grande.csv" "data/centroides_grande.csv" "grande")
echo "Tempo Serial (Grande): $serial_time_large ms"


# ============================================
# ETAPA 1: Versão OpenMP
# ============================================

echo ""
echo ""
echo "============================================="
echo "ETAPA 1: Executando Versão OpenMP"
echo "============================================="

# Configurações de teste
THREADS_LIST=(1 2 4 8 16)
NUM_RUNS=3 # Número de execuções para média
VERBOSE=false # Mudar para true para ver todas as saídas

declare -A datasets
datasets["pequeno"]="data/dados_pequeno.csv data/centroides_pequeno.csv $serial_time_small"
datasets["medio"]="data/dados_medio.csv data/centroides_medio.csv $serial_time_medium"
datasets["grande"]="data/dados_grande.csv data/centroides_grande.csv $serial_time_large"

# Arquivo de resultados
RESULT_FILE="results/benchmarks/speedup_results.csv"
echo "Dataset,Threads,Time_ms,SSE,Iters,Speedup,Efficiency" > $RESULT_FILE

# Loop principal
for dataset_name in "${!datasets[@]}"; do
    read -r data_file centroid_file serial_baseline <<< "${datasets[$dataset_name]}"
    
    echo ""
    echo "---------------------------------------------"
    echo "Processando Dataset: $dataset_name"
    echo "Baseline Serial: $serial_baseline ms"
    echo "---------------------------------------------"
    echo "Threads | Tempo (ms) | Speedup | Eficiência"
    echo "--------|------------|---------|------------"

    for threads in "${THREADS_LIST[@]}"; do
        
        total_time=0
        
        # Executar N vezes para média
        for ((i=1; i<=$NUM_RUNS; i++)); do
            output_prefix="${dataset_name}_run${i}"
            
            # Executar
            metrics=$(run_openmp "$data_file" "$centroid_file" $threads $output_prefix)
            
            # Pegar métricas da primeira execução
            if [ $i -eq 1 ]; then
                read -r iters_omp sse_omp time_ms_omp <<< "$metrics"
            fi

            # Extrair tempo
            time_run=$(extract_time_omp "${output_prefix}_t${threads}")
            total_time=$(echo "$total_time + $time_run" | bc -l)
        done
        
        # Calcular média
        time_ms_avg=$(echo "scale=2; $total_time / $NUM_RUNS" | bc -l)
        
        # Calcular Speedup e Eficiência
        speedup=$(echo "scale=2; $serial_baseline / $time_ms_avg" | bc -l)
        efficiency=$(echo "scale=2; ($speedup / $threads) * 100" | bc -l)
        
        # Imprimir tabela
        printf "%-7d | %-10.2f | %-7.2f | %-10.1f%%\n" $threads $time_ms_avg $speedup $efficiency

        # Salvar no CSV
        echo "$dataset_name,$threads,$time_ms_avg,$sse_omp,$iters_omp,$speedup,$efficiency" >> $RESULT_FILE
    done # Fim do loop de threads
done # Fim do loop de datasets

echo ""
echo "Speedup e Eficiência (OpenMP) salvos em: $RESULT_FILE"


if [ "$VERBOSE" == "true" ]; then
    echo ""
    echo "Exemplo de saída serial (dataset teste):"
    echo "----------------------------------------"
    if [ -f "results/serial/teste_output.txt" ]; then
        head -n 15 results/serial/teste_output.txt
    else
        echo "Arquivo não encontrado"
    fi
fi

# ==========================================================
# ETAPA 2: VERSÃO CUDA (GPU)
# ==========================================================

echo ""
echo ""
echo "============================================="
echo "Etapa 2: Versão CUDA (GPU)"
echo "============================================="

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se binário CUDA existe
if [ ! -f "./bin/kmeans_1d_cuda" ]; then
    echo -e "   ${YELLOW}⚠ AVISO: Binário CUDA (bin/kmeans_1d_cuda) não encontrado!${NC}"
    echo -e "   ${YELLOW}PULANDO${NC} Etapa 2 (CUDA)."
    echo -e "   ${YELLOW}Para executar, rode ./compile.sh em uma máquina com o CUDA Toolkit.${NC}"
else
    echo -e "   ${GREEN}✓${NC} Binário CUDA encontrado. Iniciando benchmarks..."
    mkdir -p results/cuda
    mkdir -p results/benchmarks_cuda

    # Usar os mesmos baselines da Etapa 0 (JÁ CALCULADOS)
    declare -A serial_times_cuda
    serial_times_cuda["pequeno"]=$serial_time_small
    serial_times_cuda["medio"]=$serial_time_medium
    serial_times_cuda["grande"]=$serial_time_large

    # Arquivo de resultados CUDA
    CUDA_RESULT_FILE="results/benchmarks_cuda/benchmark_results_cuda.csv"
    echo "Dataset,BlockSize,Time_ms,Kernel_ms,H2D_ms,D2H_ms,Host_ms,SSE,Iters,Speedup,Throughput_ps" > $CUDA_RESULT_FILE

    # Listas para iterar
    datasets_cuda=("pequeno" "medio" "grande")
    clusters_k_cuda=(4 8 16) # K correspondente
    block_sizes_cuda=(128 256 512)

    # Função para executar e extrair métricas CUDA
    run_cuda() {
        local dataset_name=$1
        local k=$2
        local block_size=$3
        local serial_baseline=$4
        
        local data_file="data/dados_${dataset_name}.csv"
        local centroid_file="data/centroides_${dataset_name}.csv"
        local output_file="results/cuda/output_${dataset_name}_b${block_size}.txt"
        local assign_file="results/cuda/${dataset_name}_b${block_size}_assign.csv"
        
        echo "   --- Executando CUDA: $dataset_name (K=$k) com Block Size $block_size ---"
        
        # Executar e salvar saída completa
        ./bin/kmeans_1d_cuda "$data_file" "$centroid_file" $block_size $MAX_ITER $EPS \
            "$assign_file" > "$output_file"
            
        if [ $? -ne 0 ]; then
            echo -e "   ${RED}✗ ERRO na execução do CUDA. Verifique $output_file${NC}"
            return
        fi
        
        # === INÍCIO DA CORREÇÃO (AWK) ===
        # A saída do .cu é "1. Kernel (Assignment): 10.00 ms (...)"
        # O valor numérico é o 4º campo, não o 3º.
        
        local time_ms=$(grep "Tempo Total:" "$output_file" | awk '{print $3}')
        local kernel_ms=$(grep "Kernel (Assignment):" "$output_file" | awk '{print $4}') # Corrigido de $3 para $4
        local h2d_ms=$(grep "Transfer H2D (C):" "$output_file" | awk '{print $5}')    # Corrigido de $4 para $5
        local d2h_ms=$(grep "Transfer D2H (assign):" "$output_file" | awk '{print $5}') # Corrigido de $4 para $5
        local host_ms=$(grep "Update (Host):" "$output_file" | awk '{print $4}')   # Corrigido de $3 para $4
        # === FIM DA CORREÇÃO ===

        local sse=$(grep "SSE final:" "$output_file" | awk '{print $3}')
        local iters=$(grep "Iterações:" "$output_file" | awk '{print $2}')
        local throughput=$(grep "Throughput:" "$output_file" | awk '{print $2}')
        
        # Calcular speedup (usando 'bc' para float)
        local speedup=0
        if (( $(echo "$time_ms > 0" | bc -l) )); then
            speedup=$(echo "scale=2; $serial_baseline / $time_ms" | bc -l)
        fi
        
        echo "     Tempo: $time_ms ms | Kernel: $kernel_ms ms | SSE: $sse | Speedup: ${speedup}x"
        
        # Salvar no CSV
        echo "$dataset_name,$block_size,$time_ms,$kernel_ms,$h2d_ms,$d2h_ms,$host_ms,$sse,$iters,$speedup,$throughput" >> $CUDA_RESULT_FILE
    }

    # Loop Principal de Benchmark CUDA
    for i in ${!datasets_cuda[@]}; do
        dataset_name=${datasets_cuda[$i]}
        k=${clusters_k_cuda[$i]}
        serial_baseline=${serial_times_cuda[$dataset_name]}
        
        echo ""
        echo "   Processando Dataset CUDA: $dataset_name (K=$k)"
        
        for block_size in "${block_sizes_cuda[@]}"; do
            run_cuda $dataset_name $k $block_size $serial_baseline
        done
    done
    
    echo ""
    echo -e "   ${GREEN}✓${NC} Benchmarks CUDA concluídos."
    echo "   Resultados salvos em: $CUDA_RESULT_FILE"
fi

# ============================================
# Resumo Final
# ============================================

echo ""
echo ""
echo "============================================="
echo "RESUMO DOS RESULTADOS"
echo "============================================="
echo ""
echo "Tempos baseline (serial):"
echo "  Teste:   Ver results/serial/teste_output.txt"
echo "  Pequeno: $serial_time_small ms"
echo "  Médio:   $serial_time_medium ms"
echo "  Grande:  $serial_time_large ms"
echo ""
echo "Resultados salvos em:"
echo "  - results/serial/       (outputs da versão serial)"
echo "  - results/openmp/       (outputs da versão OpenMP)"
echo "  - results/benchmarks/speedup_results.csv (dados completos OpenMP)"
# Adição para o CUDA:
if [ -f "./bin/kmeans_1d_cuda" ]; then
    echo "  - results/cuda/           (outputs da versão CUDA)"
    echo "  - results/benchmarks_cuda/benchmark_results_cuda.csv (dados completos CUDA)"
fi
echo ""
echo "Para visualizar os dados:"
echo "  cat results/benchmarks/speedup_results.csv"
# Adição para o CUDA:
if [ -f "./bin/kmeans_1d_cuda" ]; then
    echo "  cat results/benchmarks_cuda/benchmark_results_cuda.csv"
fi
echo ""
echo "Para análise detalhada:"
echo "  cat results/serial/pequeno_output.txt"
echo "  cat results/openmp/pequeno_t4_output.txt"
# Adição para o CUDA:
if [ -f "./bin/kmeans_1d_cuda" ]; then
    echo "  cat results/cuda/output_grande_b256.txt"
fi
echo ""
echo "Próximos passos:"
echo "  1. Analisar os gráficos de speedup: ./plot_results.py"
echo "  2. Verificar eficiência de paralelização"
# Adição para o CUDA:
if [ -f "./bin/kmeans_1d_cuda" ]; then
    echo "  3. Analisar gargalos de CUDA (Kernel vs. Transferência)"
fi