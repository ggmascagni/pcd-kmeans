#!/bin/bash

# Script de compilação para K-means 1D
# Compila TODAS as versões: Serial (Etapa 0), OpenMP (Etapa 1) e CUDA (Etapa 2)

echo "========================================="
echo "Compilando K-means 1D - Todas as Versões"
echo "========================================="
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se o gcc está disponível
if ! command -v gcc &> /dev/null; then
    echo -e "   ${RED}✗ ERRO: gcc não encontrado!${NC}"
    echo "   Não é possível compilar as versões Serial ou OpenMP."
    exit 1
fi

# Criar diretório para binários
mkdir -p bin
mkdir -p results
mkdir -p cuda # Garante que a pasta cuda exista

# Flags de compilação
CFLAGS_GCC="-O2 -std=c99 -Wall -Wextra"
LDFLAGS="-lm -lrt"

echo ""
echo "1. Compilando versão SERIAL (Etapa 0)..."
gcc $CFLAGS_GCC serial/kmeans_1d_serial.c -o bin/kmeans_1d_serial $LDFLAGS
if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓${NC} Compilado: bin/kmeans_1d_serial"
else
    echo -e "   ${RED}✗ ERRO na compilação serial${NC}"
    exit 1
fi

echo ""
echo "2. Compilando versão OPENMP (Etapa 1)..."
gcc $CFLAGS_GCC -fopenmp openmp/kmeans_1d_omp.c -o bin/kmeans_1d_omp $LDFLAGS
if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓${NC} Compilado: bin/kmeans_1d_omp"
else
    echo -e "   ${RED}✗ ERRO na compilação OpenMP${NC}"
    # Não vamos sair, o usuário pode querer continuar mesmo se o OMP falhar
fi

# --- ETAPA 2: CUDA (NVCC) ---
echo ""
echo "3. Compilando versão CUDA (Etapa 2)..."

# Verificar se o nvcc está disponível
if ! command -v nvcc &> /dev/null; then
    echo -e "   ${YELLOW}⚠ AVISO: nvcc (NVIDIA CUDA Compiler) não encontrado!${NC}"
    echo -e "   ${YELLOW}PULANDO${NC} compilação da Etapa 2 (CUDA)."
    echo -e "   ${YELLOW}Para compilar esta etapa, instale o CUDA Toolkit no WSL.${NC}"
else
    # nvcc foi encontrado, tentar compilar
    echo -e "   ${GREEN}✓${NC} nvcc encontrado, compilando..."
    
    # Verificar se o arquivo .cu existe
    if [ ! -f "cuda/kmeans_1d_cuda.cu" ]; then
         echo -e "   ${RED}✗ ERRO: Arquivo 'cuda/kmeans_1d_cuda.cu' não encontrado!${NC}"
         echo -e "   ${RED}PULANDO${NC} compilação da Etapa 2 (CUDA)."
    else
        # Flags para o NVCC
        CFLAGS_CUDA="-O2 -std=c++11 -Wno-deprecated-gpu-targets"
        
        nvcc $CFLAGS_CUDA cuda/kmeans_1d_cuda.cu -o bin/kmeans_1d_cuda $LDFLAGS
        if [ $? -eq 0 ]; then
            echo -e "   ${GREEN}✓${NC} Compilado: bin/kmeans_1d_cuda"
        else
            echo -e "   ${RED}✗ ERRO na compilação CUDA.${NC}"
            echo "   Verifique a saída do nvcc acima para mais detalhes."
        fi
    fi
fi

echo ""
echo "========================================="
echo "Compilação concluída."
echo "========================================="
echo ""
echo "Binários gerados em 'bin/':"
ls -lh bin/