#!/usr/bin/env python3

"""
Script para plotar os resultados de benchmark das Etapas 1 (OpenMP) e 2 (CUDA)
do projeto K-means 1D.

Gera os gráficos de Speedup, Eficiência e Tempo de Execução.
Adiciona gráficos comparativos (Serial vs OMP vs CUDA) e análise de gargalo CUDA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# --- CONFIGURAÇÕES ---

# Caminhos dos arquivos de resultados
OMP_CSV_FILE = "results/benchmarks/speedup_results.csv"
CUDA_CSV_FILE = "results/benchmarks_cuda/benchmark_results_cuda.csv"

# Tempos de baseline da versão SERIAL (extraídos do seu "RESUMO DOS RESULTADOS")
SERIAL_BASELINE_TIMES = {
    'pequeno': 0.34,
    'medio': 4.65,
    'grande': 70.81
}

# Diretório para salvar gráficos
OUTPUT_DIR = "results/benchmarks"

# Cores e Estilos (Mantidos do original)
COLORS = {
    'pequeno': 'C0',  # Azul
    'medio':   'C1',  # Laranja
    'grande':  'C2',  # Verde
    'Serial':  'C3',  # Vermelho
    'OpenMP':  'C0',  # Azul (para gráficos comparativos)
    'CUDA':    'C2',  # Verde (para gráficos comparativos)
}
MARKERS = ['o', 's', 'D', '^']
LINESTYLES = ['-', '--', ':', '-.']

# --- FUNÇÕES DE CARREGAMENTO DE DADOS ---

def load_openmp_data(file_path):
    """Carrega dados do OpenMP CSV."""
    if not os.path.exists(file_path):
        print(f"AVISO: Arquivo OpenMP não encontrado: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        # Assegura que 'Dataset' é uma categoria
        df['Dataset'] = df['Dataset'].astype('category')
        return df
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

def load_cuda_data(file_path):
    """Carrega dados do CUDA CSV."""
    if not os.path.exists(file_path):
        print(f"AVISO: Arquivo CUDA não encontrado: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        # Assegura que 'Dataset' é uma categoria
        df['Dataset'] = df['Dataset'].astype('category')
        return df
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

# --- FUNÇÕES DE PLOTAGEM (OPENMP DETALHADO - SEM ALTERAÇÕES NA LÓGICA) ---

def plot_speedup(df):
    """Plota Speedup vs. Número de Threads para OpenMP."""
    if df is None:
        return
    
    plt.figure(figsize=(10, 6))
    datasets = df['Dataset'].unique()
    
    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset].sort_values(by='Threads')
        plt.plot(subset['Threads'], subset['Speedup'], 
                 label=f'Dataset {dataset}',
                 marker=MARKERS[i % len(MARKERS)], 
                 color=COLORS.get(dataset, f'C{i}'))

    # Linha de speedup ideal
    max_threads = df['Threads'].max()
    ideal_speedup = [1, max_threads]
    plt.plot([1, max_threads], ideal_speedup, 
             label='Speedup Ideal', 
             linestyle='--', 
             color='gray')

    plt.title('K-means 1D - Speedup OpenMP vs. Threads', fontsize=16)
    plt.xlabel('Número de Threads', fontsize=12)
    plt.ylabel('Speedup (S = T_serial / T_omp)', fontsize=12)
    plt.xticks(df['Threads'].unique())
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "speedup_plot_omp.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico de Speedup OMP salvo em: {output_path}")

def plot_efficiency(df):
    """Plota Eficiência vs. Número de Threads para OpenMP."""
    if df is None:
        return
        
    plt.figure(figsize=(10, 6))
    datasets = df['Dataset'].unique()

    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset].sort_values(by='Threads')
        plt.plot(subset['Threads'], subset['Efficiency'], 
                 label=f'Dataset {dataset}',
                 marker=MARKERS[i % len(MARKERS)], 
                 color=COLORS.get(dataset, f'C{i}'))

    # Linha de eficiência ideal
    max_threads = df['Threads'].max()
    plt.axhline(y=100, label='Eficiência Ideal (100%)', linestyle='--', color='gray')

    plt.title('K-means 1D - Eficiência OpenMP vs. Threads', fontsize=16)
    plt.xlabel('Número de Threads', fontsize=12)
    plt.ylabel('Eficiência (E = S / P * 100%)', fontsize=12)
    plt.xticks(df['Threads'].unique())
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(0, 105) # Eficiência não pode passar de 100
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "efficiency_plot_omp.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico de Eficiência OMP salvo em: {output_path}")

def plot_time(df):
    """Plota Tempo de Execução vs. Número de Threads para OpenMP."""
    if df is None:
        return
        
    plt.figure(figsize=(10, 6))
    datasets = df['Dataset'].unique()

    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset].sort_values(by='Threads')
        plt.plot(subset['Threads'], subset['Time_ms'], 
                 label=f'Dataset {dataset}',
                 marker=MARKERS[i % len(MARKERS)], 
                 color=COLORS.get(dataset, f'C{i}'))

    plt.title('K-means 1D - Tempo de Execução OpenMP vs. Threads', fontsize=16)
    plt.xlabel('Número de Threads', fontsize=12)
    plt.ylabel('Tempo de Execução (ms)', fontsize=12)
    plt.xticks(df['Threads'].unique())
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.yscale('log') # Usar escala logarítmica para tempo
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "execution_time_plot_omp.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico de Tempo OMP salvo em: {output_path}")


# --- FUNÇÕES DE PLOTAGEM (NOVAS - COMPARATIVAS) ---

def plot_comparison_time(serial_times, omp_best, cuda_best):
    """
    Plota um gráfico de barras comparando o MELHOR tempo de execução
    entre Serial, OpenMP e CUDA.
    """
    if omp_best is None and cuda_best is None:
        return

    datasets = ['pequeno', 'medio', 'grande']
    
    # Extrair tempos
    serial_vals = [serial_times.get(d, np.nan) for d in datasets]
    omp_vals = [omp_best.loc[d]['Time_ms'] if omp_best is not None and d in omp_best.index else np.nan for d in datasets]
    cuda_vals = [cuda_best.loc[d]['Time_ms'] if cuda_best is not None and d in cuda_best.index else np.nan for d in datasets]
    
    data = {
        'Serial': serial_vals,
        'OpenMP (Melhor)': omp_vals,
        'CUDA (Melhor)': cuda_vals,
    }
    
    index = ['Pequeno', 'Médio', 'Grande']
    df_plot = pd.DataFrame(data, index=index)
    
    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind='bar', 
                      logy=True, 
                      color=[COLORS['Serial'], COLORS['OpenMP'], COLORS['CUDA']],
                      alpha=0.8)
    
    plt.title('Comparativo: Melhor Tempo de Execução (Serial vs. OMP vs. CUDA)', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Tempo de Execução (ms) - Escala Log', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Adicionar rótulos de valor
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.2f} ms", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom',
                        fontsize=9,
                        xytext=(0, 5),
                        textcoords='offset points')
                    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "comparison_time_plot.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico Comparativo de TEMPO salvo em: {output_path}")

def plot_comparison_speedup(serial_times, omp_best, cuda_best):
    """
    Plota um gráfico de barras comparando o MÁXIMO speedup
    entre OpenMP e CUDA.
    """
    if omp_best is None and cuda_best is None:
        return

    datasets = ['pequeno', 'medio', 'grande']
    
    # Calcular speedups
    omp_speedups = [omp_best.loc[d]['Speedup'] if omp_best is not None and d in omp_best.index else np.nan for d in datasets]
    
    cuda_speedups = []
    if cuda_best is not None:
        for d in datasets:
            # CORREÇÃO: Forçar dados a serem numéricos antes da divisão
            try:
                serial_time = float(serial_times[d])
                cuda_time = float(cuda_best.loc[d]['Time_ms'])
                if serial_time > 0 and cuda_time > 0:
                    speedup = serial_time / cuda_time
                    cuda_speedups.append(speedup)
                else:
                    cuda_speedups.append(np.nan)
            except (KeyError, ValueError, TypeError):
                cuda_speedups.append(np.nan)
    else:
        cuda_speedups = [np.nan] * len(datasets)
        
    data = {
        'OpenMP (Max)': omp_speedups,
        'CUDA (Max)': cuda_speedups,
    }
    
    index = ['Pequeno', 'Médio', 'Grande']
    df_plot = pd.DataFrame(data, index=index)
    
    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind='bar', 
                      color=[COLORS['OpenMP'], COLORS['CUDA']],
                      alpha=0.8)
    
    plt.title('Comparativo: Speedup Máximo (vs. Serial)', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Adicionar rótulos de valor
    for p in ax.patches:
         if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.2f}x", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom',
                        fontsize=10,
                        xytext=(0, 5),
                        textcoords='offset points')
                    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "comparison_speedup_plot.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico Comparativo de SPEEDUP salvo em: {output_path}")

def plot_cuda_bottleneck(cuda_best):
    """
    Plota um gráfico de barras empilhadas mostrando o gargalo do CUDA
    (Kernel vs. Transferência vs. Host).
    """
    if cuda_best is None:
        return
        
    # Colunas de tempo para o gargalo
    time_cols = ['Kernel_ms', 'H2D_ms', 'D2H_ms', 'Host_ms']
    
    # === INÍCIO DA CORREÇÃO ===
    # Forçar colunas a serem numéricas, tratando erros.
    # O 'coerce' transformará strings (como '(Assignment):') em NaN (Not a Number)
    df_plot_numeric = cuda_best[time_cols].copy()
    for col in time_cols:
        df_plot_numeric[col] = pd.to_numeric(df_plot_numeric[col], errors='coerce')
    
    # Substituir NaNs por 0 para o plot empilhado não falhar
    df_plot_numeric = df_plot_numeric.fillna(0)
    
    # Avisar se os dados estiverem faltando
    if df_plot_numeric.empty or df_plot_numeric.sum().sum() == 0:
        print("\nAVISO: Não foi possível plotar o gráfico de gargalo CUDA.")
        print("Verifique se 'benchmark_results_cuda.csv' contém dados numéricos corretos.")
        print("Pode ser necessário corrigir e re-executar 'run_tests.sh'.\n")
        return
    # === FIM DA CORREÇÃO ===

    # Renomear para o gráfico
    df_plot = df_plot_numeric.rename(columns={
        'Kernel_ms': 'Computação (Kernel GPU)',
        'H2D_ms': 'Transferência (CPU->GPU)',
        'D2H_ms': 'Transferência (GPU->CPU)',
        'Host_ms': 'Computação (Update CPU)'
    })
    
    # Reordenar datasets
    df_plot = df_plot.reindex(['pequeno', 'medio', 'grande'])
    df_plot.index = ['Pequeno', 'Médio', 'Grande']
    
    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind='bar', stacked=True, alpha=0.85)
    
    plt.title('Análise de Gargalo CUDA (Melhor Caso por Dataset)', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Tempo de Execução (ms)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Componente de Tempo', fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para a legenda
    output_path = os.path.join(OUTPUT_DIR, "cuda_bottleneck_plot.png")
    plt.savefig(output_path)
    print(f"✓ Gráfico de Gargalo CUDA salvo em: {output_path}")

# --- FUNÇÃO PRINCIPAL ---

def main():
    """Função principal para carregar dados e gerar todos os gráficos."""
    # ADAPTADO: Título atualizado
    print("======================================================")
    print("K-means 1D - Análise de Resultados (OMP e CUDA)")
    print("======================================================")
    
    # --- Carregar Dados ---
    print(f"Carregando dados OpenMP de: {OMP_CSV_FILE}")
    df_omp = load_openmp_data(OMP_CSV_FILE)
    
    print(f"Carregando dados CUDA de: {CUDA_CSV_FILE}")
    df_cuda = load_cuda_data(CUDA_CSV_FILE)
    
    if df_omp is None and df_cuda is None:
        print("Nenhum dado encontrado. Saindo.")
        return

    # --- Processar "Melhores" Resultados ---
    omp_best = None
    if df_omp is not None:
        # Encontra o índice (linha) do Speedup MÁXIMO para cada dataset
        omp_best_idx = df_omp.groupby('Dataset')['Speedup'].idxmax()
        omp_best = df_omp.loc[omp_best_idx].set_index('Dataset')
        print("\nMelhores resultados OpenMP (Max Speedup):")
        print(omp_best[['Threads', 'Time_ms', 'Speedup']])
        
    cuda_best = None
    if df_cuda is not None:
        # Encontra o índice (linha) do Tempo MÍNIMO para cada dataset
        cuda_best_idx = df_cuda.groupby('Dataset')['Time_ms'].idxmin()
        cuda_best = df_cuda.loc[cuda_best_idx].set_index('Dataset')
        print("\nMelhores resultados CUDA (Min Time):")
        # CORREÇÃO: Força a coluna a ser numérica ANTES de imprimir, para evitar o erro visual
        cuda_best_to_print = cuda_best[['BlockSize', 'Time_ms', 'Kernel_ms']].copy()
        cuda_best_to_print['Kernel_ms'] = pd.to_numeric(cuda_best_to_print['Kernel_ms'], errors='coerce').fillna(0)
        print(cuda_best_to_print)
    
    # --- Gerar Gráficos ---
    print("\nGerando gráficos...")
    
    # Gráficos Originais (MANTER IDÊNTICO)
    plot_speedup(df_omp)
    plot_efficiency(df_omp)
    plot_time(df_omp)
    
    # Novos Gráficos (Comparativos)
    plot_comparison_time(SERIAL_BASELINE_TIMES, omp_best, cuda_best)
    plot_comparison_speedup(SERIAL_BASELINE_TIMES, omp_best, cuda_best)
    
    # Gráfico Específico do CUDA
    plot_cuda_bottleneck(cuda_best)

    print("\n======================================================")
    print("Análise concluída!")
    print(f"Gráficos salvos em: {OUTPUT_DIR}")
    print("======================================================")

if __name__ == "__main__":
    # ADAPTADO: Garantir que o diretório exista e silenciar avisos
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()