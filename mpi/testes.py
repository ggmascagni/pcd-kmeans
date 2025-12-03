import subprocess
import re
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURAÇÕES ---
EXECUTABLE = "./kmeans_1d_mpi"
DATA_FILE = "dados.csv"       # Ajuste o caminho conforme necessário
CENTROIDS_FILE = "centroides_iniciais.csv" # Ajuste o caminho
PROCESS_COUNTS = [1, 2, 4, 8]    # Níveis de paralelismo a testar (ajuste se tiver mais cores)
ITERATIONS = 5                   # Quantas vezes rodar cada teste para tirar média
MAX_ITER_KMEANS = 100
EPSILON = 0.000001

def parse_output(output):
    """Extrai o tempo total e o tempo de comunicação do stdout."""
    time_match = re.search(r"Tempo:\s+([\d\.]+)\s+ms", output)
    comm_match = re.search(r"MPI_Allreduce total time \(sum over ranks\):\s+([\d\.]+)\s+ms", output)
    
    total_time = float(time_match.group(1)) if time_match else None
    # Se rodar com NP=1, o tempo de allreduce pode ser muito baixo ou não impresso dependendo da lógica
    comm_time = float(comm_match.group(1)) if comm_match else 0.0
    
    return total_time, comm_time

def run_benchmark():
    # Detect executable (allow .exe on Windows)
    exe_path = EXECUTABLE
    if not os.path.exists(exe_path):
        if os.path.exists(EXECUTABLE + ".exe"):
            exe_path = EXECUTABLE + ".exe"
        else:
            print(f"Erro: Executável {EXECUTABLE} não encontrado. Rode 'make' primeiro.")
            sys.exit(1)

    print(f"Iniciando Benchmark MPI...")
    print(f"Dados: {DATA_FILE}")

    results = {}  # {np: {'avg_total': float, 'avg_comm': float}}

    for proc_count in PROCESS_COUNTS:
        print(f"\n--- Testando com NP = {proc_count} ---")
        total_times = []
        comm_times = []

        for i in range(ITERATIONS):
            # try mpirun first, then mpiexec
            cmds = [
                ["mpirun", "-np", str(proc_count), exe_path, DATA_FILE, CENTROIDS_FILE, str(MAX_ITER_KMEANS), str(EPSILON)],
                ["mpiexec", "-n", str(proc_count), exe_path, DATA_FILE, CENTROIDS_FILE, str(MAX_ITER_KMEANS), str(EPSILON)],
            ]

            run_ok = False
            for cmd in cmds:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    out = result.stdout + "\n" + result.stderr
                    t_total, t_comm = parse_output(out)

                    if t_total is None:
                        print("Aviso: não foi possível parsear o tempo a partir da saída. Saída completa:")
                        print(out)
                    else:
                        total_times.append(t_total)
                        comm_times.append((t_comm or 0.0) / proc_count)
                        print(f"Run {i+1}: Total={t_total:.2f}ms | Comm(avg)={(t_comm or 0.0)/proc_count:.2f}ms")

                    run_ok = True
                    break
                except FileNotFoundError:
                    # mpirun/mpiexec not found in PATH; try next
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"Execução falhou com comando: {' '.join(cmd)}")
                    print(e.stderr)
                    # try next variant
                    continue

            if not run_ok:
                print(f"Falha: não foi possível executar com NP={proc_count} (run {i+1}).")

        if total_times:
            avg_total = sum(total_times) / len(total_times)
            avg_comm = sum(comm_times) / len(comm_times) if comm_times else 0.0
            results[proc_count] = {'total': avg_total, 'comm': avg_comm}
            print(f"Média NP={proc_count}: {avg_total:.2f} ms")

    return results

def plot_results(results):
    nps = sorted(results.keys())
    times = [results[np]['total'] for np in nps]
    comms = [results[np]['comm'] for np in nps]
    
    # Calcular Speedup (Baseado em NP=1 do próprio MPI, ou insira o tempo serial hardcoded aqui)
    t_serial = times[0] 
    speedups = [t_serial / t for t in times]
    ideal_speedup = nps

    # Gráfico 1: Speedup
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nps, speedups, 'o-', label='MPI Speedup')
    plt.plot(nps, ideal_speedup, '--', color='gray', label='Ideal')
    plt.xlabel('Número de Processos')
    plt.ylabel('Speedup')
    plt.title('Strong Scaling (MPI)')
    plt.grid(True)
    plt.legend()
    
    # Gráfico 2: Tempo Total vs Comunicação
    plt.subplot(1, 2, 2)
    plt.plot(nps, times, 's-', label='Tempo Total')
    plt.plot(nps, comms, '^-', label='Tempo Allreduce (médio)')
    plt.xlabel('Número de Processos')
    plt.ylabel('Tempo (ms)')
    plt.title('Impacto da Comunicação')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('mpi_scaling_results1M.png')
    print("\nGráfico salvo como 'mpi_scaling_results1M.png'")

if __name__ == "__main__":
    data = run_benchmark()
    if data:
        plot_results(data)