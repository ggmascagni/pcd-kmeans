#!/usr/bin/env python3
"""
Strong scaling tester for kmeans_1d_mpi.

Runs the MPI executable varying P in [1,2,4,8], captures the runtime (ms) from program output,
computes speedup relative to P=1, and plots the speedup curve.

Usage: python scripts/strong_scaling.py --exe ./kmeans_1d_mpi --data data/dados_pequeno.csv --centroids data/centroides_teste.csv

The script will try `mpirun -np P` first and fall back to `mpiexec -n P` if needed.

Outputs: prints a table and saves `results/strong_scaling.png` by default.
"""

import argparse
import subprocess
import re
import sys
import os
import math
import matplotlib.pyplot as plt


def run_mpi(exe, p, data, centroids, extra_args=None):
    extra_args = extra_args or []
    cmd_variants = [
        ["mpirun", "-np", str(p), exe, data, centroids] + extra_args,
        ["mpiexec", "-n", str(p), exe, data, centroids] + extra_args,
    ]

    last_err = None
    for cmd in cmd_variants:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError as e:
            last_err = e
            continue
        except subprocess.TimeoutExpired as e:
            print(f"Command timed out: {' '.join(cmd)}")
            return None, None

        out = proc.stdout + "\n" + proc.stderr
        if proc.returncode == 0:
            return out, cmd
        else:
            # try next variant
            last_err = (proc.returncode, out)
            continue

    print("Failed to run MPI command. Last error:", last_err)
    return None, None


def parse_time_ms(output):
    # Look for line like: "Tempo: 123.45 ms"
    m = re.search(r"Tempo:\s*([0-9]+\.?[0-9]*)\s*ms", output)
    if m:
        return float(m.group(1))
    # fallback: look for 'Time' or 'Elapsed' in ms
    m = re.search(r"Time:\s*([0-9]+\.?[0-9]*)\s*ms", output)
    if m:
        return float(m.group(1))
    return None


def parse_allreduce_ms(output):
    # Look for lines printed by program about MPI_Allreduce
    # "MPI_Allreduce total time (sum over ranks): X ms"
    m = re.search(r"MPI_Allreduce total time .*?:\s*([0-9]+\.?[0-9]*)\s*ms", output)
    if m:
        return float(m.group(1))
    return None


def plot_speedup(ps, speedups, outpath):
    plt.figure()
    plt.plot(ps, speedups, marker='o', label='Measured')
    plt.plot(ps, ps, '--', label='Ideal')
    plt.xlabel('Número de processos (P)')
    plt.ylabel('Speedup (T_serial / T_P)')
    plt.title('Strong Scaling - Speedup')
    plt.xticks(ps)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    print(f"Saved plot to {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', default='./kmeans_1d_mpi', help='Path to MPI executable')
    parser.add_argument('--data', default='data/dados_pequeno.csv', help='Path to data CSV')
    parser.add_argument('--centroids', default='data/centroides_teste.csv', help='Path to initial centroids CSV')
    parser.add_argument('--processes', default='1,2,4,8', help='Comma-separated list of P values')
    parser.add_argument('--out', default='results/strong_scaling.png', help='Output plot path')
    parser.add_argument('--extra-args', default='', help='Extra args to pass to the executable')
    args = parser.parse_args()

    ps = [int(x) for x in args.processes.split(',') if x.strip()]
    times = {}
    allreduce_times = {}

    for p in ps:
        print(f"Running P={p} ...")
        out, cmd = run_mpi(args.exe, p, args.data, args.centroids, args.extra_args.split())
        if out is None:
            print(f"Run failed for P={p}")
            sys.exit(1)
        t = parse_time_ms(out)
        ar = parse_allreduce_ms(out)
        if t is None:
            print("Warning: could not parse execution time from program output")
            print(out)
            sys.exit(1)
        times[p] = t
        allreduce_times[p] = ar
        print(f"P={p}: time = {t} ms, MPI_Allreduce total (sum ranks) = {ar} ms")

    # compute speedups relative to P=1
    if 1 not in times:
        print("Error: baseline P=1 result not found. Ensure '1' is included in --processes")
        sys.exit(1)

    t1 = times[1]
    speedups = [t1 / times[p] for p in ps]

    # print table
    print('\nResults:')
    print('P\tTime (ms)\tSpeedup\tMPI_Allreduce_total(ms)')
    for p, s in zip(ps, speedups):
        print(f"{p}\t{times[p]:.3f}\t{s:.3f}\t{(allreduce_times[p] if allreduce_times[p] is not None else float('nan'))}")

    plot_speedup(ps, speedups, args.out)


if __name__ == '__main__':
    main()
