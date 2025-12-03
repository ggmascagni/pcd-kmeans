import random
import os

# Configurações sugeridas no PDF (Tamanho Pequeno/Médio)
N = 1000000  # 1 milhão de pontos
K = 4       # 4 clusters
CENTROIDS_FILE = "centroides_iniciais.csv"
DATA_FILE = "dados.csv"

print(f"Gerando {N} pontos e {K} centróides...")

# 1. Gerar Centróides Iniciais (fixos ou aleatórios)
# Vamos colocar distantes para forçar convergência: 0, 25, 50, 75
centroids = [0.0, 25.0, 50.0, 75.0]
# Se K > 4, completa com aleatórios
while len(centroids) < K:
    centroids.append(random.uniform(0, 100))

with open(CENTROIDS_FILE, "w") as f:
    for c in centroids:
        f.write(f"{c:.6f}\n")

# 2. Gerar Dados (Clusters em torno dos centróides reais para testar corretude)
# Centróides reais dos dados: 10, 30, 60, 90 (diferentes dos iniciais para o algoritmo trabalhar)
real_centers = [10.0, 30.0, 60.0, 90.0]

with open(DATA_FILE, "w") as f:
    for _ in range(N):
        # Escolhe um centro aleatório
        center = random.choice(real_centers)
        # Adiciona ruído (distribuição normal ou uniforme)
        # Ponto = centro + ruído entre -5 e 5
        val = center + random.uniform(-5, 5) 
        f.write(f"{val:.6f}\n")

print("Arquivos gerados com sucesso!")
print(f"- {DATA_FILE}")
print(f"- {CENTROIDS_FILE}")