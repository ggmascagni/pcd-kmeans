CC = mpicc
CFLAGS = -O2 -std=c99 -Wall
LDFLAGS = -lm

TARGET = kmeans_1d_mpi
SRC = kmeans_1d_mpi.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) assign.csv centroids.csv

run_test: $(TARGET)
	# Exemplo de execução com 4 processos
	mpirun -np 4 ./$(TARGET) ../dados.csv ../centroides_iniciais.csv 100 0.0001