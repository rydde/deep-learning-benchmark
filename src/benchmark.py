from src.train import train_model, run_hyperparameter_grid
import torch
import sys
import os
import datetime

param_grid = {
        "lr": [0.001, 0.01],
        "batch_size": [64, 128],
        "optimizer_type": ["adam", "sgd"]
    }

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def benchmark():
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")
    if( os.path.exists(log_file) ):
        os.rename(log_file, log_file + ".old" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Log timestamp and environment
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n" + "="*40 + "\n")
        f.write(f"Benchmark Run: {datetime.datetime.now().isoformat()}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
        f.write("="*40 + "\n")
    sys.stdout = Logger(log_file)  # Tee print output to file

    results = {}
    #cpu_time, cpu_acc = train_model(torch.device("cpu"))
    # CPU
    device = torch.device("cpu")
    cpu_results = run_hyperparameter_grid(device, param_grid, epochs=4, patience=3, verbose=True)
    
    #results['cpu'] = {'time': cpu_time, 'accuracy': cpu_acc}
    results['cpu'] = cpu_results  # Get the first result from the grid search

    if torch.cuda.is_available():
        gpu_results = run_hyperparameter_grid(torch.device("cuda"), param_grid, epochs=4, patience=3, verbose=True)
        results['gpu'] = gpu_results
    else:
        results['gpu'] = {'time': None, 'accuracy': None}

    print("Benchmark Results:", results)
    sys.stdout.terminal.write("Benchmark Results: {}\n".format(results))  # ensure always visible
    sys.stdout = sys.stdout.terminal  # Restore printing to terminal only

    # Save summary results in a machine-readable file as well
    with open(os.path.join(log_dir, "benchmark.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{datetime.datetime.now().isoformat()} -- {results}\n")

if __name__ == "__main__":
    benchmark()