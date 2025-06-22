from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from src.resnet_module import SimpleResNet
from src.utils import get_cifar10_loaders
import time
import os

def save_model(model, device_str, accuracy, params):
    os.makedirs("models", exist_ok=True)
    param_str = "_".join([f"{k}{v}" for k,v in params.items()])
    model_path = f"models/best_model_{device_str}_{param_str}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'device': device_str,
        'hyperparameters': params
    }, model_path)
    print(f"Model saved to {model_path} with accuracy {accuracy:.2f}%")

def train_model(
    device,
    epochs=5,
    patience=5,
    lr=0.0001,
    batch_size=128,
    optimizer_type="adam",
    verbose=True,
    save_best=True,
    param_id="default"
):
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    model = SimpleResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    if verbose: print(f"Training on {device} with lr={lr}, batch_size={batch_size}, optimizer={optimizer_type}, patience={patience}")
    start_time = time.time()
    best_acc = 0.0
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if verbose and i % 100 == 0:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/(i+1):.4f}")
        # Evaluate and save best model checkpoint
        acc = test_model(model, testloader, device)
        if verbose: print(f"Epoch {epoch+1}: Test Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            epochs_since_improve = 0
            if save_best:
                save_model(model, str(device), best_acc, {
                    "lr": lr,
                    "batch": batch_size,
                    "optimizer": optimizer_type,
                    "paramid": param_id
                })
        else:
            epochs_since_improve += 1
        # Early stopping
        if epochs_since_improve >= patience:
            if verbose: print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    training_time = time.time() - start_time
    if verbose:
        print(f"Finished Training in {training_time:.2f} seconds")
        print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    return {
        "training_time": training_time,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "hyperparameters": {
            "lr": lr,
            "batch": batch_size,
            "optimizer": optimizer_type,
            "patience": patience,
            "paramid": param_id
        }
    }

def test_model(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    # CPU
    cpu_time, cpu_acc = train_model(torch.device("cpu"))
    # GPU (if available)
    if torch.cuda.is_available():
        gpu_time, gpu_acc = train_model(torch.device("cuda"))
    else:
        print("CUDA is not available. Skipping GPU benchmark.")


def run_hyperparameter_grid(
    device,
    param_grid,
    epochs=20,
    patience=5,
    verbose=True
):
    # param_grid: dict, e.g., {'lr': [0.001, 0.01], 'batch_size': [64, 128], 'optimizer_type': ["adam", "sgd"]}
    keys = list(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))
    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        params['param_id'] = f"run{i+1}"
        if verbose:
            print("\n"+"="*30)
            print(f"Grid search run {i+1}/{len(combos)}: {params}")
        res = train_model(
            device=device,
            epochs=epochs,
            patience=patience,
            lr=params['lr'],
            batch_size=params['batch_size'],
            optimizer_type=params['optimizer_type'],
            verbose=verbose,
            save_best=True,
            param_id=params['param_id']
        )
        results.append(res)
    return results

if __name__ == "__main__":
    # Example grid
    param_grid = {
        #"lr": [0.001, 0.01],
        "lr": [0.001],
        #"batch_size": [64, 128],
        "batch_size": [128],
        #"optimizer_type": ["adam", "sgd"]
        "optimizer_type": ["adam"]

    }
    # CPU
    device = torch.device("cpu")
    cpu_results = run_hyperparameter_grid(device, param_grid, epochs=1, patience=3, verbose=True)
    # GPU (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_results = run_hyperparameter_grid(device, param_grid, epochs=1, patience=3, verbose=True)
    else:
        print("CUDA is not available. Skipping GPU benchmark.")