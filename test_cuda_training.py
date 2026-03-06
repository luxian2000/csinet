import time
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print('CUDA device name:', torch.cuda.get_device_name(0))

    # Tiny model
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    # Verify parameters on device
    on_cuda = any(p.device.type == 'cuda' for p in model.parameters())
    print('Model parameters on CUDA:', on_cuda)

    # Random dataset
    N = 1024
    batch_size = 128
    x = torch.randn(N, 100)
    y = torch.randint(0, 10, (N,))

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)

    epochs = 3
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = x[idx].to(device)
            yb = y[idx].to(device)

            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= N
        print(f'Epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f}')

    used = time.time() - start
    print(f'Training finished in {used:.2f}s')

    # final sanity checks
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('torch.cuda.device_count():', torch.cuda.device_count())


if __name__ == '__main__':
    main()
