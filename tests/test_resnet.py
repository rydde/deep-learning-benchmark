import torch
from src.resnet_module import SimpleResNet
# Test for the SimpleResNet model
def test_model_initialization():
    model = SimpleResNet(num_classes=10)
    assert model is not None
    assert isinstance(model, SimpleResNet)
        #print result of assert
    print("Model initialized successfully.")

# Test for model parameters
def test_model_parameters():
    model = SimpleResNet(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0  # Ensure the model has trainable parameters
    #print result of assert
    print(f"Number of trainable parameters: {num_params}")

# Test for model device
def test_model_device():
    model = SimpleResNet(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    y = model(x)
    assert y.device.type == device.type  # Ensure the output is on the correct device
        #print result of assert
    print(f"Model device: {y.device.type}")

# Test for training and evaluation modes
def test_model_training_mode():
    model = SimpleResNet(num_classes=10)
    model.train()  # Set the model to training mode
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 10)  # Output shape should match the number of classes
        #print result of assert
    print("Model is in training mode and output shape is correct.")

def test_model_evaluation_mode():
    model = SimpleResNet(num_classes=10)
    model.eval()  # Set the model to evaluation mode
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 10)  # Output shape should match the number of classes
        #print result of assert
    print("Model is in evaluation mode and output shape is correct.")


#Test for model save and load
def test_model_save_load():
    model = SimpleResNet(num_classes=10)
    torch.save(model.state_dict(), 'test_model.pth')
    loaded_model = SimpleResNet(num_classes=10)
    loaded_model.load_state_dict(torch.load('test_model.pth'))
    x = torch.randn(1, 3, 224, 224)
    original_output = model(x)
    loaded_output = loaded_model(x)
    assert torch.allclose(original_output, loaded_output, atol=1e-6)  # Ensure outputs are similar
        #print result of assert
    print("Model saved and loaded successfully. Outputs are similar.")

# Test for batch size and input shape
def test_model_batch_size():
    model = SimpleResNet(num_classes=10)
    x = torch.randn(8, 3, 224, 224)  # Batch size of 8
    y = model(x)
    assert y.shape == (8, 10)  # Output shape should match batch size and number of classes
        #print result of assert
    print("Model batch size test passed. Output shape is correct.")

def test_model_input_shape():
    model = SimpleResNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224)  # Input shape should be (batch_size, channels, height, width)
    y = model(x)
    assert y.shape == (1, 10)  # Output shape should match the number of classes
        #print result of assert
    print("Model input shape test passed. Output shape is correct.")

#Test for model gradient computation and zeroing
def test_model_gradient():
    model = SimpleResNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    y = model(x)
    loss = y.sum()  # Simple loss for testing
    loss.backward()  # Check if gradients can be computed
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None  # Ensure gradients are computed
            assert param.grad.shape == param.shape  # Gradient shape should match parameter shape
        #print result of assert
    print("Model gradients computed successfully. All gradients are of correct shape.")

def test_model_zero_grad():
    model = SimpleResNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    y = model(x)
    loss = y.sum()  # Simple loss for testing
    loss.backward()  # Compute gradients
    model.zero_grad()  # Zero the gradients
    for param in model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)  # Ensure gradients are zeroed
        #print result of assert
    print("Model gradients zeroed successfully. All gradients are zero.")
#Test for model forward pass with different input sizes and batch sizes
def test_model_forward_with_different_input_sizes():
    model = SimpleResNet(num_classes=10)
    input_sizes = [(1, 3, 224, 224), (1, 3, 128, 128), (1, 3, 64, 64)]
    for size in input_sizes:
        x = torch.randn(size)
        y = model(x)
        assert y.shape == (size[0], 10)  # Output shape should match batch size and number of classes
        #print result of assert
        print(f"Model forward pass with input size {size} passed. Output shape is correct.")


def test_model_forward_with_different_batch_sizes():
    model = SimpleResNet(num_classes=10)
    batch_sizes = [1, 4, 8, 16]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 224, 224)
        y = model(x)
        assert y.shape == (batch_size, 10)  # Output shape should match batch size and number of classes
        #print result of assert
        print(f"Model forward pass with batch size {batch_size} passed. Output shape is correct.")  

#Test for model forward pass with different heights and widths
def test_model_forward_with_different_heights_and_widths():
    model = SimpleResNet(num_classes=10)
    sizes = [(1, 3, 224, 224), (1, 3, 128, 128), (1, 3, 64, 64)]
    for size in sizes:
        x = torch.randn(size)
        y = model(x)
        assert y.shape == (size[0], 10)  # Output shape should match batch size and number of classes
        #print result of assert
        print(f"Model forward pass with input size {size} passed. Output shape is correct.")

