import torch

# based on code from TA
def evaluate(model, test_loader, device: str):
    correct = 0
    total = 0
    # make sure to remove the training specific layers
    with torch.no_grad():
        for data in test_loader:
            # load up the data
            inputs, labels = data
            # make sure data is on the correct hardware
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            # run the input through the model
            outputs = model(inputs)
            # what is the prediction?
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            # was our prediction correct?
            correct += (predicted == labels).sum().item()
        # how many were correct? 
        accuracy = 100 * correct / total
        print('Accuracy: %d %%' % (100 * correct / total))
    return accuracy
