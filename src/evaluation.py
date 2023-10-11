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


            outputs = model(inputs.to('cuda:0'))
            labels = labels.to(outputs.device)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            # was our prediction correct?
            correct += (predicted == labels).sum().item()
        # how many were correct? 
        accuracy = 100 * correct / total
        print('Accuracy: %d %%' % (100 * correct / total))
    return accuracy
