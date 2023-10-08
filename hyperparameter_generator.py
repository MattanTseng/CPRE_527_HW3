import yaml
import numpy as np

# this is a simple python script that creates a bunch of different config files.
epoch_array = np.arange(25, 26, 5)
learning_rate_array = np.linspace(0.001, 0.1, 5)
batch_size_array = np.arange(10, 30, 10)

validation_ratio = 0.1
train_ratio = 0.9
print("Epochs: ", epoch_array)
print("Learning rate: ", learning_rate_array)
print("Batch size: ", batch_size_array)

i = 1

# for epoch in epoch_array:
#     for lr in learning_rate_array:
#         for bs in batch_size_array:
#             temp_dict = {"epochs": int(epoch), 
#                          "learning_rate": float(lr), 
#                          "batch_size": int(bs), 
#                          "validation_ratio": validation_ratio, 
#                          "train_ratio": train_ratio}
#             filename = "hyperparameters" + str(i) + ".YAML"
#             with open(filename, 'w') as yaml_file:
#                 yaml.dump(temp_dict, yaml_file, default_flow_style=False)
#             i = 1+ i
