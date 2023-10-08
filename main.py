from src import load_cifar10, Net, training_step, evaluate, config_loader
import numpy as np
import time
import torch
import os
import sys
from src.cnn import Net_500k
from src.analytics import My_Analytics

# this is largely based on the code from the TA
if __name__ == '__main__':

    # check to see what hardware we can run this on.
    if torch.backends.mps.is_available():
        # this is for apple sillicon
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # this is for CUDA gpus
        device = "cuda"
    else:
        # and if none of the above, then default to using the cpu
        device = "cpu"

    # I want to save the analytics to an adjacent location.
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # use runtime command to differentiate between different config files
    config_file_name = "hyperparameters" + sys.argv[1] + ".YAML"

    # based on where this file is.... locate the hyper parameter file. 
    config_location = os.path.join(current_dir, config_file_name)

    # read the config file to get the hyper paraemters
    hyper_params = config_loader(config_location)

    # define some variables based on the config file
    n_epochs = hyper_params["epochs"]
    learning_rate = hyper_params["learning_rate"]
    

    print("Hyperparameters: \n", hyper_params)

    # instantiate some empty arrays to hold some stats
    mean_train_losses = np.empty(0)
    validation_accuracies = np.empty(0)

    # which model do you want to run? 
    model = Net()
    # model = Net_500k()

    # create a directory to hold the results
    dir_name = "E_" + str(n_epochs) + "_lr_" + str(learning_rate) + "_BS_" + str(hyper_params["batch_size"]) + "_" + model.__str__() 
    output_location = os.path.join(current_dir, dir_name)

    # instantiate the analytics generator
    analytics = My_Analytics(output_location)

    # make sure teh model is using the appropriate hardware
    model.to(device)

    # load up the data (based on TAs code)
    print("loading data: ")
    train_loader, test_loader,val_loader, classes = load_cifar10(hyper_params)


    # here's were we actually do the training
    print("Starting training: ")
    start_time = time.time()
    for epoch in range(n_epochs):
        #train
        mean_train_losses = np.concatenate((mean_train_losses, training_step(model, train_loader, epoch, device, learning_rate)))
        # run validation set
        validation_accuracies = np.concatenate((validation_accuracies, np.array([evaluate(model, val_loader, device)])))
        print("-"*10,"Training finshed","-"*10)

    # stop timeer    
    end_time = time.time()
    print("Done training")
    run_time = end_time - start_time

    test_accuracy = evaluate(model, test_loader, device)

    # create the graphs
    analytics.create_1D_graphs(mean_train_losses, "image*10^4", "loss", "Training Losses")
    analytics.create_1D_graphs(validation_accuracies, "image*10^4", "accuracy", "Validation Accuracies")
    analytics.create_report(run_time, hyper_params, test_accuracy, device)



    # print out results to console
    print("Here are the average losses for every epoch: ", mean_train_losses)
    print("Here are the validation_accuracies of every epoch: ", validation_accuracies)
    print("Here are is the accuracy on the test set: ", test_accuracy)
    print("The run took: ", run_time, " seconds to run")
