import numpy as np
from termcolor import colored


def verify_part_a():
    """
    Verifying results of Part a.
    :return: Nothing
    """

    # Get the predicted and actual weights, output files.
    actual_output = np.loadtxt("verify/model_outputfile_a.txt")
    actual_weights = np.loadtxt("verify/model_weightfile_a.txt")
    predicted_output = np.loadtxt("Results/Part_a/predictions.txt")
    predicted_weights = np.loadtxt("Results/Part_a/weights.txt")

    if (actual_output.shape[0] != predicted_output.shape[0]):
        print(colored("Prediction file of wrong dimensions for part a", "red"))
        exit()
    if (actual_weights.shape[0] != predicted_weights.shape[0]):
        print(colored("Weight file of wrong dimensions for part a", "red"))
        exit()

    pred_error = np.sum(np.square(predicted_output - actual_output)) / np.sum(
        np.square(actual_output))  # Error in output
    weight_error = np.sum(np.square(predicted_weights - actual_weights)) / np.sum(
        np.square(actual_weights))  # Error in weights

    print(colored("Error in Predictions for part a : " + str(pred_error), "green"))
    print(colored("Error in Weights for part a : " + str(weight_error), "green"))
