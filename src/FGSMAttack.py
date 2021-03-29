import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.autonotebook import tqdm
from time import sleep


class FGSMAttacker:
    def __init__(self, modelName, model, device, epsilons):
        super(FGSMAttacker, self).__init__()
        self.modelName = modelName
        self.device = device
        self.model = model
        self.epsilons = epsilons

    def __del__(self):
        print("Destructur called.")

    def test(self):
        print("Starting attack on", self.modelName, "...")
        # Accuracy counter
        allAdversarialExamples = []
        allAccuracies = []

        for epsilon in self.epsilons:
            correct = 0
            loss = 0
            currentExamples = []
            # Loop over all examples in test set
            for batch in tqdm(self.model.testDL):

                # Get image data and true classification from batch
                data, target = batch

                # Send the data and label to the device
                data, target = data.to(self.device), target.to(self.device)

                # Set requires_grad attribute of tensor. Important for Attack
                data.requires_grad = True

                # Forward pass the data through the model
                probabilities, prediction = self.predict(data)

                # Calculate the loss
                currentLoss = F.nll_loss(probabilities, target)

                # Zero all existing gradients
                self.model.zero_grad()

                # Calculate gradients of model in backward pass
                currentLoss.backward(retain_graph=True)

                # Call FGSM Attack
                dataGrad = data.grad
                perturbedData = self.attack(data, dataGrad, epsilon)

                # Re-classify the perturbed image
                probabilities, finalPred = self.predict(perturbedData)

                for i in range(len(data)):

                    # If initial prediction was incorrect, skip image
                    if prediction[i].item() != target[i].item():
                        continue

                    # Save adversarial example
                    adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                    npProb = probabilities.detach().numpy()

                    # Get one image from batch
                    if self.model.batchSize == 64:
                        adv_ex = adv_ex[i]
                        npProb = npProb[i]
                    # Check for success
                    if finalPred[i].item() == target[i].item():
                        correct += 1
                        # Special case for saving 0 epsilon examples
                        if (epsilon == 0) and (len(currentExamples) < 5):
                            currentExamples.append((prediction[i].item(), finalPred[i].item(), adv_ex, npProb))
                    else:
                        # Save some adv examples for visualization later
                        if len(currentExamples) < 5:
                            currentExamples.append((prediction[i].item(), finalPred[i].item(), adv_ex, npProb))

            # Calculate final accuracy for this epsilon
            accuracy = correct / 10000
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                                     10000,
                                                                     accuracy))
            # Sleep for console output
            sleep(0.1)
            # Append results from current epsilon to output
            allAccuracies.append(accuracy)
            allAdversarialExamples.append(currentExamples)
        return allAccuracies, allAdversarialExamples

    def attack(self, data, dataGrad, epsilon):
        # Get the sign of data gradient
        dataGradSign = dataGrad.sign()
        # Apply FGSM attack
        perturbedImg = data + epsilon * dataGradSign
        # Adding clipping to maintain [0,1] range
        perturbedImg = torch.clamp(perturbedImg, 0, 1)
        return perturbedImg

    def predict(self, data):
        output = self.model(data)
        prediction = None
        if self.modelName == "LeNet":
            probs = output
            prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        elif self.modelName == "ResNet":
            m = nn.Softmax(1)
            probs = m(output)
            prediction = torch.argmax(probs, dim=1)
        return probs, prediction

    def run(self):
        accuracies,  advEx = self.test()
        return accuracies,  advEx

    def createAdversarialExamples(self):
        loss = 0
        examples = []
        predict = []
        orig = []
        # Only get one batch

        for batch in tqdm(self.model.testDL):

            # Get image data and true classification from batch
            data, target = batch

            # Send the data and label to the device
            data, target = data.to(self.device), target.to(self.device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            probabilities, prediction = self.predict(data)

            for eps in self.epsilons:
                # Calculate the loss
                currentLoss = F.nll_loss(probabilities, target)

                # Zero all existing gradients
                self.model.zero_grad()

                # Calculate gradients of model in backward pass
                currentLoss.backward(retain_graph=True)
                loss += currentLoss.detach().item()

                # Call FGSM Attack
                dataGrad = data.grad
                perturbedData = self.attack(data, dataGrad, eps)

                # Re-classify the perturbed image
                probabilities, finalPred = self.predict(perturbedData)
                adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                npProb = probabilities.detach().numpy()
                if self.model.batchSize == 64:
                    adv_ex = adv_ex[0]
                    npProb = npProb[0]
                if prediction[0].item() == target[0].item():
                    examples.append(adv_ex)
                    predict.append(finalPred[0].item())
                    orig.append(target[0].item())

            break
        return examples, predict, orig
