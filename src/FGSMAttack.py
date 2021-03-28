import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from time import sleep


class FGSMAttacker():
    def __init__(self, modelName, model, device, epsilons):
        super(FGSMAttacker, self).__init__()
        self.modelName = modelName
        self.device = device
        self.model = model
        self.epsilons = epsilons
        self.adversarialExamples = []

    def __del__(self):
        print("Destructur called.")

    def test(self):
        print("Starting attack on", self.modelName, "...")
        # Accuracy counter
        adv_examples = []
        accuracies = []
        loss = []

        for epsilon in self.epsilons:
            correct = 0
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
                l = F.nll_loss(probabilities, target)

                # Zero all existing gradients
                self.model.zero_grad()

                # Calculate gradients of model in backward pass
                l.backward(retain_graph=True)
                l += l

                # Call FGSM Attack
                dataGrad = data.grad
                perturbedData = self.attack(data, dataGrad, epsilon)

                # Re-classify the perturbed image
                probabilities, finalPred = self.predict(perturbedData)

                for i in range(len(data)):
                    # Check for success
                    if finalPred[i].item() == target[i].item():
                        correct += 1
                        # Special case for saving 0 epsilon examples
                        if (epsilon == 0) and (len(adv_examples) < 5):
                            adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                            adv_examples.append((prediction[i].item(), finalPred[i].item(), adv_ex))
                    else:
                        # Save some adv examples for visualization later
                        if len(adv_examples) < 5:
                            adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                            adv_examples.append((prediction[i].item(), finalPred[i].item(), adv_ex))

            # Calculate final accuracy for this epsilon
            accuracy = correct / 10000
            accuracies.append(accuracy)
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                                     10000,
                                                                     accuracy))
            sleep(0.1)
            loss.append(l)
        return accuracies

    def attack(self, data, dataGrad, epsilon):
        # Apply FGSM attack
        dataGrad = dataGrad.sign()
        perturbedImg = data + epsilon * dataGrad
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
        accuracies = self.test()
        return accuracies
