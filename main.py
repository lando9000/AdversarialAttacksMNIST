import torch
from src import FGSMAttack, ResNet18, LeNet
from utils import visualization


# Main function for running a FGSM adversarial attack against LeNet and ResNet18 trained on the MNIST dataset.
def __main__():
    # Define Epsilon Values for image perturbation
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    # Define the device where using
    device = torch.device("cpu")

    # Load instance of the pretrained LeNet neuronal network from file
    lenetModel = LeNet.LeNet().to(device)
    lenetModel.load_state_dict(torch.load("networks/lenet_mnist.pth", map_location='cpu'))

    # Load instance of the pretrained ResNet neuronal network from file
    resnet18Model = ResNet18.ResNet18().load_from_checkpoint("networks/resnet18_mnist.pt")

    # Instance of attack against LeNet
    FGSMAttackerLeNet = FGSMAttack.FGSMAttacker("LeNet", lenetModel, device, epsilons)
    # Run the attack
    accLeNet, lossLeNet, advExLN = FGSMAttackerLeNet.run()

    # Instance of attack against ResNet18
    FGSMAttackerResNet = FGSMAttack.FGSMAttacker("ResNet", resnet18Model, device, epsilons)
    # Run the attack
    accResNet, lossResNet, advExRN = FGSMAttackerResNet.run()

    # Plot results
    visualization.plotAccuracy(epsilons, [accLeNet, accResNet])
    visualization.plotAdversarialExamples(epsilons, advExLN)
    visualization.plotAdversarialExamples(epsilons, advExRN)


if __name__ == '__main__':
    __main__()
