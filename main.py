import torch
from src import FGSMAttack, ResNet18, LeNet
from utils import visualization

def __main__():
    # Define Epsilon Values for image perturbation
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    # Define the device where using for training
    device = torch.device("cpu")

    # Instance of the pretrained LeNet neuronal network
    lenetModel = LeNet.LeNet().to(device)
    lenetModel.load_state_dict(torch.load("networks/lenet_mnist.pth", map_location='cpu'))

    # Instance of the pretrained ResNet neuronal network
    resnet18Model = ResNet18.ResNet18().load_from_checkpoint("networks/resnet18_mnist.pt")

    # Instance of attack against the neuronal network
    FGSMAttackerLeNet = FGSMAttack.FGSMAttacker("LeNet", lenetModel, device, epsilons)
    accLeNet = FGSMAttackerLeNet.run()

    FGSMAttackerResNet = FGSMAttack.FGSMAttacker("ResNet", resnet18Model, device, epsilons)
    accResNet = FGSMAttackerResNet.run()

    visualization.printAccuracy(epsilons, [accLeNet, accResNet],
                                ["LeNet", "ResNet"])
    #visualization.printLoss(epsilons, [lossLeNet, lossResNet],
    #                        ["LeNet", "ResNet"])


if __name__ == '__main__':
    __main__()
