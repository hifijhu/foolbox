#!/usr/bin/env python3
"""
This test is an example about how to use the attack who need a distance. It is useful for 
LinearSearchBlendedUniformNoiseAttack, GaussianBlurAttack, BinarySearchContrastReductionAttack 
and InversionAttack. 
env: 
    docker 19.03.09:
        OS: Ubuntu 22.04
        DTK: 25.04.2
        Pytorch: 2.5.1
        Python: 3.10
        Foolbox: 3.3.4
        DCU: Z100L * 7
        Model: Resnet-18
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as atks
from foolbox.attacks import LinfPGD
from foolbox.distances import LpDistance

def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    distance = LpDistance(1)
    # apply the attack
    attack = atks.BinarySearchContrastReductionAttack(distance=distance)
    print(f"current test api: {attack.__class__.__name__}")
    try:
        epsilons = [0.1]
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

        # calculate and report the robust accuracy (the accuracy of the model when
        # it is attacked)
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        print("robust accuracy for perturbations with")
        
        print(f"  Linf norm ≤ {epsilons[0]:<6}: {robust_accuracy.item() * 100:4.1f} %")

    except Exception as e:
        print(f"An error Occured:{e} while testing api {attack.__class__.__name__}")
    
if __name__ == "__main__":
    main()

    