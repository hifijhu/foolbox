#!/usr/bin/env python3
"""
This test is used to test all attack apis from foolbox on the DCU and 
I really hope there would no dependency problem.
env: 
    docker 19.03.09:
        OS: Ubuntu 22.04
        DTK: 25.04.2
        Pytorch: 2.5.1
        Python: 3.10
        Foolbox: 3.3.4
        DCU: Z100L * 7
        Model: Resnet-18
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as atks
from foolbox.attacks import LinfPGD

import inspect

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

    # apply the attack
    attacks = [
        atks.L2ContrastReductionAttack,
        atks.VirtualAdversarialAttack,
        atks.DDNAttack,
        atks.L2ProjectedGradientDescentAttack,
        atks.LinfProjectedGradientDescentAttack,
        atks.L2BasicIterativeAttack,
        atks.LinfBasicIterativeAttack,
        atks.L2FastGradientAttack,
        atks.LinfFastGradientAttack,

        atks.L2AdditiveGaussianNoiseAttack,
        atks.L2AdditiveUniformNoiseAttack,
        atks.L2ClippingAwareAdditiveGaussianNoiseAttack,
        atks.L2ClippingAwareAdditiveUniformNoiseAttack,
        atks.LinfAdditiveUniformNoiseAttack,
        atks.L2RepeatedAdditiveGaussianNoiseAttack,
        atks.L2RepeatedAdditiveUniformNoiseAttack,
        atks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack,
        atks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack,
        atks.LinfRepeatedAdditiveUniformNoiseAttack,
        atks.InversionAttack,
        atks.BinarySearchContrastReductionAttack,
        atks.LinearSearchContrastReductionAttack,

        atks.HopSkipJumpAttack,

        atks.L2CarliniWagnerAttack,
        atks.NewtonFoolAttack,
        atks.EADAttack,
        atks.GaussianBlurAttack,
        atks.L2DeepFoolAttack,
        atks.LinfDeepFoolAttack,
        atks.SaltAndPepperNoiseAttack,
        atks.LinearSearchBlendedUniformNoiseAttack,
        atks.BinarizationRefinementAttack,
        atks.DatasetAttack,
        atks.BoundaryAttack,
        atks.L0BrendelBethgeAttack,
        atks.L1BrendelBethgeAttack,
        atks.L2BrendelBethgeAttack,
        atks.LinfinityBrendelBethgeAttack,
        atks.L0FMNAttack,
        atks.L1FMNAttack,
        atks.L2FMNAttack,
        atks.LInfFMNAttack,
        atks.PointwiseAttack,

        atks.FGM,
        atks.FGSM,
        atks.L2PGD,
        atks.LinfPGD,
        atks.PGD

    ]


    errorList = []
    for attack in attacks:
        print(f"current test api: {attack.__str__():s}")
        try:
                
            attack = LinfPGD()
            epsilons = 0.1
            raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

            # calculate and report the robust accuracy (the accuracy of the model when
            # it is attacked)
            robust_accuracy = 1 - success.float32().mean(axis=-1)
            print("robust accuracy for perturbations with")
            
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

            # we can also manually check this
            # we will use the clipped advs instead of the raw advs, otherwise
            # we would need to check if the perturbation sizes are actually
            # within the specified epsilon bound
            print("manually chech the robust accuracy for perturbations with")

            acc2 = accuracy(fmodel, advs_, labels)
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            print("    perturbation sizes:")
            perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
            print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
            if acc2 == 0:
                break
            print("OK")
        except Exception as e:
            print(f"An error Occured:{e} while testing api {attack.__str__()}")
            errorList.append(attack)
            continue


    if len(errorList) == 0:
        print("Congratulation! All api had passed the test!")
    else:
        print("Sorry to tell you that the following api hasn't pass the test:")
        for item in errorList:
            print(item__str__())
    
if __name__ == "__main__":
    main()
    