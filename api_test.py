#!/usr/bin/env python3
"""
This test is used to test all attack apis from foolbox on the DCU and 
I really hope there would no dependency problem. :(
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
        atks.VirtualAdversarialAttack, # failed
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
        atks.InversionAttack, # failed
        atks.BinarySearchContrastReductionAttack, # failed
        atks.LinearSearchContrastReductionAttack,

        # atks.HopSkipJumpAttack, OOM

        # atks.L2CarliniWagnerAttack, timeout
        atks.NewtonFoolAttack,
        # atks.EADAttack, timeout
        atks.GaussianBlurAttack, # failed
        atks.L2DeepFoolAttack,
        atks.LinfDeepFoolAttack,
        atks.SaltAndPepperNoiseAttack,
        atks.LinearSearchBlendedUniformNoiseAttack, # failed
        atks.BinarizationRefinementAttack, # failed
        atks.DatasetAttack,  # failed
        atks.BoundaryAttack,
        # atks.L0BrendelBethgeAttack, timeout and DCU dont work
        # atks.L1BrendelBethgeAttack, userwarning and block the process
        


    ]
    attacks_two =[
        atks.L2BrendelBethgeAttack,
        # atks.LinfinityBrendelBethgeAttack, timeout and DCU dont work
        atks.L0FMNAttack, # failed
        atks.L1FMNAttack,
        atks.L2FMNAttack,
        atks.LInfFMNAttack,
        atks.PointwiseAttack, # failed

        atks.FGM,
        atks.FGSM,
        atks.L2PGD,
        atks.LinfPGD, 
        atks.PGD
    ]


    errorList = []
    for attack in attacks:
        print(f"current test api: {attack.__name__}")
        try:
            epsilons = [0.1]
            raw_advs, clipped_advs, success = attack()(fmodel, images, labels, epsilons=epsilons)

            # calculate and report the robust accuracy (the accuracy of the model when
            # it is attacked)
            robust_accuracy = 1 - success.float32().mean(axis=-1)
            print("robust accuracy for perturbations with")
            
            print(f"  Linf norm ≤ {epsilons[0]:<6}: {robust_accuracy.item() * 100:4.1f} %")

        except Exception as e:
            print(f"An error Occured:{e} while testing api {attack.__name__}")
            errorList.append(attack)
            continue


    if len(errorList) == 0:
        print("Congratulation! All api had passed the test!")
    else:
        print(f"Not passed/Amount: {len(errorList)}/{len(attacks)}")
        print("Sorry to tell you that the following api hasn't pass the test:")
        for item in errorList:
            print(item.__name__)
    
if __name__ == "__main__":
    main()

    