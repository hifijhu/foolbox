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
    attacks_two =[

    ]


    l1 = []
    for attack in attacks:
        try:
            atk = attack(steps=10)
            
            if hasattr(atk, "steps"):
                l1.append(attack)
        except Exception as e:
            continue
    for item in l1:
        print(f"'{item.__name__}',")

    
if __name__ == "__main__":
    main()

    