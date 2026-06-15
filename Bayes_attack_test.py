from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "foolbox"))

import eagerpy as ep
import torchvision.models as models

from foolbox import PyTorchModel, accuracy, samples

from Bayes_attack import BayesianConfidenceAttack


def main() -> None:
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    print("note: plain ResNet-18 in eval mode is deterministic, so MC sampling repeats identical logits.")

    attack = BayesianConfidenceAttack(num_samples=16, lambda_param=0.5, steps=20)
    epsilons = [
        0.0,
        0.0001,
        0.0002,
        0.0005,
        0.0010,
        0.0015,
        0.0020,
        0.0050,
        0.0200,
        0.0500,
        0.1000,
        0.2000,
        0.5,
        0.8,
        1
    ]

    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    assert len(raw_advs) == len(epsilons)
    assert len(clipped_advs) == len(epsilons)
    assert success.shape == (len(epsilons), len(images))

    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm <= {eps:<6}: {acc.item() * 100:4.1f} %")

    print()
    print("manual check using clipped adversarials:")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print(f"  Linf norm <= {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()
