import eagerpy as ep
import torch
from foolbox.attacks.base import FixedEpsilonAttack
from foolbox.models import PyTorchModel

class BayesianConfidenceAttack(FixedEpsilonAttack):
    """
    贝叶斯不确定性攻击：通过 MC Dropout 采样，攻击模型的预测确定性。
    """
    def __init__(self, num_samples=10, lambda_param=0.5, steps=20):
        # num_samples: MC Dropout 的采样次数
        # lambda_param: 不确定性损失的权重
        self.num_samples = num_samples
        self.lambda_param = lambda_param
        self.steps = steps

    def __call__(self, model, inputs, labels, epsilon):
        # 将输入转为 EagerPy 张量以保持框架兼容
        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        
        # 初始扰动
        adv_x = x.clone()
        step_size = epsilon / self.steps

        for _ in range(self.steps):
            # 定义损失函数并计算梯度
            def loss_fn(perturbed_inputs):
                logits_list = []
                # 执行 MC Dropout 采样
                for _ in range(self.num_samples):
                    logits_list.append(model(perturbed_inputs))
                
                # 计算平均概率分布
                probs = [l.softmax(axis=-1) for l in logits_list]
                avg_probs = ep.stack(probs).mean(axis=0)
                
                # 1. 基础对抗损失 (分类错误)
                l_adv = -ep.cross_entropy(avg_probs, y).mean()
                
                # 2. 不确定性损失 (预测熵：越小代表越自信)
                # Entropy = -sum(p * log(p))
                entropy = -(avg_probs * ep.log(avg_probs + 1e-10)).sum(axis=-1).mean()
                
                return l_adv + self.lambda_param * entropy

            # 获取梯度
            _, gradients = ep.value_and_grad(loss_fn, adv_x)
            
            # 沿着梯度上升方向更新 (PGD 思想)
            adv_x = adv_x + step_size * gradients.sign()
            
            # 投影回 epsilon 球面和原始像素范围 [0, 1]
            adv_x = ep.clip(adv_x, x - epsilon, x + epsilon)
            adv_x = ep.clip(adv_x, 0, 1)

        return adv_x.raw # 返回原始框架（如 PyTorch）的张量