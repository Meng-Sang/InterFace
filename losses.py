import torch
import math

from torch import distributed


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class InterFace(torch.nn.Module):

    def __init__(self, s=64.0, margin=0.5, mid=0.2, alpha=0.1, is_dyn=False):
        super(InterFace, self).__init__()
        self.is_dyn = is_dyn
        self.scale = s
        self.m = margin
        self.alpha = alpha
        self.mid = mid
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()
        # self.alpha = 0.15

    def get_target_logit(self, local_index, local_target_logit):
        _indexs = [
            torch.zeros_like(local_index)
            for _ in range(self.world_size)
        ]
        _target_logits = [
            torch.zeros_like(local_target_logit)
            for _ in range(self.world_size)
        ]
        distributed.all_gather(_indexs, local_index)
        distributed.all_gather(_target_logits, local_target_logit)
        target_logits = torch.zeros_like(local_index)
        target_logits = target_logits.type_as(local_target_logit)
        for i, item in enumerate(_indexs):
            _index = torch.where(item != -1)[0]
            target_logits[_index, 0] = _target_logits[i][_index, item[_index].view(-1)]
        return target_logits

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, center_logits: torch.Tensor):
        '''
            数据准备
        '''
        target_logits = self.get_target_logit(labels, logits)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        logits.acos_()
        target_logit.acos_()
        # indexs, target_logits = self.get_target_logit(index, target_logit)
        cln_target_logits = target_logits.acos().clone().detach().reshape(-1, 1)
        cln_center_logits = center_logits.acos().clone().detach()

        target_logit.add_(self.m)
        # target_logit.clamp_(0, math.pi)
        target_logit.cos_()

        cln_center_logits.clamp_(0.01, math.pi)  # 防止除后溢出现象
        m_hat = cln_target_logits / cln_center_logits * -1
        # todo
        if self.is_dyn:
            mid = self.mid / cln_center_logits
            m_hat = pow(math.e, mid * -1) - m_hat.exp()
        else:
            m_hat = pow(math.e, self.mid * -1) - m_hat.exp()
        m_hat = m_hat * self.alpha
        # print(self.plus)
        logits.sub_(m_hat)
        # logits.clamp_(0, math.pi)
        logits.cos_()
        logits[index, labels[index].view(-1)] = target_logit
        logits.mul_(self.scale)
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
