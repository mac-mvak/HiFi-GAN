import torch.nn as nn



class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, true_disc, generated_disc):
        loss = 0
        for true, gen in zip(true_disc, generated_disc):
            l1 = ((true-1)**2).mean()
            l2 = (gen**2).mean()
            loss = loss + l1 + l2
        return loss


