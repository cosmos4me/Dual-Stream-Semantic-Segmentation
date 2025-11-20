
class DiceLoss(nn.Module): 
    def __init__(self, smooth=1e-6, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, labels_float, labels_orig):
        probs = torch.sigmoid(logits)
        if self.ignore_index is not None:
            mask = (labels_orig != self.ignore_index).unsqueeze(1) # (B, 1, H, W)
            probs = probs * mask
            labels_float = labels_float * mask
        intersection = (probs * labels_float).sum(dim=(1, 2, 3))
        total = (probs.sum(dim=(1, 2, 3)) + labels_float.sum(dim=(1, 2, 3)))
        dice_score = (2. * intersection + self.smooth) / (total + self.smooth)
        return (1.0 - dice_score).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels_float, labels_orig):
        bce_val = self.bce_loss(logits, labels_float)
        probs = torch.sigmoid(logits)
        p_t = probs * labels_float + (1 - probs) * (1 - labels_float)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        alpha_factor = self.alpha * labels_float + (1 - self.alpha) * (1 - labels_float)
        focal_loss_val = alpha_factor * modulating_factor * bce_val
        if self.ignore_index is not None:
            mask = (labels_orig != self.ignore_index).unsqueeze(1)
            focal_loss_val = focal_loss_val * mask
            if mask.sum() > 0:
                return focal_loss_val.sum() / mask.sum()
            else:
                return focal_loss_val.sum()
        else:
            return focal_loss_val.mean()

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - torch.cumsum(gt_sorted, 0)
    union = gts + torch.cumsum(1.0 - gt_sorted, 0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(LovaszLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        B, C, H, W = probs.shape

        probs = probs.view(B, -1)
        labels = labels.view(B, -1)

        losses = torch.zeros(B, dtype=torch.float32, device=device)

        for i in range(B):
            prob = probs[i]
            label = labels[i]
            if self.ignore_index is not None:
                mask = (label != self.ignore_index)
                if not mask.any(): continue
                prob = prob[mask]
                label = label[mask]
            if len(prob) == 0: continue
            errors = (label - prob).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            gt_sorted = label[perm]
            grad = lovasz_grad(gt_sorted)
            losses[i] = torch.dot(errors_sorted, grad)

        return losses.mean()
