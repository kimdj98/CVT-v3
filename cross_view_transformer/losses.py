import torch
import logging

from fvcore.nn import sigmoid_focal_loss


logger = logging.getLogger(__name__)

def velocityLoss(pred, label, gamma=None, reduction='mean'):
    """
    End_point error loss
    """
    assert pred.shape == label.shape

    error = torch.abs(pred - label)

    if gamma != None:
        mask = (label != 0)
        loss = (error**2)*mask*gamma + (error**2)*(~mask)*(1-gamma)
        
    else:
        loss = error**2

    return loss.mean() if reduction == 'mean' else loss.sum()

class VelocityLoss(torch.nn.Module):
    def __init__(
        self,
        gamma=0.95,
        reduction='mean' # 'mean' or 'sum'
    ):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred_x = pred['x']
            pred_y = pred['y']
            pred = torch.cat([pred_x, pred_y], dim=1)

        batch = batch["velocity_map"]

        return velocityLoss(pred, batch, self.gamma, self.reduction)

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)


class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        label = batch['center']
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        modules = dict()
        weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                weights[key.replace('_weight', '')] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

                # Assign weight to 1.0 if not explicitly set.
                if key not in weights:
                    logger.warn(f'Weight for {key} was not specified.')
                    weights[key] = 1.0

        assert modules.keys() == weights.keys()

        super().__init__(modules)

        self._weights = weights

    def forward(self, pred, batch):
        cur_visibility_loss = self['cur_visible'](pred["occ_curr"]["bev"], batch)         # BinarySegmentationLoss
        prev_visibility_loss = self['prev_visible'](pred["occ_prev"]["bev"], batch)       # BinarySegmentationLoss
        cur_center_loss = self['cur_center'](pred["occ_curr"]["center"], batch)              # CenterLoss
        prev_center_loss = self['prev_center'](pred["occ_prev"]["center"], batch)            # CenterLoss
        vel_loss = self['velocity'](pred["vel"], batch)                                                  # VelocityLoss

        outputs = dict()
        outputs.update(cur_visible=cur_visibility_loss)
        outputs.update(prev_visible=prev_visibility_loss)
        outputs.update(cur_center=cur_center_loss)
        outputs.update(prev_center=prev_center_loss)
        outputs.update(velocity=vel_loss)

        total = sum(self._weights[k] * o for k, o in outputs.items())

        return total, outputs
