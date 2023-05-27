# Date: 2023-5-19
# Author: Generated by GoCodeo.


import unittest
import torch
from losses import sigmoid_focal_loss, SigmoidFocalLoss

class TestSigmoidFocalLoss(unittest.TestCase):

    def test_positive(self):
        pred = torch.tensor([0.9, 0.2, 0.4])
        label = torch.tensor([1, 0, 1])
        loss_fn = SigmoidFocalLoss()
        loss = loss_fn(pred, label)
        self.assertAlmostEqual(loss.item(), 0.236, places=3)

    def test_negative(self):
        pred = torch.tensor([0.1, 0.8, 0.6])
        label = torch.tensor([1, 0, 1])
        loss_fn = SigmoidFocalLoss()
        loss = loss_fn(pred, label)
        self.assertAlmostEqual(loss.item(), 1.473, places=3)

    def test_error(self):
        pred = torch.tensor([0.9, 0.2, 0.4])
        label = torch.tensor([1, 0, 1])
        loss_fn = SigmoidFocalLoss(alpha=0.5, gamma=1.0, reduction='sum')
        loss = loss_fn(pred, label)
        self.assertAlmostEqual(loss.item(), 0.75, places=3)

    def test_edge(self):
        pred = torch.tensor([0.0, 1.0])
        label = torch.tensor([1, 0])
        loss_fn = SigmoidFocalLoss()
        loss = loss_fn(pred, label)
        self.assertAlmostEqual(loss.item(), 0.693, places=3)

if __name__ == '__main__':
    unittest.main()