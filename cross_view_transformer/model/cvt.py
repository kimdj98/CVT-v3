import torch
import torch.nn as nn
from einops import rearrange

class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        occ_decoder,
        vel_decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        occ_dim = 2 # visibility and center
        vel_dim = 2 # x and y velocity

        self.encoder = encoder
        self.occ_decoder = occ_decoder
        self.vel_decoder = vel_decoder
        self.outputs = outputs

        self.occ_to_logits = nn.Sequential(
            nn.Conv2d(self.occ_decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, occ_dim, 1))
        
        # TODO: you may need to modifiy dimensions
        self.vel_to_logits = nn.Sequential(
            nn.Conv2d(self.vel_decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, vel_dim, 1))

    def forward(self, batch):
        occ_prev, occ_curr, vel = self.encoder(batch)

        occ_prev = self.occ_decoder(occ_prev)
        occ_curr = self.occ_decoder(occ_curr)
        vel = self.vel_decoder(vel)

        occ_prev = self.occ_to_logits(occ_prev)
        occ_curr = self.occ_to_logits(occ_curr)
        vel = self.vel_to_logits(vel)

        occ_prev = {"bev": occ_prev[:,0,:,:].unsqueeze(1),
                    "center": occ_prev[:,1,:,:].unsqueeze(1)}
        
        occ_curr = {"bev": occ_curr[:,0,:,:].unsqueeze(1),
                    "center": occ_curr[:,1,:,:].unsqueeze(1)}
        
        vel = {"x": vel[:,0,:,:].unsqueeze(1),
               "y": vel[:,1,:,:].unsqueeze(1)}

        output = dict()
        output.update(occ_prev=occ_prev, occ_curr=occ_curr, vel=vel)

        return output