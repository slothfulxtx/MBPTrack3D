import torch

from .backbone import DGCNN
from .transformer import Transformer
from models.base_model import BaseModel
from .rpn import RPN

class MBPTrack(BaseModel):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = Transformer(cfg.transformer_cfg)
        self.loc_net = RPN(cfg.rpn_cfg)

    def forward_embed(self, input):
        pcds = input['pcds']
        batch_size, duration, npts, _ = pcds.shape
        
        pcds = pcds.view(batch_size*duration, npts, -1)
        b_output = self.backbone_net(pcds)
        xyz = b_output['xyz']
        feat = b_output['feat']
        idx = b_output['idx']
        assert len(idx.shape) == 2
        return dict(
            xyzs=xyz.view(batch_size, duration, xyz.shape[1], xyz.shape[2]),
            feats=feat.view(batch_size, duration, feat.shape[1], feat.shape[2]),
            idxs=idx.view(batch_size, duration, idx.shape[1])
        )
 
    def forward_update(self, input):
        memory = input.pop('memory', None)
        layer_feats = input['layer_feats'] # nl, b, c, n
        xyz = input['xyz'] # b, n, 3
        mask = input['mask'] # b, n
        new_memory = dict()
        new_memory['feat'] = torch.cat((memory['feat'], layer_feats.unsqueeze(3)), dim=3) if memory is not None else layer_feats.unsqueeze(3)
        # nl, b, c, t, n
        new_memory['xyz'] = torch.cat((memory['xyz'], xyz.unsqueeze(1)), dim=1) if memory is not None else xyz.unsqueeze(1)
        # b, t, n, 3
        new_memory['mask'] = torch.cat((memory['mask'], mask.unsqueeze(1)), dim=1) if memory is not None else mask.unsqueeze(1)
        # b, t, n
        if self.training:
            memory_size = self.cfg.train_memory_size
        else:
            memory_size = self.cfg.eval_memory_size
        
        if new_memory['feat'].shape[3] > memory_size:
            new_memory['feat'] = new_memory['feat'][:, :, :, 1:, :]
            new_memory['xyz'] = new_memory['xyz'][:, 1:, :, :]
            new_memory['mask'] = new_memory['mask'][:, 1:, :]

        return dict(
            memory=new_memory
        )

        
    def forward_localize(self, input):
        return self.loc_net(input)

    def forward_propagate(self, input):
        
        memory = input.pop('memory', None)
        feat = input.pop('feat')
        xyz = input.pop('xyz')
        if memory is None:
            assert 'first_mask_gt' in input
            first_mask_gt = input.pop('first_mask_gt')
            # b,n
            mem = dict(
                mask=first_mask_gt.unsqueeze(1), # b,1,n
            )
        else:
            mem = memory
        trfm_input = dict(
            memory=mem,
            feat=feat,
            xyz=xyz
        )
        trfm_output = self.transformer(trfm_input)
        return trfm_output

    def forward(self, input, mode):
        
        forward_dict = dict(
            embed=self.forward_embed,
            propagate=self.forward_propagate,
            localize=self.forward_localize,
            update=self.forward_update,
        )    
        assert mode in forward_dict, '%s has not been supported' % mode
        
        forward_func = forward_dict[mode]
        output = forward_func(input)
        return output
        