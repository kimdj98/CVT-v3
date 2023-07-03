import torch
import re

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torchmetrics import MetricCollection
from pathlib import Path

from .model.model_module import ModelModule
from .data.data_module import DataModule
from .losses import MultipleLoss

from collections.abc import Callable
from typing import Tuple, Dict, Optional

from omegaconf import open_dict

def setup_config(cfg: DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)

    if override is not None:
        override(cfg)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)


def setup_network(cfg: DictConfig):
    return instantiate(cfg.model)


def setup_model_module(cfg: DictConfig) -> ModelModule:
    backbone = setup_network(cfg)
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})
    model_module = ModelModule(backbone, loss_func, metrics,
                               cfg.optimizer, cfg.scheduler,
                               cfg=cfg)

    return model_module


def setup_data_module(cfg: DictConfig) -> DataModule:
    return DataModule(cfg.data.dataset, cfg.data, cfg.loader)


def setup_viz(cfg: DictConfig) -> Callable:
    return instantiate(cfg.visualization)


def setup_experiment(cfg: DictConfig) -> Tuple[ModelModule, DataModule, Callable]:
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)
    viz_fn = setup_viz(cfg)

    return model_module, data_module, viz_fn


def load_backbone(checkpoint_path: str, backbone, prefix: str = 'backbone'):
    checkpoint = torch.load(checkpoint_path)

    cfg = DictConfig(checkpoint['hyper_parameters'])

    if 'cvt_nuscenes_vehicles_50k' in str(checkpoint_path): # if load pretrained model for the first-time
        with open_dict(cfg.model.outputs):
            cfg.model.outputs['velocity_x'] = [2, 3]
            cfg.model.outputs['velocity_y'] = [3, 4]
            
    # cfg = OmegaConf.to_object(checkpoint['hyper_parameters'])
    # cfg['model']['encoder']['backbone']['image_height'] = cfg['model']['encoder']['backbone'].pop('input_height')
    # cfg['model']['encoder']['backbone']['image_width'] = cfg['model']['encoder']['backbone'].pop('input_width')
    # cfg['model']['encoder']['cross_view'].pop('spherical')
    # cfg['model']['encoder']['bev_embedding']['sigma'] = 1.0
    # cfg['model']['encoder']['bev_embedding']['offset'] = 0.0
    cfg = DictConfig(cfg)

    state_dict = remove_prefix(checkpoint['state_dict'], prefix)

    # remove parts where different architecture is used
    if 'cvt_nuscenes_vehicles_50k' in str(checkpoint_path):
        keys_to_remove = [key for key in state_dict.keys() if (
                                                            # 'to_logits' in key \
                                                            # or re.match(r'decoder\.layers\.\d+\.conv\.1\.weight', key)\
                                                            # or re.match(r'decoder\.layers\.\d+\.up\.weight', key)\
                                                            # or re.match(r'backbone\.decoder\.layers\.conv\.1\.weight', key)\
                                                            # or re.match(r'backbone\.decoder\.layers\.\d+\.up\.weight', key)\
                                                            # or re.match(r'backbone\.to_logits\.3\.(weight|bias)', key)
                                                            )]

        for key in keys_to_remove:
            del state_dict[key]
    
    backbone.load_state_dict(state_dict, strict=False)

    return backbone


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

        key = '.'.join(tokens)
        result[key] = v

    return result
