from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra
import wandb

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback

# only for debugging
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()

    # modify below to change checkpoints
    checkpoints = list(save_dir.glob(f'./*.ckpt'))

    # dim = 32
    # checkpoints = list(save_dir.glob(f'**/0625_013647/checkpoints/*.ckpt'))
    # checkpoints = list(save_dir.glob(f'**/0626_122912/checkpoints/*.ckpt')) # broken...
    # checkpoints = list(save_dir.glob(f'**/0626_172545/checkpoints/*.ckpt'))
    # checkpoints = list(save_dir.glob(f'**/0626_182857/checkpoints/*.ckpt'))
    # checkpoints = list(save_dir.glob(f'**/0626_211529/checkpoints/*.ckpt'))
    # checkpoints = list(save_dir.glob(f'**/0626_232705/checkpoints/*.ckpt')) 

    # dim = 64
    checkpoints = list(save_dir.glob(f'**/0627_140220/checkpoints/*.ckpt'))


    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    if cfg.wandb.active:
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)
    else:
        logger = pl.loggers.TensorBoardLogger(save_dir=cfg.experiment.save_dir,
                                            name=cfg.experiment.uuid)
        
    wandb.restore('model.ckpt')

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)
    
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)



    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         # strategy=DDPStrategy(find_unused_parameters=True),
                         accelerator="gpu",
                         **cfg.trainer,
                         fast_dev_run=False)
    
    # ckpt_path = None
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
