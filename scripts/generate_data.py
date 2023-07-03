import sys
sys.path.insert(0, '.')

import torch
import json
import hydra
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.data.transforms import LoadDataTransform
from cross_view_transformer.common import setup_config, setup_data_module, setup_viz


def setup(cfg):
    # Don't change these
    cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False

    # Uncomment to debug errors hidden by multiprocessing
    # cfg.loader.num_workers = 0
    # cfg.loader.prefetch_factor = 2
    # cfg.loader.persistent_workers = False

@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')    # hydra main function
def main(cfg):    # main function
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        01234.json
        01234/
            bev_0001.png
            bev_0002.png
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)    # setup the configuration

    data = setup_data_module(cfg)    # setup the dataset
    viz_fn = None

    if 'visualization' in cfg:    # if visualization flag is passed then
        viz_fn = setup_viz(cfg)    # setup the visualization
        load_xform = LoadDataTransform(cfg.data.dataset_dir, cfg.data.labels_dir,
                                       cfg.data.image, cfg.data.num_classes)    # Load the data into memory

    labels_dir = Path(cfg.data.labels_dir)    # Path of labels directory
    labels_dir.mkdir(parents=False, exist_ok=True)    # Make directory if not exist

    for split in ['train', 'val']:    # For each split in train and test
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):    # Iterating through each episode
            episode.__getitem__(0)
            scene_dir = labels_dir / episode.scene_name     # Path of scene directory
            scene_dir.mkdir(exist_ok=True, parents=False)   # Make the directory if not exist

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)   # Load the data into memory
            info = []   # List to store the data

            for i, batch in enumerate(tqdm(loader, position=1, leave=False)):    # Iterate through each batch
                info.extend(batch)   # Extend the list with batch data

                # Load data from disk to test if it was saved correctly
                if i == 0 and viz_fn is not None:    # if visualization flag is passed then
                    unbatched = [load_xform(s) for s in batch]    # Load data in memory
                    rebatched = torch.utils.data.dataloader.default_collate(unbatched)  

                    viz = np.vstack(viz_fn(rebatched))    # Show the data on screen

                    cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))    # Show the image on screen
                    cv2.waitKey(1)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'    # Save the data into json format
            scene_json.write_text(json.dumps(info))    # Dumps the data


if __name__ == '__main__':
    main()
