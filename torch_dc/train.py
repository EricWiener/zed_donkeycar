import math
import os
from pathlib import Path
from typing import List, Any

from .types import TubRecord
from donkeycar.pipeline.sequence import TubSequence
from .types import TubDataset
from donkeycar.pipeline.augmentations import ImageAugmentation
from donkeycar.utils import get_model_by_type, normalize_image

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from donkeycar.parts.tub_v2 import Tub

# for get_default_transform
from torchvision import transforms


from .DonkeyTorch18 import DonkeyTorch18
import pytorch_lightning as pl


def get_default_transform(for_video=False, for_inference=False):
    """
    Creates a default transform to work with torchvision models

    Video transform:
    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB videos of shape (3 x T x H x W), 
    where H and W are expected to be 112, and T is a number of video frames 
    in a clip. The images have to be loaded in to a range of [0, 1] and 
    then normalized using mean = [0.43216, 0.394666, 0.37645] and 
    std = [0.22803, 0.22145, 0.216989].
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = (224, 224)

    if for_video:
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        input_size = (112, 112)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform

class TorchTubDataset(Dataset):
    '''
    Loads the dataset, and creates a train/test split.
    '''

    def __init__(self, config: Any, tub_paths: List[str], transform=None):
        """Create a PyTorch Tub Dataset

        Args:
            config (object): the configuration information
            tub_paths (List[str]): a list of paths to the tubs to use (minimum size of 1).
                                   Each tub path corresponds to another training run.
            shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        """
        self.config = config
        self.tub_paths = tub_paths

        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()

        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()

        # Loop through all the different tubs and load all the records for each of them
        for tub in self.tubs:
            for underlying in tub:
                record = TubRecord(self.config, tub.base_path,
                                   underlying=underlying)
                self.records.append(record)

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return torch.tensor([angle, throttle])

    def x_transform(self, record: TubRecord):
        # Loads the result of Image.open()
        img_arr = record.image(cached=True, as_nparray=False)
        return self.transform(img_arr)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        y = self.y_transform(self.records[idx])
        x = self.x_transform(self.records[idx])
        return x, y


def train(cfg, tub_paths, model, model_type):
    """
    Train the model
    """
    model_name, model_ext = os.path.splitext(model)

    # ######
    is_torch_model = model_ext == '.pt'
    if is_torch_model:
        model = f'{model_name}.pt'

    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    tubs = tub_paths.split(',')
    tub_paths = [os.path.expanduser(tub) for tub in tubs]
    # train_type = 'linear' if 'linear' in model_type else model_type

    dataset = TorchTubDataset(cfg, tub_paths)
    train_size = int(len(dataset) * cfg.TRAIN_TEST_SPLIT)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    print('Records # Training %s' % train_size)
    print('Records # Validation %s' % val_size)

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    }

    assert val_size > 0, "Not enough validation data, decrease the batch " \
                         "size or add more data."

    if torch.cuda.is_available():
        print('Using CUDA')
        gpus = -1
    else:
        print('Not using CUDA')
        gpus = 0


    logger = None
    if cfg.VERBOSE_TRAIN:
        from pytorch_lightning.loggers import TensorBoardLogger

        # Create Tensorboard logger
        logger = TensorBoardLogger('tb_logs', name='DonkeyNet')

    output_dir = Path(model).parent

    cfg.MAX_EPOCHS = 3
    trainer = pl.Trainer(gpus=gpus, logger=logger, progress_bar_refresh_rate=30,
                         max_epochs=cfg.MAX_EPOCHS, default_root_dir=output_dir)
    model = DonkeyTorch18(output_size=2)

    if cfg.PRINT_MODEL_SUMMARY:
        # print(kl.model.summary())
        pass

    trainer.fit(model, data_loader['train'], data_loader['val'])

    # if is_torch_model:
    #     checkpoint_model_path = f'{os.path.splitext(output_path)[0]}.pt'
    # @TODO save mode to path checkpoint_model_path

    return None
