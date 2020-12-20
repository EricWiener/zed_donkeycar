# For the model
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim

def load_resnet18():
    # Load the pre-trained model (on ImageNet)
    model = models.resnet18(pretrained=True)

    # Don't allow model feature extraction layers to be modified
    for layer in model.parameters():
        layer.requires_grad = False

    # Change the classifier layer
    model.fc = nn.Linear(512, 2)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


class DonkeyTorch18(pl.LightningModule):
  def __init__(self, output_size):
    super().__init__()

    # Metrics
    self.train_acc = pl.metrics.Accuracy()
    self.valid_acc = pl.metrics.Accuracy()
    self.train_precision = pl.metrics.Precision()
    self.valid_precision = pl.metrics.Precision()

    self.model = load_resnet18()

  def forward(self, x):
    # Forward defines the prediction/inference action
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self.model(x)

    loss = F.l1_loss(logits, y)
    self.log('train_loss', loss)

    # Log Metrics
    self.train_acc(logits, y)
    self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

    self.train_precision(logits, y)
    self.log('train_precision', self.train_precision,
             on_step=False, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self.forward(x)
    loss = F.l1_loss(logits, y)

    # Log Metrics
    self.log('val_loss', loss)

    self.valid_acc(logits, y)
    self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

    self.valid_precision(logits, y)
    self.log('valid_precision', self.valid_precision,
             on_step=False, on_epoch=True)

  def configure_optimizers(self):
    optimizer = optim.Adam(
        self.model.parameters(), lr=0.0001, weight_decay=0.0005)
    return optimizer

  def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
      """
      Donkeycar parts interface to run the part in the loop.

      :param img_arr:     uint8 [0,255] numpy array with image data
      :param other_arr:   numpy array of additional data to be used in the
                          pilot, like IMU array for the IMU model or a
                          state vector in the Behavioural model
      :return:            tuple of (angle, throttle)
      """
      from PIL import Image
      pil_image = Image.fromarray(img_arr)
      return self.forward(pil_image)
