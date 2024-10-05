import os
import argparse
import numpy as np
import torch
import torch.nn as nn


def train(opt):
    torch.manual_seed(123)
    