import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import torchvision
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torchvision import datasets
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchvision.transforms import ToTensor
from google.colab import drive
from timeit import default_timer as timer
import random
import functools
