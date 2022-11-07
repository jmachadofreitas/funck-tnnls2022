"""
Adult (50k) vs KDD Census Income (300k):
    Note that Incomes have been binned at the $50K level to present a binary classification problem,
    much like the original UCI/ADULT database.
    The goal field of this data, however, was drawn from the "total person income" field rather than the
    "adjusted gross income" and may, therefore, behave differently than the orginal ADULT goal field.
"""
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
from filelock import FileLock
import itertools
from dataclasses import dataclass
from enum import Enum, auto

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.utils import download_url
import pytorch_lightning as pl
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder

from ..utils import get_logger
from .shared import *


SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz"
TRAIN_RAW_FILENAME = "census-income.data"
TEST_RAW_FILENAME = "census-income.test"

CONTEXT_NAME = "sex"
TARGET_NAME = "Income"
POSITIVE_OUTCOME = "50000+"
RAW_COLUMNS = OrderedDict([
    ("sex", F(CAT, ["Male", "Female", "Unknown/Invalid"])),  # context
    ("Income", F(ORD, ["- 50000", "+ 50000"]))  # target
])
DROP = ["encounter_id", "weight", "payer_code", "medical_specialty"]

# FIXME: Like Adult but with 300k instead of 50k observations?
