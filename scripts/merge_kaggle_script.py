with open(r'src/model_moe_v3.py', 'r', encoding='utf-8') as f:
    m_code = f.read()

with open(r'scripts/train_v3_kaggle.py', 'r', encoding='utf-8') as f:
    t_code = f.read()

t_code_clean = t_code.replace('from model_moe_v3 import DorinaMoEV3, MoEConfig, get_amp_precision', '# Model components defined above')

# model_moe_v3 içindeki selftest ve benchmark main'ini dışarıda bırakıyoruz
m_parts = m_code.split('def main():')
m_core = m_parts[0]

combined = f'''#!/usr/bin/env python3
"""
Dorina V3 All-in-One Standalone DDP Trainer for Kaggle
Optimized for Dual Tesla T4 GPUs (32GB VRAM total)
"""
import os
import sys
import time
import math
import argparse
import glob
from dataclasses import dataclass, replace
from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

# ----------------- Dorina V3 Architecture -----------------
{m_core}

# ----------------- DDP Training Loop -----------------
{t_code_clean}
'''

with open(r'kaggle_kernel/train_v3_ddp.py', 'w', encoding='utf-8') as f:
    f.write(combined)

print("Generated kaggle_kernel/train_v3_ddp.py, size:", len(combined))
