# OmniReward-Factory

### 🚀 Installation

```bash
git clone https://github.com/jinzhuoran/OmniReward-Factory.git
conda create -n omnireward python=3.10
conda activate omnireward
```

We recommend using **`torch==2.2.0`** for best compatibility.

Install PyTorch (choose one based on your CUDA version):

```bash
# For CUDA 11.8:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
cd OmniReward-Factory
pip install -r requirements.txt
```

### 🏋️‍♀️ Training Omni-Reward

```bash
bash scripts/train.sh
bash scripts/train_t2t.sh
bash scripts/train_ti2t.sh
bash scripts/train_t2iv.sh
```
