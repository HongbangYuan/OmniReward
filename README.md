# <img src="./files/logo.png" alt="RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models" width="5%">  Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences


<p align="center">
  <a href="https://huggingface.co/datasets/HongbangYuan/OmniRewardBench"> 🤗 Benchmark</a></a> |
  <a href="https://hf.co/datasets/jinzhuoran/OmniRewardData"> 🤗 Dataset</a> | 
  <a href="https://hf.co/jinzhuoran/OmniRewardModel"> 🤗 Model</a> | 
  <a href="https://omnireward.github.io/"> 🏠 Homepage</a>
</p>

Reward models (RMs) play a critical role in aligning AI behaviors with human preferences.
We propose **Omni-Reward**, a step toward generalist omni-modal reward modeling with support for free-form
preferences, consisting of:

+ 📈 **Evaluation**: We introduce <a href="https://huggingface.co/datasets/HongbangYuan/OmniRewardBench"> OmniRewardBench</a></a>, the 
first omni-modal reward benchmark with free-form preferences, covering nine tasks 
across five modalities including text, image, video, audio, and 3D.

+ 📚 **Data**: We 
construct <a href="https://hf.co/datasets/jinzhuoran/OmniRewardData"> OmniRewardData</a> , a multimodal preference dataset comprising 248K 
general preference pairs and 69K instruction-tuning pairs for training generalist 
omni-modal RMs.

+ 🧠 **Model**: We propose <a href="https://hf.co/jinzhuoran/OmniRewardModel"> OmniRewardModel</a>, which includes 
both discriminative and generative RMs, and achieves strong performance on 
Omni-RewardBench as well as other widely used RM benchmark.


 
# 📈 Evaluation

## 🌐 Data Download
Our dataset is hosted on huggingface and we recommend downloading them with the following command.
```bash
huggingface-cli download HongbangYuan/OmniRewardBench --repo-type dataset --local-dir ./OmniRewardBench
```
⚠️ Note: The most time-consuming part of the download is the `media_data.zip` file (~3.5 GB), which contains all original image, audio, and video resources required for evaluation.
Depending on your internet speed, this step might take a while.

We recommend using the utility functions provided in `./dataset/OmniRewardBench/load_omni_reward_bench.py` for loading the dataset. You should specify the task argument to load data for a particular task. 

## 🚀 Running Evaluation  
 

To evaluate an API-accessible model on our full benchmark suite, you can run the provided launch script:

```bash
bash scripts/eval/run_eval_api.sh <your_model_name>
```
Remember to Sspecifying the model name as a command-line argument (e.g., `gpt-4`, `claude-3`) for logging and tracking.


The `scripts/eval/run_eval_api.sh` script supports:

* ✅ **Evaluating all tasks or selected ones**
  By default, the script runs on all supported tasks.
  To evaluate only specific tasks, simply comment out the unused tasks in the `tasks` list.

* ✅ **Two evaluation modes**
  For each task, the script runs:

  * Without Tie Evaluation (default)
  * WithTie valuation (`--with_tie`)

* ✅ **Parallel execution**
  Each pair of evaluations (w/ and w/o TIE) runs in parallel to speed up the process.

* ✅ **Customizable API endpoint**
  The API URL is set to `https://api.vveai.com/v1/chat/completions` by default.
  You can modify this value in the script to use any OpenAI-compatible endpoint. 
  For example, if you are serving a local model using [vLLM](https://github.com/vllm-project/vllm), you can set:
  ```bash
  api_url="http://localhost:8000/v1/chat/completions"
  ```
  This allows you to benchmark models hosted on your own machine.

 


# ⚙️ Training

## 🛠️ Environment Setup


To reproduce the training process in our paper, please make sure to set up the environment as described below.
Our training code is built upon the [llama-factory](https://github.com/hiyouga/llama-factory)  framework.

```bash
git clone https://github.com/HongbangYuan/OmniReward.git
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
cd OmniReward/OmniReward-Factory
pip install -r requirements.txt
```

## 🏋️‍♀️  Training Omni-Reward

To reproduce the training results described in our paper, please navigate to the OmniReward-Factory directory and run the following scripts:

```bash
cd OmniReward-Factory
bash scripts/train.sh
bash scripts/train_t2t.sh
bash scripts/train_ti2t.sh
bash scripts/train_t2iv.sh
```






