# ref:https://github.com/opendatalab/LOKI/blob/main/lm_evaluate/models/openai_api.py#L52


from abc import ABC, abstractmethod
from typing import Union, List

from PIL import Image


class LMM(ABC):
    prepared = False
    supported_modalities = []
    _rank = 0
    _world_size = 1
    support_batching = False

    def __init__(self):
        """
        Defines the base model class.
        All models should be able to do:
            1. Prepare model for evaluation, i.e., load huggingface checkpoints or prepare api credentials.
            2. Generate texts based on provided visuals and contexts.
        """
        self._model = None
        self._rank = 0  # For MultiGPU Inference
        self._world_size = 1  # For MultiGPU Inference

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def generate(
            self,
            visuals: Union[Image.Image, List[Image.Image], str, List[str]],
            contexts: Union[str, List[str]],
            **kwargs
    ):
        pass

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

MODEL_REGISTRY = {}
TASK_REGISTRY = {}

def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(cls, LMM), f"Model '{name}' ({cls.__name__}) must extend LMM class"

            assert name not in MODEL_REGISTRY, f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


import re
import base64
import os
import time
import requests as url_requests

from datetime import timedelta
from loguru import logger as eval_logger

try:
    import openai
except Exception as e:
    eval_logger.debug(f"Please install openai packages to use GPT\n{e}")

import PIL
import numpy as np
import torch

from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from io import BytesIO
from typing import List, Union
from copy import deepcopy

from decord import VideoReader, cpu
from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

NUM_SECONDS_TO_SLEEP = 30

def parse_judgement(generation: str):
    labels_dict = {
        "[[A]]": "response1",
        "[[B]]": "response2",
        r"\boxed{A}": "response1",
        r"\boxed{B}": "response2",
        "[[C]]": "tie",
        r"\boxed{C}": "tie",
    }
    if "[[A]]" in generation and "[[B]]" in generation:
        return None
    if r"\boxed{A}" in generation and r"\boxed{B}" in generation:
        return None
    for kw, label in labels_dict.items():
        if kw in generation:
            return label
    return None

def parse_judgement_score(generation: str):
    labels_dict = {
        "[[1]]": "1",
        "[[2]]": "2",
        "[[3]]": "3",
        "[[4]]": "4",
        "[[5]]": "5",
    }
    # 统计出现在 generation 中的标签数量
    matches = [label for label in labels_dict if label in generation]

    # 如果有两个或以上标签同时出现，返回 None
    if len(matches) >= 2:
        return None

    for kw, label in labels_dict.items():
        if kw in generation:
            return label
    return None

@register_model("gpt-4o", "gpt-4v", "qwen")
class OpenAI(LMM):
    supported_modalities = ["video", "image", "text"]

    def __init__(
            self,
            model_version: str = "gpt-4o",
            api_url: str = API_URL,
            max_num_frames: int = 10,
            timeout: int = 120,
            api_type: str = "openai",
            api_key: str = os.getenv("OPENAI_API_KEY", "YOUR_KEY"),
            default_headers: dict = None,
    ) -> None:

        self.model_version = model_version
        self.api_url = api_url
        self.api_key = api_key

        self.timeout = timeout
        self.max_num_frames = max_num_frames
        self.api_type = api_type

        if default_headers:
            self.default_headers = default_headers

        self.prepare_model()

    def encode_image(self, image: Image.Image) -> str:
        output_buffer = BytesIO()

        # 新增代码：转换颜色模式
        if image.mode in ('RGBA', 'LA'):
            # 创建一个白色背景来替代透明区域
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # 用alpha通道作为mask
            image = background
        else:
            image = image.convert('RGB')  # 其他模式统一转RGB

        image.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def encode_video(self, video_path: str, for_get_frames_num: int) -> List[str]:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            base64_frames.append(self.encode_image(img))

        return base64_frames

    def prepare_model(self):

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU,
                                                    DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": 1,
                    "train_batch_size": 1 * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        if "qwen" in self.model_version and "vl" not in self.model_version:
            self.supported_modalities = ["text_only"]

        self._set_headers()

        self.prepared = True
        eval_logger.info(f"GPT activated. API_URL: {self.api_url}. MODEL_VERSION: {self.model_version}")

    def _prepare_generate_kwargs(self, payload, generate_kwargs):
        if "max_new_tokens" in generate_kwargs:
            payload["max_tokens"] = generate_kwargs["max_new_tokens"]
        if "temperature" in generate_kwargs:
            payload["temperature"] = generate_kwargs["temperature"]
        if "top_p" in generate_kwargs:
            payload["top_p"] = generate_kwargs["top_p"]
        if "num_beams" in generate_kwargs:
            payload["num_beams"] = generate_kwargs["num_beams"]
        if "response_format" in generate_kwargs.keys() and generate_kwargs["response_format"] is not None:
            payload["response_format"] = generate_kwargs["response_format"]

        return payload

    def _set_headers(self):
        if self.api_type == "openai" and (
                'gpt' in self.model_version or 'claude' in self.model_version or 'llama' in self.model_version or 'gemini' in self.model_version):
            api_key = self.api_key
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        elif self.api_type == "azure" and 'gpt' in self.model_version:
            api_key = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
            self.headers = {
                "api-key": api_key,
                "Content-Type": "application/json",
            }
        elif self.api_type == "openai" and 'qwen' in self.model_version:
            api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            self.headers = {
                "Authorization": "EMPTY",
                "Content-Type": "application/json",
            }
        # 将 default_headers 合并到 headers 中
        if hasattr(self, 'default_headers') and self.default_headers:
            self.headers.update(self.default_headers)

    def generate(
            self,
            visuals: Union[Image.Image, str, List[Union[Image.Image, str]]] = None,
            contexts: str = None,
            system_prompt: str = None,
            **generate_kwargs
    ) -> str:
        """
            Call gpt for response with visuals and contexts. Visuals can be a list of strings(representing video paths), or a list of PIL.Image.Image or a combination of both. Returns a piece of response text.

            Args:
                visuals: Media objects. Visuals can be one image, one video path, or a list of them.
                contexts: Prompt text.
                generate_kwargs: Generation kwargs. Currently we only support greedy decoding. # TODO: Support diverse decoding strategy.
            Return:
                A piece of response text
        """
        imgs = []
        video_frames = 0

        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, str):
                    video_frames += len(self.encode_video(visual, self.max_num_frames))
                    imgs.extend(self.encode_video(visual, self.max_num_frames))
                elif isinstance(visual, Image.Image):
                    imgs.append(self.encode_image(visual))
                else:
                    error_msg = f"Expected visual type to be Image.Image or str. Got: {type(visual)}"
                    eval_logger.error(TypeError(error_msg))
        elif isinstance(visuals, str):
            video_frames += len(self.encode_video(visuals, self.max_num_frames))
            imgs.extend(self.encode_video(visuals, self.max_num_frames))

        elif isinstance(visuals, Image.Image):
            imgs.append(self.encode_image(visuals))

        # Construct messages for request

        # First, convert video place holders to image place holders according to max_num_frames
        if video_frames > 0:
            contexts = contexts.replace(VIDEO_TOKEN, f"{IMAGE_TOKEN} " * (video_frames // contexts.count("<video>")))

        payload = {"messages": []}

        if API_TYPE == "openai":
            payload["model"] = self.model_version

        if contexts.count(IMAGE_TOKEN) != len(imgs):
            eval_logger.warning(
                f"Image tokens are mismatched! Contexts: {contexts}. Image token num: {contexts.count(IMAGE_TOKEN)}. Number of images(frames) num: {len(imgs)}")
            if contexts.count(IMAGE_TOKEN) < len(imgs):
                contexts = f"{IMAGE_TOKEN} " * (len(imgs) - (contexts.count(IMAGE_TOKEN))) + contexts

        contexts = contexts.split(IMAGE_TOKEN)

        response_json = {"role": "user", "content": []}

        payload["messages"].append(deepcopy(response_json))

        for idx, img in enumerate(imgs):
            if not all(s == " " or s == '\n' for s in contexts[idx]):
                payload["messages"][0]["content"].append({"type": "text", "text": contexts[idx]})
            payload["messages"][0]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

        # If n image tokens are in the contexts
        # contexts will be splitted into n+1 chunks
        # Manually add it into the payload
        if not all(s == " " or s == '\n' for s in contexts[-1]):
            payload["messages"][0]["content"].append({"type": "text", "text": contexts[-1]})

        payload = self._prepare_generate_kwargs(payload, generate_kwargs)
        if system_prompt:
            payload["messages"].insert(0,{"role":"system","content":system_prompt})

        for attempt in range(5):
            try:
                response = url_requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
                eval_logger.debug(response)
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"].strip()
                break  # If successful, break out of the loop

            except Exception as e:
                try:
                    error_msg = response.json()
                except:
                    error_msg = ""

                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nResponse: {error_msg}.")
                if system_prompt:
                    contents = [content for content in payload["messages"][1]["content"] if content["type"] == "text"]
                else:
                    contents = [content for content in payload["messages"][0]["content"] if content["type"] == "text"]
                eval_logger.info(f"Content: {contents}")
                if attempt < 4:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {error_msg}")
                    response_text = ""

        return response_text


if __name__ == '__main__':
    # test the newly created video llm utils
    print("Hello World!")