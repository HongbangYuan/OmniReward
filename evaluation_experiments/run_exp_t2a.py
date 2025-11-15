
import dashscope
from dashscope import MultiModalConversation  # 导入qwen-audio所需的模块

from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()


from utils.video_utils import OpenAI, IMAGE_TOKEN
from PIL import Image
import os
from openai import OpenAI
from utils.video_utils import  parse_judgement
from utils import write_to_json, read_json
from loguru import logger as eval_logger
import json
from copy import deepcopy
import re
from dataset.OmniRewardBench.load_omni_reward_bench import load_omni_reward_bench
import argparse
import importlib

def get_unique_id(elem):
    return elem["prompt"] + elem["criteria"] + elem["response1"] + elem["response2"]

import librosa
import numpy as np
import soundfile as sf
import tempfile
import base64

def merge_audio_files(audioA_path, audioB_path, target_sr=16000, silence_duration=1.0):
    """
    加载两个音频文件，插入 silence_duration 秒的静音作为分隔，
    合并后写入一个临时 WAV 文件，返回该临时文件的路径。
    """
    try:
        # 加载两个音频文件，确保采样率一致
        audioA, _ = librosa.load(audioA_path, sr=target_sr)
        audioB, _ = librosa.load(audioB_path, sr=target_sr)
        sr = target_sr

        # 生成 silence 信号
        silence = np.zeros(int(silence_duration * sr))

        # 合并音频：A + silence + B
        merged_audio = np.concatenate([audioA, silence, audioB])
        
        # 写入临时文件，使用 WAV 格式
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, merged_audio, sr)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        eval_logger.error(f"Failed to merge audio files {audioA_path} and {audioB_path}: {e}")
        return None

def encode_audio(audio_path):
    """
    读取本地音频文件并返回 base64 编码后的字符串
    """
    try:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        eval_logger.error(f"Failed to encode audio {audio_path}: {e}")
        return None


def call_qwen_omni_turbo_model(audioA_path, audioB_path, system_prompt,user_prompt):
    """
    合并两个本地音频文件后，通过 qwen-omni-turbo 模型进行推理，使用 streaming 模式返回结果。
    """
    try:
        # 合并两个音频文件，中间插入 1 秒静音
        merged_audio_path = merge_audio_files(audioA_path, audioB_path, target_sr=16000, silence_duration=1.0)
        if merged_audio_path is None:
            return None

        # 对合并后的音频文件进行 base64 编码
        base64_audio = encode_audio(merged_audio_path)
        if base64_audio is None:
            os.remove(merged_audio_path)
            return None

        # 根据临时文件的后缀确定格式（此处为 wav）
        file_ext = os.path.splitext(merged_audio_path)[1].lstrip('.').lower() or "wav"
        audio_data = f"data:;base64,{base64_audio}"

        # 构造消息：仅传递一个 input_audio，同时附带完整提示文本
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": file_ext,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # 发起聊天推理请求，使用 streaming 模式逐步返回输出
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )

        response_text = ""
        for chunk in completion:
            # 检查 chunk 是否包含 choices 且列表非空
            if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    response_text += delta.content
                elif delta.function_call and hasattr(delta.function_call, "arguments") and delta.function_call.arguments:
                    response_text += delta.function_call.arguments
            else:
                eval_logger.warning("Received a chunk with no choices, skipping.")
                
        # 删除临时合并的音频文件
        os.remove(merged_audio_path)
        return response_text
    except Exception as e:
        eval_logger.error(f"Error with audio {audioA_path} and {audioB_path}: {e}")
        return ""


if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加 --task 参数
    parser.add_argument(
        "--task",
        type=str,
        default="text_to_audio",
        choices=["text_to_text", "text_video_to_text", "text_image_to_text","text_to_video","text_to_image","text_image_to_image","text_to_3D","text_to_audio","text_audio_to_text"],
        help="Specify the task to run."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-omni-flash",
        help="Specify the model to evaluate."
    )

    parser.add_argument(
        "--model_version",
        type=str,
        help="Path to the model version."
    )

    parser.add_argument(
        "--api_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="URL for the API endpoint."
    )

    parser.add_argument(
        "--api_key_name",
        type=str,
        default='DASHSCOPE_API_KEY',
        help="API key for authentication."
    )
    # 添加 --continue 参数
    parser.add_argument(
        "--continue_eval",
        action="store_true",  # 如果提供了 --continue，则设置为 True
        help="continue evaluation from existing result file"
    )
    # 添加 --overwrite 参数
    parser.add_argument(
        "--overwrite",
        action="store_true",  # 如果提供了 --overwrite，则设置为 True
        help="overwrite the existing result file"
    )
    parser.add_argument(
        "--with_tie",
        action="store_true",  # 如果提供了 --overwrite，则设置为 True
        help="whether to use the tie-labeled examples"
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=0,
        help="whether to use the tie-labeled examples"
    )
    # 解析命令行参数
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task

    # 设置eval logger日志文件以及结果保存文件

    if args.with_tie:
        save_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/{model_name}_preferences_with_tie_inversed.json'
        log_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/log_{model_name}_preferences_with_tie_inversed.log'
    else:
        save_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/{model_name}_preferences_inversed.json'
        log_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/log_{model_name}_preferences_inversed.log'
    visual_path = '/home/hongbang/projects/ATTBenchmark/results/OmniRewardBenchBack20250315/media_data'
    eval_logger.add(
        log_file,
        level="INFO",
        rotation="20 MB",  # 文件大小达到 10MB 时轮转
        retention="7 days"  # 保留 7 天的日志
    )
    eval_logger.info(f"Start running for task {task}.")

    results = []
    id_set = set()
    id2sample = {}
    if args.continue_eval:
        if os.path.isfile(save_file):
            print(f"Continue eval from file {save_file}")
            results = read_json(save_file)
            results = [elem for elem in results if elem["predicted"] is not None]
            print(f"Load {len(results)} results...")
            id_set = set([get_unique_id(elem) for elem in results])
            # assert len(id_set) == len(results)
            id2sample = {get_unique_id(elem) :elem for elem in results}
        else:
            print(f"File {save_file} does not exists! Ignore the continue_eval parameter.")
    elif args.overwrite:
        if os.path.isfile(save_file):
            print(f"Choose to overwrite existing file {save_file}")
        else:
            print(f"File {save_file} does not exists! Ignore the overwrite parameter.")
    else:
        if os.path.isfile(save_file):
            raise ValueError(f"Save file {save_file} already exists! Please use --continue_eval or  --overwrite.")

    # load data and model service
    samples = load_omni_reward_bench(task=task)
    if not args.with_tie:
        samples = [elem for elem in samples if elem['criteria_preference'] != 'tie']
    eval_logger.info(f"Save file : {save_file}")

    # 从环境变量中获取 API 密钥
    api_key = os.getenv(args.api_key_name)

    if args.max_num_frames == 0:
        max_num_frames = 10 if task == 'text_video_to_text' else 5
    else:
        max_num_frames = args.max_num_frames
    assert max_num_frames >= 1


    client = OpenAI(
        # 请确保环境变量 DASHSCOPE_API_KEY 已配置，否则请直接替换为实际 API Key 字符串
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt_module = importlib.import_module(f"utils.prompts.{task}")
    system_prompt_template = prompt_module.system_prompt_template
    prompt_template = prompt_module.prompt_template
    system_prompt_template_with_tie = prompt_module.system_prompt_template_with_tie
    if args.with_tie:
        print("Evaluation with the tie examples!")

 
    N = len(samples)
    for idx, sample in enumerate(samples[:]):
        curr_id = get_unique_id(sample)
        if curr_id in id_set and id2sample[curr_id]["predicted"] is not None:
            # print(f"Sample {idx + 1} already in result file. Skip evaluation")
            continue

        print(f"************* Sample {idx + 1} / {N}  *******************")
        predicted = None
        response = None


        if args.with_tie:
            system_prompt = system_prompt_template_with_tie.format(
                criteria=sample["criteria"], 
            )
        else:
            system_prompt = system_prompt_template.format(
                criteria=sample["criteria"], 
            )
        user_prompt = prompt_template.format(
            prompt=sample["prompt"]
        )
        for retry in range(5):
            response = call_qwen_omni_turbo_model(
                audioA_path=os.path.join(visual_path,sample["response2"]),
                audioB_path=os.path.join(visual_path,sample["response1"]),
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
            print(response)
            predicted = parse_judgement(response)
            if predicted == 'response1':
                predicted = 'response2'
            elif predicted == 'response2':
                predicted = 'response1'
            if predicted is not None:
                break
            else:
                print(f"Get parsed none in response. Retrying ({retry+1}/5)...")
        sample["predicted"] = predicted
        print(f"Predicted:{predicted} / GroundTruth:{sample['criteria_preference']}")
        sample[f"{model_name}_raw_response"] = response
        results.append(deepcopy(sample))
        write_to_json(results, save_file, indent=4)
    write_to_json(results, save_file, indent=4)

    true_count = 0
    for elem in results:
        if elem["predicted"] == elem["criteria_preference"]:
            true_count += 1

    acc = true_count / len(results)
    eval_logger.info(f"Task:{task}")
    eval_logger.info(f"Acc: {acc:.4f} ({true_count}/{len(results)})")

    eval_logger.info(f"Successfully writing {len(results)} results to {save_file}!")
    eval_logger.info(f"Finished Running for task {task}!")
