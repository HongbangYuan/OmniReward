from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()


from utils.video_utils import OpenAI, IMAGE_TOKEN
from PIL import Image
import os
from utils.video_utils import OpenAI, VIDEO_TOKEN, parse_judgement
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

def get_response_from_sample(client,sample,task,with_tie,visual_path):
    criteria = sample["criteria"]
    # 动态导入对应task的prompt模板
    try:
        prompt_module = importlib.import_module(f"utils.prompts.{task}")
        system_prompt_template = prompt_module.system_prompt_template
        prompt_template = prompt_module.prompt_template
        system_prompt_template_with_tie = prompt_module.system_prompt_template_with_tie
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Prompt templates for task '{task}' not found: {str(e)}")
    if with_tie:
        system_prompt = system_prompt_template_with_tie.format(criteria=criteria)
    else:
        system_prompt = system_prompt_template.format(criteria=criteria)

    if task == 'text_to_text':
        prompt = sample["prompt"]
        responseA, responseB = sample["response1"], sample["response2"]
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=responseA,
            responseB=responseB,
        )
        response = client.generate(
            contexts=full_prompt,
            system_prompt=system_prompt,
        )
    elif task == 'text_to_image' or task == 'text_to_3D':
        prompt = sample["prompt"]
        responseA_image = os.path.join(visual_path, sample["response1"])
        responseB_image = os.path.join(visual_path, sample["response2"])
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=IMAGE_TOKEN,
            responseB=IMAGE_TOKEN,
        )
        response = client.generate(
            contexts=full_prompt,
            system_prompt=system_prompt,
            visuals=[Image.open(elem) for elem in [responseA_image, responseB_image]],
        )

    elif task == 'text_to_video':
        prompt = sample["prompt"]
        responseA_video = os.path.join(visual_path, sample["response1"])
        responseB_video = os.path.join(visual_path, sample["response2"])
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=VIDEO_TOKEN,
            responseB=VIDEO_TOKEN,
        )
        response = client.generate(
            contexts=full_prompt,
            system_prompt=system_prompt,
            visuals=[responseA_video, responseB_video],
        )
    elif task == 'text_video_to_text':
        prompt = sample["prompt"]
        responseA = sample["response1"]
        responseB = sample["response2"]
        video_path = os.path.join(visual_path, sample["video"])
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=responseA,
            responseB=responseB,
        )
        response = client.generate(
            contexts=full_prompt + f"{VIDEO_TOKEN}",
            system_prompt=system_prompt,
            visuals=video_path,
        )
    elif task == 'text_image_to_text':
        prompt = sample["prompt"]
        image_path = os.path.join(visual_path,sample["image"])
        responseA,responseB = sample["response1"],sample["response2"]
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=responseA,
            responseB=responseB,
            image=IMAGE_TOKEN,
        )
        response = client.generate(
            contexts=full_prompt,
            system_prompt=system_prompt,
            visuals=[Image.open(elem) for elem in [image_path]],
        )
    elif task == 'text_image_to_image':
        prompt = sample["prompt"]
        image_path = os.path.join(visual_path,sample["image"])
        responseA_image = os.path.join(visual_path,sample["response1"])
        responseB_image = os.path.join(visual_path,sample["response2"])
        full_prompt = prompt_template.format(
            prompt=prompt,
            responseA=IMAGE_TOKEN,
            responseB=IMAGE_TOKEN,
            original_image=IMAGE_TOKEN,
        )
        response = client.generate(
            contexts=full_prompt,
            system_prompt=system_prompt,
            visuals=[Image.open(elem) for elem in [image_path,responseA_image,responseB_image]],
        )
    else:
        raise f"Task {task} is not supported!"

    return response

if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加 --task 参数
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["text_to_text", "text_video_to_text", "text_image_to_text","text_to_video","text_to_image","text_image_to_image","text_to_3D"],
        help="Specify the task to run."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Specify the model to evaluate."
    )

    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Path to the model version."
    )

    parser.add_argument(
        "--api_url",
        type=str,
        required=True,
        help="URL for the API endpoint."
    )

    parser.add_argument(
        "--api_key_name",
        type=str,
        default='API_KEY',
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
        save_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/{model_name}_preferences_with_tie.json'
        log_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/log_{model_name}_preferences_with_tie.log'
    else:
        save_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/{model_name}_preferences.json'
        log_file = f'/home/hongbang/projects/ATTBenchmark/results/omnireward_preference_eval/{task}/log_{model_name}_preferences.log'
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
        model_version=args.model_version,
        api_type='openai',
        api_key=api_key,
        api_url=args.api_url,
        max_num_frames=max_num_frames,
    )

    N = len(samples)
    for idx, sample in enumerate(samples[:]):
        curr_id = get_unique_id(sample)
        if curr_id in id_set and id2sample[curr_id]["predicted"] is not None:
            # print(f"Sample {idx + 1} already in result file. Skip evaluation")
            continue

        print(f"************* Sample {idx + 1} / {N}  *******************")
        predicted = None
        response = None
        for retry in range(5):
            response = get_response_from_sample(
                client=client,
                sample=sample,
                task=task,
                with_tie=args.with_tie,
                visual_path=visual_path
            )
            print(response)
            predicted = parse_judgement(response)
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
