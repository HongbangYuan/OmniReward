import os
from utils import read_json

def load_omni_reward_bench(task="text_to_text"):
    """
    Load OmniRewardBench data for a given task.

    :param task: a string specifying the task type (e.g., 'text_to_text').
    :return: a dict containing the loaded JSON data, or None if no file is found.
    """
    # Define base paths
    base_paths = [
        "/home/hongbang/projects/ATTBenchmark/results/OmniRewardBenchBack20250315",
        "/home/zhuoran/hongbang/projects/ATTBenchmark/results/OmniRewardBenchBack20250315",
        "/home/zhuoran/hongbang/projects/ATTBenchmark/results/OmniRewardBenchBack20250315",
        "/mnt/userdata/projects/ATTBenchmark/results/OmniRewardBenchBack20250315",
    ]

    # Task-to-file mapping
    task_files = {
        "text_to_text": "text_to_text_preferences.json",
        "text_to_image": "text_to_image_preferences.json",
        "text_image_to_text": "text_image_to_text_preferences.json",
        "text_image_to_image": "text_image_to_image_preferences.json",
        "text_to_3D": "text_to_3D_preferences.json",
        "text_video_to_text": "text_video_to_text_preferences.json",
        "text_to_video": "text_to_video_preferences.json",
        "text_to_audio": "text_to_audio_preferences.json",
        "text_audio_to_text": "text_audio_to_text_preferences.json"
    }

    # Check if the requested task is supported
    if task not in task_files:
        raise ValueError(f"Task '{task}' is not supported.")

    # Construct candidate file paths
    filename = task_files[task]
    file_paths = [
        os.path.join(base_path, filename) for base_path in base_paths
    ]

    # Attempt to read the first existing file
    human_result = None
    for path in file_paths:
        if os.path.exists(path):
            human_result = read_json(path)
            print(f"Read data from {path}")
            break

    if human_result is None:
        print(f"No files found for task '{task}'!")

    return human_result


if __name__ == '__main__':
    print("Hello World!")
    samples = load_omni_reward_bench(task='text_to_video')





