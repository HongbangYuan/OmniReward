#!/bin/bash
# 在脚本执行期间按 Ctrl + C 时，杀死脚本及其所有子进程；启用以下信号捕获
trap "echo 'Terminating all subprocesses...'; kill 0; exit 130" SIGINT

export PYTHONPATH='/home/hongbang/projects/ATTBenchmark'

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "请提供一个基于API的模型名称作为参数。"
  exit 1
fi

# 定义任务列表
tasks=(
    "text_to_text"
    "text_video_to_text"
    "text_image_to_text"
    "text_to_video"
    "text_to_image"
    "text_image_to_image"
    "text_to_3D"
)


model_name="$1"
echo "Model Name: ${model_name}"
model_version=$model_name
api_url="https://api.vveai.com/v1/chat/completions"

# 遍历每个任务
for task in "${tasks[@]}"; do
    # 运行不带 --with_tie 参数的实验
    echo "Running command: python experiments/run_exp.py --task $task --model_name $model_name --model_version $model_version --api_url $api_url"
    python experiments/run_exp.py --task "$task" --model_name "$model_name" --model_version "$model_version" --api_url "$api_url" --continue_eval &

    # 运行带 --with_tie 参数的实验
    echo "Running command: python experiments/run_exp.py --task $task --model_name $model_name --model_version $model_version --api_url $api_url --with_tie"
    python experiments/run_exp.py --task "$task" --model_name "$model_name" --model_version "$model_version" --api_url "$api_url" --with_tie --continue_eval &

    # 等待同一任务的两个进程执行完毕后，再进行下一个任务
    wait
    echo "$task completed."
done

echo "All experiments completed."