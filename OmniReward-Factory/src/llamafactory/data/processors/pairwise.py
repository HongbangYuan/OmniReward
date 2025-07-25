# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_mm_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    audios: Sequence["AudioInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if len(images) != 0 and len(videos) != 0:
        chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], [images[0]], [videos[0]], audios, processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], [images[1]], [videos[1]], audios, processor)
    elif len(images) != 0:
        if isinstance(images[0], list):
            chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images[0], videos, audios, processor)
            rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images[1], videos, audios, processor)
        elif len(images) > 2:
            chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images[:len(images) // 2], videos, audios, processor)
            rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images[len(images) // 2:], videos, audios, processor)
        else:
            chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], [images[0]], videos, audios, processor)
            rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], [images[1]], videos, audios, processor)
    elif len(videos) != 0:
        chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, [videos[0]], audios, processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, [videos[1]], audios, processor)
    elif len(audios) != 0:
        chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, [audios[0]], processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, [audios[1]], processor)
    else:
        chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_chosen_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    prompt_rejected_ids, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    if len(images) != 0 and len(videos) != 0:
        prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, [images[0]], [videos[0]], audios, tokenizer, processor)
        prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, [images[1]], [videos[0]], audios, tokenizer, processor)
    elif len(images) != 0:
        if isinstance(images[0], list):
            prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, images[0], videos,
                                                                        audios, tokenizer, processor)
            prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, images[1],
                                                                          videos, audios, tokenizer, processor)
        elif len(images) > 2:
            prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, images[:len(images) // 2], videos, audios, tokenizer, processor)
            prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, images[len(images) // 2:], videos, audios, tokenizer, processor)
        else:
            prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, [images[0]], videos, audios, tokenizer, processor)
            prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, [images[1]], videos, audios, tokenizer, processor)
    elif len(videos) != 0:
        prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, images, [videos[0]], audios, tokenizer, processor)
        prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, images, [videos[1]], audios, tokenizer, processor)
    elif len(audios) != 0:
        prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, images, videos, [audios[0]], tokenizer, processor)
        prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, images, videos, [audios[1]], tokenizer, processor)
    else:
        prompt_chosen_ids, _ = template.mm_plugin.process_token_ids(prompt_chosen_ids, None, images, videos, audios, tokenizer, processor)
        prompt_rejected_ids, _ = template.mm_plugin.process_token_ids(prompt_rejected_ids, None, images, videos, audios, tokenizer, processor)

    # consider the response is more important
    target_len, source_len = infer_seqlen(max(len(chosen_ids), len(rejected_ids)), max(len(prompt_chosen_ids), len(prompt_rejected_ids)), cutoff_len)
    prompt_chosen_ids = prompt_chosen_ids[:source_len]
    prompt_rejected_ids = prompt_rejected_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_chosen_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_rejected_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    audios: Sequence["AudioInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, audios, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, audios, processor)
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, audios, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        if (examples["_images"][i] is not None and len(examples["_images"][i]) == 2) or (examples["_videos"][i] is not None and len(examples["_videos"][i]) == 2) \
                or (examples["_audios"][i] is not None and len(examples["_audios"][i]) == 2):
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_mm_pairwise_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
            )
            split = False
        elif (examples["_images"][i] is not None and len(examples["_images"][i]) > 2 and examples["_response"][i][0] == examples["_response"][i][1]):
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_mm_pairwise_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
            )
            split = True
        else:
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
            )
            split = False
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["audios"].append(examples["_audios"][i])
        model_inputs["split"].append(split)

    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
