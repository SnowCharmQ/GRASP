import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText
from model.processing_qwen3_vl import (
    BatchFeature,
    Qwen3VLProcessor,
)
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from model.monkey_patch_timestamp_format import apply_timestamp_format_patch
from eval.vllm_inference.utils import get_dataset_type

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video evaluation (temporal grounding or MCQ) with Qwen3-VL"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="charades",
        choices=[
            "charades",
            "activitynet",
            "qvhighlights",
            "videomme",
            "mlvu",
            "tempcompass",
            "longvideobench",
            "lvbench",
            "mvbench",
            "cgbench",
            "auroracap",
        ],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path or hub id for the Qwen3-VL-8B model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (dataset-specific)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root folder containing dataset videos and annotations",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=64, help="Generation budget"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples for a quick sanity check",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="If set, save per-sample predictions to this jsonl file",
    )
    parser.add_argument(
        "--timing_json",
        type=str,
        default=None,
        help=(
            "Path to write timing JSON (infer_seconds, total_seconds, etc.). "
            "If unset but --output_jsonl is set, defaults to <output_stem>_timing.json beside it."
        ),
    )
    parser.add_argument(
        "--curr_idx",
        type=int,
        default=0,
        help="Current shard index (0-based) when running multi-GPU eval",
    )
    parser.add_argument(
        "--total_idx",
        type=int,
        default=1,
        help="Total number of shards; set to num GPUs for data parallel eval",
    )
    parser.add_argument(
        "--ts_format",
        type=str,
        default="seconds",
        help="Timestamp format style for video prompts; 'seconds' mimics the original behavior.",
    )
    parser.add_argument(
        "--ts_prompt",
        type=str,
        default=None,
        help=(
            "Compact timestamp formatting string, e.g. "
            "'seconds;unit=1;brackets=0;append_colon=1;decimals=1;trailing_colon=0'. "
            "Overrides individual ts_format/units if provided."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override to control timestamp calculation for video inputs",
    )
    parser.add_argument(
        "--no_duration_in_prompt",
        action="store_false",
        dest="include_duration",
        help="If set, omit the video duration line from the text prompt",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for inference instead of transformers.generate",
    )
    parser.add_argument(
        "--use_timelens",
        action="store_true",
        help="Use TimeLENS-processed annotations if available (for Charades/ActivityNet)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=256,
        help="Maximum number of frames to sample from each video",
    )
    parser.add_argument("--total_tokens", type=int, default=8192)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM. If not set, uses all visible GPUs.",
    )

    parser.add_argument(
        "--nothink",
        type=str,
        default="False",
    )

    return parser.parse_args()


def prepare_inputs_for_vllm(messages, processor):
    # Build chat prompt string with generation marker
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils >= 0.0.14 required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def load_charades(split: str, data_root: str, use_timelens: bool = True):
    if use_timelens:
        ann_path = Path(data_root) / "Charades_anno" / f"charades_timelens.json"
    # else:
    #     ann_path = Path(data_root) / "Charades_anno" / f"Charades_sta_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ann_path}. Generate it via eval/vllm_inference/data/data_loader.py"
        )
    data = json.load(open(ann_path))
    samples = []
    qid = 0
    for vid, meta in data.items():
        video_path = Path(data_root) / "Charades_v1" / f"{vid}.mp4"
        for ts, sent in zip(meta["timestamps"], meta["sentences"]):
            samples.append(
                {
                    "qid": f"charades_{qid}",
                    "video": str(video_path),
                    "duration": meta["duration"],
                    "timestamp": ts,
                    "sentence": sent.strip(),
                }
            )
            qid += 1
    return samples


def load_activitynet(split: str, data_root: str, use_timelens: bool = True):
    if split == "default":
        split = "val"
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train/val/test for ActivityNet")

    if use_timelens:
        ann_path = Path(data_root) / "annotations" / "sentence_temporal_grounding" / "activitynet-timelens.json"
    # else:
    #     ann_path = Path(data_root) / "annotations" / "sentence_temporal_grounding" / f"{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    data = json.load(open(ann_path))
    samples = []
    qid = 0
    for vid, meta in data.items():
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            candidate = Path(data_root) / "videos" / f"{vid}.{ext}"
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            raise FileNotFoundError(f"Video for {vid} not found under {data_root}/videos")

        for ts, sent in zip(meta["timestamps"], meta["sentences"]):
            samples.append(
                {
                    "qid": f"activitynet_{qid}",
                    "video": str(video_path),
                    "duration": meta["duration"],
                    "timestamp": ts,
                    "sentence": sent.strip(),
                }
            )
            qid += 1
    return samples


def load_qvhighlights(split: str, data_root: str, use_timelens: bool = True):
    """Load QVHighlights dataset for temporal grounding evaluation."""
    if use_timelens:
        ann_path = Path(data_root) / "qvhighlights-timelens.json"
    else:
        raise ValueError("QVHighlights currently only supports timelens format")
    
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    
    data = json.load(open(ann_path))
    samples = []
    qid = 0
    
    # Video root path
    video_root = Path(data_root) / "video_shards" / "qvhighlights" / "videos" / "qvhighlights"
    
    for vid_key, meta in data.items():
        # Video files are named using the full vid_key (e.g., "NUsG9BgSes0_210.0_360.0.mp4")
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            candidate = video_root / f"{vid_key}.{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            raise FileNotFoundError(f"Video for {vid_key} not found under {video_root}")
        
        # Process spans and queries
        spans = meta.get("spans", [])
        queries = meta.get("queries", [])
        duration = meta.get("duration", 0.0)
        
        # Match spans with queries (one-to-one correspondence)
        for idx, (span, query) in enumerate(zip(spans, queries)):
            if len(span) >= 2:
                timestamp = [float(span[0]), float(span[1])]
                samples.append(
                    {
                        "qid": f"qvhighlights_{qid}",
                        "video": str(video_path),
                        "duration": duration,
                        "timestamp": timestamp,
                        "sentence": query.strip() if isinstance(query, str) else str(query).strip(),
                    }
                )
                qid += 1
    
    return samples


def _strip_option_prefix(opt: str) -> str:
    match = re.match(r"[A-Z][\.\)]\s*(.*)", opt.strip())
    return match.group(1) if match else opt.strip()


def load_videomme(split: str, data_root: str):
    valid_split = {"default", "short", "medium", "long", "test"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for VideoMME")
    if split == "test":
        split = "default"

    data_path = Path(data_root) / "test-00000-of-00001.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"VideoMME annotations not found at {data_path}")

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if split != "default" and record.get("duration") != split:
                continue

            video_id = record.get("videoID") or record.get("video_id")
            video_path = Path(data_root) / "data" / f"{video_id}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video file missing: {video_path}")

            options = [_strip_option_prefix(opt) for opt in record.get("options", [])]
            answer_letter = (record.get("answer") or "").strip().upper()
            if not answer_letter:
                raise ValueError(f"Missing answer for {record.get('question_id')}")
            answer_idx = ord(answer_letter[0]) - ord("A")

            samples.append(
                {
                    "qid": f"videomme_{record['question_id']}",
                    "video": str(video_path),
                    "duration": record.get("duration"),
                    "question": record.get("question", ""),
                    "options": options,
                    "answer": answer_idx,
                    "task_type": record.get("task_type"),
                }
            )
    return samples


def load_mlvu(split: str, data_root: str, video_suffix: str = ""):
    if split == "test":
        split = "default"
    valid_split = {"default"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for MLVU")

    data_path = Path(data_root) / "json"
    video_dir = {
        "plotQA": Path(data_root) / f"video{video_suffix}/1_plotQA",
        "findNeedle": Path(data_root) / f"video{video_suffix}/2_needle",
        "ego": Path(data_root) / f"video{video_suffix}/3_ego",
        "count": Path(data_root) / f"video{video_suffix}/4_count",
        "order": Path(data_root) / f"video{video_suffix}/5_order",
        "anomaly_reco": Path(data_root) / f"video{video_suffix}/6_anomaly_reco",
        "topic_reasoning": Path(data_root) / f"video{video_suffix}/7_topic_reasoning",
        "subPlot": Path(data_root) / f"video{video_suffix}/8_sub_scene",
        "summary": Path(data_root) / f"video{video_suffix}/9_summary",
    }

    samples = []
    for file_name in os.listdir(data_path):
        data = json.load(open(data_path / file_name))
        for qid, itm in enumerate(data):
            video_name = itm["video"]
            task_type = itm["question_type"]
            video_path = video_dir[task_type] / video_name
            if "candidates" in itm:
                samples.append(
                    {
                        "qid": f"mlvu|{task_type}|{qid}",
                        "video": str(video_path),
                        "question": itm["question"],
                        "options": [opt.strip() for opt in itm["candidates"]],
                        "answer": itm["candidates"].index(itm["answer"]),
                        "duration": itm["duration"],
                        "task_type": itm["question_type"],
                    }
                )
    return samples


def load_tempcompass(split: str, data_root: str):
    """Load TempCompass dataset for multiple-choice question evaluation."""
    if split == "test":
        split = "default"
    valid_split = {"default"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for TempCompass")

    # Load multi-choice.json
    ann_path = Path(data_root) / "questions" / "multi-choice.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"TempCompass annotations not found at {ann_path}")

    data = json.load(open(ann_path))
    video_root = Path(data_root) / "videos"
    
    samples = []
    qid = 0
    
    for video_id, video_data in data.items():
        # Find video file
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            candidate = video_root / f"{video_id}.{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            # Try without extension in case video_id already includes extension
            candidate = video_root / video_id
            if candidate.exists():
                video_path = candidate
            else:
                raise FileNotFoundError(f"Video for {video_id} not found under {video_root}")
        
        # Process each category (e.g., "action", "direction")
        for task_type, questions in video_data.items():
            if not isinstance(questions, list):
                continue
            
            # Process each question in the category
            for question_item in questions:
                question_text = question_item.get("question", "")
                answer_text = question_item.get("answer", "")
                
                if not question_text or not answer_text:
                    continue
                
                # Parse question and options from question_text
                # Format: "Question text?\nA. option1\nB. option2\nC. option3"
                lines = question_text.strip().split("\n")
                if len(lines) < 2:
                    continue
                
                question = lines[0].strip()
                options = []
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        # Extract option text (remove "A. ", "B. ", etc.)
                        option_text = _strip_option_prefix(line)
                        options.append(option_text)
                
                if not options:
                    continue
                
                # Parse answer from answer_text (e.g., "A. dunking a basketball")
                answer_letter = answer_text.strip().upper()
                answer_match = re.match(r"([A-Z])[\.\)]", answer_letter)
                if answer_match:
                    answer_idx = ord(answer_match.group(1)) - ord("A")
                    # Validate answer index
                    if answer_idx < 0 or answer_idx >= len(options):
                        continue
                else:
                    # Try to find the answer by matching the text
                    answer_text_clean = _strip_option_prefix(answer_text)
                    answer_idx = None
                    for idx, opt in enumerate(options):
                        if opt.lower() == answer_text_clean.lower():
                            answer_idx = idx
                            break
                    if answer_idx is None:
                        continue
                
                samples.append(
                    {
                        "qid": f"tempcompass_{video_id}_{task_type}_{qid}",
                        "video": str(video_path),
                        "question": question,
                        "options": options,
                        "answer": answer_idx,
                        "duration": None,
                        "task_type": task_type,
                    }
                )
                qid += 1
    
    return samples


def load_longvideobench(split: str, data_root: str, video_suffix: str = ""):
    if split == "default":
        split = "test"
    split = "val"
    valid_split = {"val", "test"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for LongVideoBench")

    data_path = Path(data_root) / ("lvb_test_wo_gt.json" if split == "test" else "lvb_val.json")
    if not data_path.exists():
        raise FileNotFoundError(f"LongVideoBench annotations not found at {data_path}")

    duration_dict = {"15": "very short", "60": "short", "600": "medium", "3600": "long"}
    data = json.load(open(data_path, "r"))
    samples = []
    for itm in data:
        video_path = Path(data_root) / f"videos{video_suffix}" / itm["video_path"]
        samples.append(
            {
                "video": str(video_path),
                "question": itm["question"],
                "options": [opt.strip() for opt in itm["candidates"]],
                "answer": itm.get("correct_choice"),
                "duration": duration_dict.get(str(itm["duration_group"])),
                "task_type": itm.get("question_category"),
                "qid": f"longvideobench_{itm['id']}",
            }
        )
    return samples


def load_lvbench(split: str, data_root: str, video_suffix: str = ""):
    if split == "test":
        split = "default"
    valid_split = {"default"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for LVBench")

    meta_path = Path(data_root) / "data" / "video_info.meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"LVBench metadata not found at {meta_path}")

    samples = []
    with open(meta_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            itm = json.loads(line)
            video_path = Path(data_root) / f"all_videos{video_suffix}" / f"{itm['key']}.mp4"
            for qa in itm.get("qa", []):
                question, *options = qa["question"].split("\n")
                samples.append(
                    {
                        "video": str(video_path),
                        "question": question,
                        "options": [op for op in options],
                        "answer": ord(qa["answer"]) - ord("A"),
                        "duration": None,
                        "task_type": qa.get("question_type"),
                        "qid": f"lvbench_{qa['uid']}",
                    }
                )
    return samples


def load_mvbench(split: str, data_root: str):
    if split == "test":
        split = "default"
    valid_split = {"default"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for MVBench")

    data_path = Path(data_root) / "json"
    dataset_config = {
        "action_sequence": Path(data_root) / "video/star/Charades_v1_480/",
        "action_prediction": Path(data_root) / "video/star/Charades_v1_480/",
        "action_antonym": Path(data_root) / "video/ssv2_video/",
        "fine_grained_action": Path(data_root) / "video/Moments_in_Time_Raw/videos/",
        "unexpected_action": Path(data_root) / "video/FunQA_test/test/",
        "object_existence": Path(data_root) / "video/clevrer/video_validation/",
        "object_interaction": Path(data_root) / "video/star/Charades_v1_480/",
        "object_shuffle": Path(data_root) / "video/perception/videos/",
        "moving_direction": Path(data_root) / "video/clevrer/video_validation/",
        "action_localization": Path(data_root) / "video/sta/sta_video/",
        "scene_transition": Path(data_root) / "video/scene_qa/video/",
        "action_count": Path(data_root) / "video/perception/videos/",
        "moving_count": Path(data_root) / "video/clevrer/video_validation/",
        "moving_attribute": Path(data_root) / "video/clevrer/video_validation/",
        "state_change": Path(data_root) / "video/perception/videos/",
        "fine_grained_pose": Path(data_root) / "video/nturgbd/",
        "character_order": Path(data_root) / "video/perception/videos/",
        "egocentric_navigation": Path(data_root) / "video/vlnqa/",
        "episodic_reasoning": Path(data_root) / "video/tvqa/output_videos/",
        "counterfactual_inference": Path(data_root) / "video/clevrer/video_validation/",
    }

    samples = []
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".json"):
            continue
        data_type = file_name.split(".")[0]
        records = json.load(open(data_path / file_name))
        for qid, itm in enumerate(records):
            video_name = itm["video"]
            video_path = dataset_config[data_type] / video_name
            entry = {
                "video": str(video_path),
                "question": itm["question"],
                "options": [opt.strip() for opt in itm["candidates"]],
                "answer": itm["candidates"].index(itm["answer"]),
                "duration": None,
                "task_type": data_type,
                "qid": f"mvbench|{data_type}|{qid}",
            }
            if "start" in itm and "end" in itm:
                split_name = (
                    itm["video"].split(".mp4")[0]
                    + "_"
                    + str(itm["start"]).replace(".", "-")
                    + "_"
                    + str(itm["end"]).replace(".", "-")
                    + ".mp4"
                )
                entry["video"] = str(dataset_config[data_type] / "split" / split_name)
            else:
                if "start" in itm:
                    entry["video_start"] = itm["start"]
                if "end" in itm:
                    entry["video_end"] = itm["end"]
            samples.append(entry)
    return samples


def load_cgbench(split: str, data_root: str):
    if split == "test":
        split = "default"
    valid_split = {"default", "subset"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for CGBench")

    data_path = Path(data_root) / ("cgbench_mini.json" if split == "subset" else "cgbench.json")
    if not data_path.exists():
        raise FileNotFoundError(f"CGBench annotations not found at {data_path}")

    samples = []
    data = json.load(open(data_path, "r"))
    for itm in data:
        video_path = Path(data_root) / "cg_videos_720p" / f"{itm['video_uid']}.mp4"
        samples.append(
            {
                "video": str(video_path),
                "question": itm["question"],
                "options": [opt.strip() for opt in itm["choices"]],
                "answer": ord(itm["right_answer"]) - ord("A"),
                "duration": itm.get("duration"),
                "task_type": itm.get("sub_category"),
                "qid": f"cgbench|{itm['qid']}",
            }
        )
    return samples


def load_auroracap(split: str, data_root: str):
    valid_split = {"default", "background", "camera", "detailed", "main_object", "short"}
    if split not in valid_split:
        raise ValueError(f"split must be one of {sorted(valid_split)} for AuroraCap")

    data_path = Path(data_root) / "VDC_1k.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"AuroraCap annotations not found at {data_path}")

    tasks = [
        "background",
        "camera",
        "detailed",
        "main_object",
        "short",
    ] if split == "default" else [split]

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            itm = json.loads(line)
            video_path = Path(data_root) / "videos/videos" / itm["video_name"]
            for task in tasks:
                samples.append(
                    {
                        "video": str(video_path),
                        "answer": itm[f"{task}_caption"],
                        "qid": f"auroracap|{task}|{itm['video_id']}",
                        "task_type": task,
                    }
                )
    return samples


def zigzag_split(data: List[dict], curr_idx: int, total_idx: int) -> List[dict]:
    """Zigzag shard split: pick two mirrored chunks for better coverage."""
    n = len(data)
    if n == 0:
        return []

    parts = 2 * total_idx
    chunk_size = (n + parts - 1) // parts

    selected = []
    for idx in (curr_idx, parts - 1 - curr_idx):
        start = idx * chunk_size
        end = min(n, (idx + 1) * chunk_size)
        if start < end:
            selected.extend(data[start:end])
    return selected


def load_finished_tg(out_path: Path):
    if not out_path.exists():
        return set(), 0.0, 0, 0, 0, 0

    finished = set()
    total_iou = 0.0
    count = 0
    over_03 = 0
    over_05 = 0
    over_07 = 0

    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("qid")
            if qid is None:
                continue
            finished.add(qid)

            iou_val = record.get("iou")
            if isinstance(iou_val, (int, float)):
                total_iou += float(iou_val)
                count += 1
                over_03 += iou_val >= 0.3
                over_05 += iou_val >= 0.5
                over_07 += iou_val >= 0.7

    return finished, total_iou, count, over_03, over_05, over_07


def load_finished_mcq(out_path: Path):
    if not out_path.exists():
        return set(), 0, 0

    finished = set()
    total_correct = 0
    count = 0

    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("qid")
            if qid is None:
                continue
            finished.add(qid)

            pred = record.get("pred")
            target = record.get("target")
            if isinstance(record.get("correct"), int):
                total_correct += int(record["correct"])
                count += 1
            elif pred is not None and target is not None:
                total_correct += int(pred == target)
                count += 1

    return finished, total_correct, count


def load_finished_caption(out_path: Path):
    if not out_path.exists():
        return set(), 0

    finished = set()
    count = 0
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("qid")
            if qid is None:
                continue
            finished.add(qid)
            count += 1
    return finished, count


def format_prompt(sentence: str, duration: float, include_duration: bool = True, is_pretrained_model: bool=False) -> str:
    # prompt = (
    #     "You will watch a video. Find when the described textual query happens and "
    #     "return only the start and end time in seconds as 'start to end'. "
    #     "If unsure, make your best guess within the video duration.\n"
    #     f"Query: {sentence}"
    # )
    if is_pretrained_model == True:
        prompt = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event. Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time", For example: "12.54 to 17.83"."""
    else:
        prompt = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event. Output your thought process within the <think> </think> tags.
Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

    prompt = prompt.replace("[EVENT]", sentence)
    if include_duration:
        prompt += f"\nVideo duration: {duration:.2f} seconds."
    return prompt


def format_mcq_prompt(question: str, options: List[str]) -> str:
    labeled = [f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options)]
    option_block = "\n".join(labeled)
    return (
        "You will watch a video and answer a multiple-choice question. "
        "Select the single best option and respond with only the option letter.\n"
        f"Question: {question}\n"
        f"Options:\n{option_block}\n"
        "Answer:"
    )


def format_caption_prompt(task_type: Optional[str] = None) -> str:
    focus = task_type.replace("_", " ") if task_type else "detailed"
    return (
        "You will watch a video and provide a concise caption. "
        f"Focus on {focus} aspects of the content and keep the caption informative."
    )


def extract_pred_timestamps(text: str):
    """Parse start/end timestamps from a variety of response formats."""

    time_re = r"(\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?|\d{1,2}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?)"

    def _normalize_min_sec(segment: str) -> str:
        # Convert tokens like 1min23s into 1:23 to reuse the core patterns.
        def repl(match: re.Match) -> str:
            minutes = match.group(1)
            seconds = match.group(2)
            return f"{minutes}:{seconds}"

        return re.sub(r"(\d+)\s*min\s*(\d+(?:\.\d+)?)s?", repl, segment, flags=re.IGNORECASE)

    def _parse_time(val: str):
        clean = val.strip().lower()
        clean = re.sub(r"[^0-9:.,s]", "", clean)
        clean = clean.replace(",", ".")
        if clean.endswith("s"):
            clean = clean[:-1]

        if ":" in clean:
            parts = clean.split(":")
            try:
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + float(s)
                if len(parts) == 2:
                    m, s = parts
                    return int(m) * 60 + float(s)
            except ValueError:
                return None

        try:
            return float(clean)
        except ValueError:
            return None

    def _gather_candidates(segment: str):
        candidates = []
        patterns = [
            rf"{time_re}\s*(?:to|\-|–|—|~)\s*{time_re}",
            rf"start\s*[:=]?\s*{time_re}.*?end\s*[:=]?\s*{time_re}",
            rf"\[{time_re}\s*,\s*{time_re}\]",
        ]
        for pat in patterns:
            for match in re.findall(pat, segment, flags=re.IGNORECASE | re.DOTALL):
                if isinstance(match, tuple) or isinstance(match, list):
                    candidates.append(match[-2:])
                else:
                    # Flatten simple 2-group matches
                    parts = re.findall(time_re, match)
                    if len(parts) >= 2:
                        candidates.append(parts[:2])
        return candidates

    search_segments = [_normalize_min_sec(text)]
    bracket = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if bracket:
        search_segments.insert(0, _normalize_min_sec(bracket.group(1)))

    found = []
    for segment in search_segments:
        found.extend(_gather_candidates(segment))

    for raw_start, raw_end in reversed(found):
        start = _parse_time(raw_start)
        end = _parse_time(raw_end)
        if start is None or end is None:
            continue
        return [float(start), float(end)]

    return None


def extract_mcq_answer(text: str, num_options: int):
    """Extract a zero-based option index from model output."""

    if not isinstance(text, str) or num_options <= 0:
        return None

    bracket = re.search(r"<answer>\s*([A-Z])\s*</answer>", text, flags=re.IGNORECASE)
    if bracket:
        letter = bracket.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < num_options:
            return idx

    cleaned = text.upper()
    tokens = re.split(r"[^A-Z]+", cleaned)
    for tok in tokens:
        if len(tok) == 1:
            idx = ord(tok) - ord("A")
            if 0 <= idx < num_options:
                return idx
    return None


def iou(pred, gt):
    if pred is None or None in pred:
        return 0.0
    ps, pe = pred
    gs, ge = gt
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0:
        return 0.0
    return inter / union


def main():
    args = parse_args()
    t_wall_start = time.perf_counter()

    # Apply timestamp formatting monkey patch before building the processor.
    # ts_prompt (if provided) has highest precedence; ts_format kept for backward compatibility.
    # apply_timestamp_format_patch(ts_prompt=args.ts_prompt, format_style=args.ts_format)

    if args.total_idx <= 0:
        raise ValueError("total_idx must be positive")
    if not (0 <= args.curr_idx < args.total_idx):
        raise ValueError("curr_idx must satisfy 0 <= curr_idx < total_idx")

    dataset_type = get_dataset_type(args.dataset)
    default_roots = {
        "charades": "./dataset/charades",
        "activitynet": "./dataset/activitynet",
        "qvhighlights": "TimeLens-Bench",
        "videomme": "./dataset/videomme",
        "mlvu": "./dataset/mlvu/MLVU",
        "tempcompass": "DATASET/TempCompass",
        "longvideobench": "./dataset/longvideobench",
        "lvbench": "./dataset/lvbench",
        "mvbench": "./dataset/mvbench",
        "cgbench": "./dataset/cgbench",
        "auroracap": "./dataset/auroracap",
    }
    data_root = args.data_root or default_roots[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.model_path)

    # vLLM requires this to avoid fork-related issues
    if args.use_vllm:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    processor = Qwen3VLProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    if args.use_vllm:
        if args.tensor_parallel_size is not None:
            tp = args.tensor_parallel_size
        else:
            tp = max(1, torch.cuda.device_count())
        model = LLM(
            model=args.model_path,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=tp,
            seed=0,
        )
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )

    if args.dataset == "charades":
        samples = load_charades(args.split, data_root)
    elif args.dataset == "activitynet":
        samples = load_activitynet(args.split, data_root)
    elif args.dataset == "qvhighlights":
        samples = load_qvhighlights(args.split, data_root)
    elif args.dataset == "videomme":
        samples = load_videomme(args.split, data_root)
    elif args.dataset == "mlvu":
        samples = load_mlvu(args.split, data_root)
    elif args.dataset == "tempcompass":
        samples = load_tempcompass(args.split, data_root)
    elif args.dataset == "longvideobench":
        samples = load_longvideobench(args.split, data_root)
    elif args.dataset == "lvbench":
        samples = load_lvbench(args.split, data_root)
    elif args.dataset == "mvbench":
        samples = load_mvbench(args.split, data_root)
    elif args.dataset == "cgbench":
        samples = load_cgbench(args.split, data_root)
    elif args.dataset == "auroracap":
        samples = load_auroracap(args.split, data_root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.dataset == "videomme":
        samples = zigzag_split(samples, args.curr_idx, args.total_idx)
    else:
        samples = [
            s for i, s in enumerate(samples) if i % args.total_idx == args.curr_idx
        ]

    if args.max_samples:
        samples = samples[: args.max_samples]

    finished = set()
    prompt_dumped = False
    writer = None

    if dataset_type == "tg":
        total_iou = 0.0
        count = 0
        over_03 = 0
        over_05 = 0
        over_07 = 0
    elif dataset_type == "mcq":
        total_correct = 0
        count = 0
    else:  # caption
        count = 0

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_type == "tg":
            (
                finished,
                total_iou,
                count,
                over_03,
                over_05,
                over_07,
            ) = load_finished_tg(out_path)
        elif dataset_type == "mcq":
            finished, total_correct, count = load_finished_mcq(out_path)
        else:
            finished, count = load_finished_caption(out_path)
        if finished:
            print(f"Resuming from {out_path}, found {len(finished)} finished samples.")
        samples = [s for s in samples if s["qid"] not in finished]
        print(f"Skipping {len(finished)} finished samples, {len(samples)} remaining.")
        writer = open(out_path, "a")
    is_pretrained_model = False

    if args.nothink == "True":      
        is_pretrained_model = True
    print("****"*60, "nothink is: ", args.nothink, is_pretrained_model)
    t_infer_start = time.perf_counter()
    for start in tqdm(range(0, len(samples), args.batch_size), desc=f"{args.dataset} eval"):
        batch = samples[start : start + args.batch_size]
        messages = []
        for sample in batch:
            if dataset_type == "tg":
                print('--'*1000 +"include_duration is: ", args.include_duration)
                prompt = format_prompt(sample["sentence"], sample["duration"], include_duration=args.include_duration, is_pretrained_model=is_pretrained_model)
            elif dataset_type == "mcq":
                prompt = format_mcq_prompt(sample["question"], sample["options"])
            else:
                prompt = format_caption_prompt(sample.get("task_type"))
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": sample["video"],
                                "min_pixels": 4 * 32 * 32,
                                "max_pixels": 768 * 32 * 32,
                                "total_pixels": args.total_tokens * 32 * 32,
                                "fps": args.fps,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
        if (not prompt_dumped) and args.output_jsonl:
            # Diagnostic only: re-decodes the first video through the HF processor
            # (pyav backend), which can fail with EAGAIN once vLLM has saturated
            # the thread budget. Never let it abort the actual eval.
            try:
                dump_inputs = processor.apply_chat_template(
                    messages[:1],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    fps=args.fps,
                )
                decoded = processor.tokenizer.decode(dump_inputs["input_ids"][0])
                out_dir = Path(args.output_jsonl).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "first_sample_prompt.txt").write_text(decoded)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] skipped first_sample_prompt dump: {exc}")
            prompt_dumped = True
        if args.use_vllm:
            vllm_inputs = [prepare_inputs_for_vllm(msg, processor) for msg in messages]
            outputs = model.generate(vllm_inputs, sampling_params=sampling_params)
            output_texts = [out.outputs[0].text for out in outputs]
        else:
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # process_vision_info will resize frames; avoid double-resize in processor below
            images, videos, video_kwargs = process_vision_info(
                messages, 
                image_patch_size=processor.image_processor.patch_size, 
                return_video_kwargs=True, 
                return_video_metadata=True
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            encoded = processor(
                text=chat_text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )
            inputs = encoded.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_texts = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        for sample, output_text in zip(batch, output_texts):
            if dataset_type == "tg":
                pred_ts = extract_pred_timestamps(output_text)
                gt_ts = sample["timestamp"]
                sample_iou = iou(pred_ts, gt_ts)

                total_iou += sample_iou
                count += 1
                over_03 += sample_iou >= 0.3
                over_05 += sample_iou >= 0.5
                over_07 += sample_iou >= 0.7

                if writer:
                    writer.write(
                        json.dumps(
                            {
                                "qid": sample["qid"],
                                "video": sample["video"],
                                "target": gt_ts,
                                "pred": pred_ts,
                                "iou": sample_iou,
                                "output_text": output_text,
                            }
                        )
                        + "\n"
                    )
            elif dataset_type == "mcq":
                pred_idx = extract_mcq_answer(output_text, len(sample["options"]))
                target_idx = sample["answer"]
                correct = int(pred_idx == target_idx) if pred_idx is not None else 0

                total_correct += correct
                count += 1

                if writer:
                    writer.write(
                        json.dumps(
                            {
                                "qid": sample["qid"],
                                "video": sample["video"],
                                "target": target_idx,
                                "pred": pred_idx,
                                "correct": correct,
                                "task_type": sample.get("task_type"),
                                "duration": sample.get("duration"),
                                "output_text": output_text,
                            }
                        )
                        + "\n"
                    )
            else:  # caption
                count += 1
                if writer:
                    writer.write(
                        json.dumps(
                            {
                                "qid": sample["qid"],
                                "video": sample["video"],
                                "target": sample.get("answer"),
                                "task_type": sample.get("task_type"),
                                "duration": sample.get("duration"),
                                "output_text": output_text,
                            }
                        )
                        + "\n"
                    )

    if writer:
        writer.close()

    infer_seconds = time.perf_counter() - t_infer_start
    total_seconds = time.perf_counter() - t_wall_start

    if dataset_type == "tg":
        avg_iou = total_iou / max(count, 1)
        print("===== Temporal Grounding =====")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {count}")
        print(f"mIoU: {avg_iou * 100:.2f}")
        print(
            f"R@0.3: {over_03 / max(count, 1) * 100:.2f} | "
            f"R@0.5: {over_05 / max(count, 1) * 100:.2f} | "
            f"R@0.7: {over_07 / max(count, 1) * 100:.2f}"
        )
    elif dataset_type == "mcq":
        acc = total_correct / max(count, 1) * 100
        print("===== Video QA (MCQ) =====")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {count}")
        print(f"Accuracy: {acc:.2f}")
    else:
        print("===== Video Captioning =====")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {count}")

    print("===== Timing =====")
    print(
        f"Total evaluation wall time: {total_seconds:.2f} s "
        f"({total_seconds / 60:.2f} min)"
    )
    print(f"Inference loop time: {infer_seconds:.2f} s ({infer_seconds / 60:.2f} min)")

    timing_path: Optional[Path] = None
    if args.timing_json:
        timing_path = Path(args.timing_json)
    elif args.output_jsonl:
        p = Path(args.output_jsonl)
        timing_path = p.parent / f"{p.stem}_timing.json"

    if timing_path is not None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_record = {
            "dataset": args.dataset,
            "split": args.split,
            "curr_idx": args.curr_idx,
            "total_idx": args.total_idx,
            "sample_count": count,
            "infer_seconds": round(infer_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "infer_minutes": round(infer_seconds / 60.0, 6),
            "total_minutes": round(total_seconds / 60.0, 6),
        }
        timing_path.write_text(json.dumps(timing_record, indent=2, ensure_ascii=False) + "\n")
        print(f"Timing written to {timing_path}")


if __name__ == "__main__":
    main()
