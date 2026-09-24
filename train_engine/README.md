<div align="center">
  <h1> Video-OPD: Efficient Post-Training of MLLMs for Temporal Video Grounding via On-Policy Distillation </h1>

  <h3>🏆 ICML 2026</h3>

  <br>

  <a href="https://arxiv.org/pdf/2602.02994">
    <img src="https://img.shields.io/badge/arXiv-2602.02994-b31b1b.svg?style=flat" alt="Paper">
  </a>
  <a href="https://huggingface.co/williamljz/Video-opd">
    <img src="https://img.shields.io/badge/🤗 Hugging Face-Model-FFD21E.svg?style=flat" alt="Model">
  </a>
  <a href="https://huggingface.co/williamljz/Video-opd-teacher-Qwen3-VL-32B-GRPO">
    <img src="https://img.shields.io/badge/🤗 Hugging Face-Teacher Model (32B)-FFD21E.svg?style=flat" alt="Teacher Model">
  </a>
  <a href="https://huggingface.co/datasets/williamljz/Video-opd-Dataset">
    <img src="https://img.shields.io/badge/🤗 Hugging Face-Dataset-FFD21E.svg?style=flat" alt="Dataset">
  </a>
</div>

<br>

<p align="center">
Jiaze Li<sup>1*†</sup>, Hao Yin<sup>1*</sup>, Haoran Xu<sup>2*</sup>, Boshen Xu<sup>3</sup>, Wenhui Tan<sup>3</sup>, Zewen He<sup>1</sup>, Jianzhong Ju<sup>1‡</sup>, Zhenbo Luo<sup>1</sup>, Jian Luan<sup>1</sup>
<br>
<sup>1</sup> MiLM Plus, Xiaomi Inc. &nbsp; <sup>2</sup> Zhejiang University &nbsp; <sup>3</sup> Renmin University of China
<br>
<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Project leader &nbsp; <sup>‡</sup> Corresponding author
</p>

---

## Abstract

Reinforcement learning has emerged as a principled post-training paradigm for Temporal Video Grounding (TVG) due to its on-policy optimization, yet existing GRPO-based methods remain fundamentally constrained by sparse reward signals and substantial computational overhead. We propose **Video-OPD**, an efficient post-training framework for TVG inspired by recent advances in on-policy distillation. Video-OPD optimizes trajectories sampled directly from the current policy, thereby preserving alignment between training and inference distributions, while a frontier teacher supplies dense, token-level supervision via a reverse KL divergence objective. This formulation preserves the on-policy property critical for mitigating distributional shift, while converting sparse, episode-level feedback into fine-grained, step-wise learning signals. Building on Video-OPD, we introduce **Teacher-Validated Disagreement Focusing (TVDF)**, a lightweight training curriculum that iteratively prioritizes trajectories that are both teacher-reliable and maximally informative for the student, thereby improving training efficiency. Empirical results demonstrate that Video-OPD consistently outperforms GRPO while achieving substantially faster convergence and lower computational cost, establishing on-policy distillation as an effective alternative to conventional reinforcement learning for TVG.

## News

- **[2026/06]** Code, model weights, and training dataset are released!
- **[2026/05]** Video-OPD is accepted to **ICML 2026**!
- **[2026/02]** Paper is available on [arXiv](https://arxiv.org/pdf/2602.02994).

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/SeerRay-Lab/Video-OPD.git
cd Video-OPD

# Install dependencies
pip install -e .
pip install deepspeed==0.17.4
pip install qwen_vl_utils==0.0.14
pip install flash_attn==2.7.4.post1
```

### Training

#### 1. Download Training Data

Download the training dataset from [Hugging Face](https://huggingface.co/datasets/williamljz/Video-opd-Dataset):

```bash
huggingface-cli download williamljz/Video-opd-Dataset --repo-type dataset --local-dir ./data/Video-opd-Dataset
```

#### 2. Construct Training Data

Convert raw data into the ms-swift training format:

```bash
python scripts/datasets/construct_trainingdataset_format.py \
    --json_path <path_to_source_json> \
    --output_path <path_to_output_json> \
    --video_root_dir <path_to_video_directory>
```

#### 3. Launch Training

Training uses on-policy distillation with 8 GPUs on a single node. We use [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) as the base model and [Video-OPD Teacher (Qwen3-VL-32B-GRPO)](https://huggingface.co/williamljz/Video-opd-teacher-Qwen3-VL-32B-GRPO) as the teacher model.

```bash
bash scripts/train/train.sh
```

### Evaluation

We provide evaluation scripts for temporal video grounding benchmarks. Example usage for evaluating on QVHighlights:

```bash
bash eval/eval.sh
```


## Model Weights

Pre-trained model weights are available on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| Video-OPD | Final student model | [williamljz/Video-opd](https://huggingface.co/williamljz/Video-opd) |
| Video-OPD Teacher | GRPO-trained Qwen3-VL-32B teacher | [williamljz/Video-opd-teacher-Qwen3-VL-32B-GRPO](https://huggingface.co/williamljz/Video-opd-teacher-Qwen3-VL-32B-GRPO) |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2026video,
  title={Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation},
  author={Li, Jiaze and Yin, Hao and Xu, Haoran and Xu, Boshen and Tan, Wenhui and He, Zewen and Ju, Jianzhong and Luo, Zhenbo and Luan, Jian},
  journal={arXiv preprint arXiv:2602.02994},
  year={2026}
}
```

## Acknowledgements

Our training framework is built upon [ms-swift](https://github.com/modelscope/ms-swift), an efficient and lightweight framework for fine-tuning large language models. We thank the ms-swift team for their excellent open-source contribution.

## License

This project is released under the [Apache 2.0 License](./LICENSE).
