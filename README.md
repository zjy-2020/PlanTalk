<div align="center">

<h2>PlanTalk: Future-Grounded Motion Planning for Co-Speech Gesture Generation</h2>

<p>
<strong>Official implementation of PlanTalk, a future-grounded speech-to-plan-to-motion framework for holistic co-speech gesture generation.</strong>
</p>

<!-- Replace the links below after release -->

<a href="static/pdfs/TMM_20260903.pdf"><img src="https://img.shields.io/badge/Paper-red"></a>
<a href="https://zjy-2020.github.io/PlanTalk/"><img src="https://img.shields.io/badge/Project-purple"></a>
<a href="#"><img src="https://img.shields.io/badge/Model_Weights-Coming_Soon-yellow"></a>


<!-- Replace with your teaser figure -->

<img src="static/teaser.png" alt="PlanTalk teaser" style="width:100%;">

</div>

# 📣 Updates

* **[2026-09]** Release training and inference code for PlanTalk.
* **[Coming Soon]** Release pretrained future motion planner and PlanTalk checkpoints.
* **[Coming Soon]** Release evaluation scripts and qualitative examples on BEAT2.


# ⚡ Quick Start

## 1. Create Environment

We recommend using a dedicated Conda environment.

```bash
conda create -n PlanTalk python=3.8 -y
conda activate PlanTalk
```

Install PyTorch and the required dependencies:

```bash
pip install -r requirements.txt
```

Please ensure that the CUDA and PyTorch versions are compatible with your local environment.

# 📦 Dataset

## BEAT2

PlanTalk is primarily evaluated on the **BEAT2** dataset using SMPL-X motion
representations.

Please download BEAT2 following the official dataset instructions and organize it as:

```text
PlanTalk/
├── BEAT2/
│   └── beat_english_v2.0.0/
├── configs/
├── dataloaders/
├── models/
├── weights/
├── utils/
└── train.py
```

The original dataset is not redistributed in this repository.

# 📥 Pretrained Models

PlanTalk requires several pretrained components, including:

* motion representation models,
* speech feature encoder,
* future motion-plan representation model,
* speech-to-plan predictor,
* holistic motion generator.

After downloading the released checkpoints, organize the weights as:

```text
weights/
├── pretrained_vq/
├── future_plan/
│   └── PlanTalk_future_plan_vq.bin
├── speech_encoder/
├── base_motion/
└── PlanTalk/
    └── PlanTalk.bin
```

Pretrained checkpoints will be released after publication.

# 🚀 Training

PlanTalk follows a staged training strategy.

## Stage 1: Learn the Future Motion-Plan Space

The first stage learns a discrete representation from future motion windows.

Conceptually,

```text
Future Motion Window
        ↓
Motion Plan Encoder
        ↓
Vector Quantization
        ↓
Discrete Future Plan
```

The learned codebook provides a compact representation of upcoming motion organization.

Example:

```bash
python train.py \
    --config configs/PlanTalk_future_plan.yaml \
    --train_plan
```

After training, save the best future-plan representation model under:

```text
weights/future_plan/
```

## Stage 2: Train Speech-to-Plan Prediction

Given speech context, the planner predicts the discrete motion plan corresponding to the
upcoming motion window.

```text
Speech Context
      ↓
Speech Encoder
      ↓
Planner
      ↓
Predicted Future Plan
```

Example:

```bash
python train.py \
    --config configs/PlanTalk.yaml \
    --train_planner \
    --plan_vq_ckpt weights/future_plan/PlanTalk_future_plan_vq.bin
```

## Stage 3: Train PlanTalk Motion Generation

The predicted future plan is used by both the event-aware semantic gate and hierarchical
plan-conditioned modulation modules.

```text
                    ┌──────────────────────┐
Speech ────────────→│ Speech Representation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Future Plan Predictor│
                    └──────────┬───────────┘
                               │
                     Predicted Future Plan
                         ┌─────┴─────┐
                         ▼           ▼
                 Event-Aware     Plan-Conditioned
                 Semantic Gate     Modulation
                         \           /
                          \         /
                           ▼       ▼
                         Holistic Motion
```

Example training command:

```bash
python train.py \
    --config configs/PlanTalk.yaml \
    --plan_vq_ckpt weights/future_plan/PlanTalk_future_plan_vq.bin
```

# 🧪 Testing

To evaluate a trained PlanTalk model:

```bash
python train.py \
    --config configs/PlanTalk.yaml \
    --test_state \
    --load_ckpt weights/PlanTalk/PlanTalk_best.bin \
    --plan_vq_ckpt weights/future_plan/PlanTalk_future_plan_vq.bin
```

The evaluation includes metrics for:

* motion distribution quality,
* gesture diversity,
* speech-motion synchronization,
* semantic relevance,
* motion reconstruction,
* semantic event prediction,
* future-plan prediction.

Representative metrics include:

```text
FGD
BC
DIV
SRGR
MSE
LVD
Event F1
Plan Acc@1
Plan Acc@5
```

# 🎙️ Inference

At inference time, PlanTalk requires **speech only**.

No future motion or oracle motion-plan information is used.

The inference pipeline is:

```text
Speech
  ↓
Speech Representation
  ↓
Future Plan Prediction
  ↓
Event-Aware Semantic Gating
  +
Plan-Conditioned Motion Generation
  ↓
SMPL-X Motion
```

Example:

```bash
python train.py \
    --config configs/PlanTalk.yaml \
    --inference \
    --audio_infer_path demo/example.wav \
    --load_ckpt weights/PlanTalk/PlanTalk_best.bin \
    --plan_vq_ckpt weights/future_plan/PlanTalk_future_plan_vq.bin
```


# 🗂️ Repository Layout

The main repository structure is expected to be:

```text
PlanTalk/
├── BEAT2/
├── configs/
│   ├── PlanTalk.yaml
│   └── PlanTalk_future_plan.yaml
├── dataloaders/
├── datasets/
├── models/
│   ├── motion representations
│   ├── future motion planner
│   ├── event-aware semantic gate
│   └── PlanTalk motion generator
├── utils/
├── weights/
│   ├── pretrained_vq/
│   ├── future_plan/
│   └── PlanTalk/
├── demo/
├── train.py
└── README.md
```

# 🎨 Visualization

Generated motions are represented using SMPL-X parameters.

We recommend using the official SMPL-X model together with Blender or the BEAT2
visualization tools to render generated motions.

The repository will also provide scripts for generating qualitative comparison videos.



# 📄 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{zhang2026planttalk,
  title   = {PlanTalk: Future-Grounded Motion Planning for Co-Speech Gesture Generation},
  author  = {Jiye Zhang, Guibiao Liao, Dingwei Liu, Gaolin Yang, Xiuhua Jiang, and Jiangbo Xu},
  year    = {2026}
}
```

> The BibTeX entry will be updated after publication.

# 🙏 Acknowledgments

Our implementation builds upon several excellent open-source projects and prior work in
holistic co-speech gesture generation.

We sincerely thank the authors of **BEAT2**, **EMAGE**, **SemTalk**, and other related
projects for releasing their datasets, models, and code.

# 📜 License

Please follow the licenses of the corresponding datasets, SMPL-X models, pretrained
speech models, and third-party repositories.

The license for the PlanTalk source code will be provided upon release.
