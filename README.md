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

# 💡 Overview

Natural co-speech gestures are not merely reactive responses to the current speech signal.
Humans often organize upcoming gestures before their salient realization, involving
preparation, semantic activation, and coordinated evolution across multiple body regions.

Most existing speech-driven gesture generation methods directly map speech features to
low-level motion sequences, leaving high-level gesture organization and detailed motion
realization entangled.

**PlanTalk** introduces an explicit intermediate **future motion plan** to model upcoming
gesture organization before detailed motion generation.

The framework follows a

**Speech → Future Motion Plan → Holistic Motion**

generation paradigm.

Given speech context, PlanTalk first predicts a discrete future-oriented motion plan that
captures the organization of an upcoming motion window. The predicted plan is subsequently
used to regulate **when** semantic motion should be activated and **how** the planned motion
should be realized across different body regions.

The framework consists of three key components:

1. **Future Motion Planning**
   A discrete motion-plan space is learned from future motion windows. At inference time,
   the corresponding future plan is predicted using speech alone.

2. **Event-Aware Semantic Gating**
   The predicted plan cooperates with speech semantics to determine when semantically
   meaningful motion should be emphasized, reducing delayed or spurious gesture activation.

3. **Hierarchical Plan-Conditioned Modulation**
   The predicted future plan modulates motion generation across different anatomical
   regions, enabling coordinated realization of upper-body, hand, and lower-body motion.

<!-- Replace with your framework figure -->

<p align="center">
<img src="assets/framework.png" width="100%">
</p>

# 🧠 Future-Grounded Motion Planning

Instead of representing only the current motion state, PlanTalk explicitly summarizes
an upcoming motion window

$$
\mathcal{W}_{t}^{+}
$$

into a discrete motion-plan token.

During training, future motion provides supervision for learning the motion-plan space.
At inference time, future motion is unavailable; therefore, a speech-to-plan predictor
estimates the upcoming motion plan directly from speech context.

This design encourages the intermediate representation to capture higher-level information
such as:

* upcoming gesture activation,
* motion intensity and evolution,
* body-part participation,
* preparation-to-realization structure,
* cross-part motion organization.

The predicted plan subsequently provides structured guidance for detailed motion synthesis.

# 📊 Results

## BEAT2 — Speaker 2 Protocol

| Method         |     FGD ↓ |      BC ↑ |     DIV ↑ |    SRGR ↑ | MSE ↓ |     LVD ↓ |
| -------------- | --------: | --------: | --------: | --------: | ----: | --------: |
| EMAGE          |     5.512 |     0.772 |     13.06 |     0.323 | 7.680 |     7.556 |
| ProbTalk       |     5.127 |     0.771 |     13.27 |     0.300 | 8.617 |     7.634 |
| MambaTalk      |     5.366 |     0.781 |     13.05 |     0.290 | 6.289 |     6.897 |
| RAG-Gesture    |     8.082 |     0.734 |     11.97 |     0.390 | 7.248 |     6.947 |
| SemTalk        |     4.278 |     0.777 |     12.91 |     0.430 | 6.153 |     6.938 |
| **PlanTalk** | **3.812** | **0.817** | **13.95** | **0.442** |     — | **6.462** |

PlanTalk achieves improved motion quality and semantic relevance while maintaining strong
gesture diversity and speech-motion synchronization. In particular, the improvements in
FGD and SRGR indicate that explicitly reasoning about upcoming gesture organization benefits
both holistic motion generation and semantically meaningful gesture realization.

> **Note:** The table will be updated with the complete results and all evaluated baselines
> after the official release.

# 🔬 Planning Analysis

To investigate whether the proposed representation behaves as a meaningful future motion
plan rather than merely an additional latent variable, we evaluate several alternative
planning conditions.

| Planning Condition        |     FGD ↓ |    SRGR ↑ | Event F1 ↑ |   Acc@1 ↑ |   Acc@5 ↑ |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: |
| No Plan                   |     4.347 |     0.424 |      0.668 |         – |         – |
| Random Plan               |     4.412 |     0.418 |      0.641 |         – |         – |
| Current Motion Token      |     4.286 |     0.429 |      0.706 |     0.462 |     0.731 |
| **Predicted Future Plan** | **3.812** | **0.442** |  **0.784** | **0.536** | **0.798** |
| Oracle Future Plan        |     3.615 |     0.457 |      0.836 |         – |         – |

Compared with current-state motion tokens, future-grounded plans provide more informative
guidance for semantic event prediction and motion generation. The oracle-plan result further
reveals the potential performance gain obtainable from improved speech-to-plan prediction.

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
