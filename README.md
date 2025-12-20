<div align="center">

# SAM-Audio

![CI](https://github.com/facebookresearch/sam-audio/actions/workflows/ci.yaml/badge.svg)

![model_image](assets/sam_audio_main_model.png)

</div>

Segment Anything Model for Audio [[**Blog**](https://ai.meta.com/blog/sam-audio/)] [[**Paper**](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)] [[**Demo**](https://aidemos.meta.com/segment-anything/editor/segment-audio)]

SAM-Audio is a foundation model for isolating any sound in audio using text, visual, or temporal prompts. It can separate specific sounds from complex audio mixtures based on natural language descriptions, visual cues from video, or time spans.

SAM-Audio and the Judge model crucially rely on [Perception-Encoder Audio-Visual (PE-AV)](https://huggingface.co/facebook/pe-av-large), which you can read more about [here](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)

## Setup

**Requirements:**
- Python >= 3.11
- CUDA-compatible GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

Install dependencies using uv:

```bash
uv sync
```

For development (includes Jupyter notebooks):

```bash
uv sync --group dev
```

## Quick Start: CLI

The easiest way to use SAM-Audio is via the command-line interface, which automatically detects your GPU and selects the appropriate model size.

⚠️ Before using SAM Audio, please request access to the checkpoints on the SAM Audio
Hugging Face [repo](https://huggingface.co/facebook/sam-audio-large). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `huggingface-cli login` after generating an access token.)

### Basic Usage

```bash
# Separate a sound described by text
uv run sam-audio --input audio.wav --description "guitar solo" --verbose

# The CLI will:
# - Detect your GPU and available VRAM
# - Select the best model (small/base/large) for your hardware
# - Use span prediction for better results
# - Output target.wav and residual.wav
```

### GPU Auto-Detection

The CLI dynamically estimates VRAM requirements based on model parameter counts and selects the largest model that fits in your available memory:

| Model | Estimated VRAM | Parameters |
|-------|----------------|------------|
| `sam-audio-large` | ~6GB | ~1.2B |
| `sam-audio-base` | ~4.5GB | ~600M |
| `sam-audio-small` | ~3.5GB | ~300M |

A 10% safety margin is applied to avoid OOM errors. Reranking candidates are also adjusted automatically.

### CLI Options

```bash
uv run sam-audio --help

# Key options:
#   -i, --input         Input audio file (required)
#   -d, --description   Sound to isolate (required)
#   -o, --output        Output path for target audio
#   --residual          Output path for residual audio
#   -m, --model         Override auto-selected model
#   -c, --candidates    Override reranking candidates
#   --no-predict-spans  Disable span prediction
#   -v, --verbose       Show detailed progress
```

### Performance Metrics

SAM-Audio includes a comprehensive metrics system to track execution time, minimal VRAM usage, and GPU utilization.

- **Automatic Reporting**: A summary table is printed after each run.
- **JSON Export**: Save metrics to a JSON file using `--metrics-json <path>` (or auto-generated in `logs/metrics/` if logging is enabled).
- **Control**: Disable metrics with `--no-metrics` or `SAM_AUDIO_METRICS=0`.

Example report:
```text
============================================================
                  PERFORMANCE METRICS REPORT
============================================================
 TIMING
Metric                         |       Last |       Mean |      Total
------------------------------------------------------------------
model_loading_time             |      1.23s |      1.23s |      1.23s
separation_process_time        |      2.45s |      2.45s |      2.45s
total_execution_time           |      3.78s |      3.78s |      3.78s

 MEMORY (VRAM)
Metric                         |       Peak |        Ave |       Last
------------------------------------------------------------------
pre_separation_vram_allocated  |      2.50GB |     2.50GB |     2.50GB
...
```

```

## Performance Optimizations

We have implemented several optimizations to ensure fast model loading and efficient execution:

1.  **Parallel Component Loading**:
    - `T5TextEncoder` and `DACVAE` are initialized asynchronously in background threads to hide their startup latency.

2.  **Meta Device Initialization**:
    - Large Transformer models (`DiT`) are initialized on the `meta` device to skip expensive random weight allocation.
    - Parameters are materially allocated mostly during checkpoint loading, reducing CPU memory spikes and initialization time.

3.  **Zero-Copy Loading**:
    - Checkpoints are memory-mapped and directly assigned to model parameters (`assign=True`), minimizing data copying and memory usage.

These optimizations reduce the initial model load time from **~16s to ~3s** (on typical hardware).

## Python API

For programmatic usage, import SAM-Audio directly:

```python
from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import torch

model = SAMAudio.from_pretrained("facebook/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()

file = "<audio file>" # audio file path or torch tensor
description = "<description>"

batch = processor(
    audios=[file],
    descriptions=[description],
).to("cuda")

with torch.inference_mode():
    # NOTE: `predict_spans` and `reranking_candidates` have a large impact on performance.
    # Setting `predict_span=True` and `reranking_candidates=8` will give you better results at the cost of
    # latency and memory. See the "Span Prediction" section below for more details
   result = model.separate(batch, predict_spans=False, reranking_candidates=1)

# Save separated audio
sample_rate = processor.audio_sampling_rate
torchaudio.save("target.wav", result.target.cpu(), sample_rate)      # The isolated sound
torchaudio.save("residual.wav", result.residual.cpu(), sample_rate)  # Everything else
```

### Prompting Methods

SAM-Audio supports three types of prompts:

1. **Text Prompting**: Describe the sound you want to isolate using natural language
   ```python
   processor(audios=[audio], descriptions=["A man speaking"])
   ```

2. **Visual Prompting**: Use video frames and masks to isolate sounds associated with visual objects
   ```python
   processor(audios=[video], descriptions=[""], masked_videos=processor.mask_videos([frames], [mask]))
   ```

3. **Span Prompting**: Specify time ranges where the target sound occurs
   ```python
   processor(audios=[audio], descriptions=["A horn honking"], anchors=[[["+", 6.3, 7.0]]])
   ```

See the [examples](examples) directory for more detailed examples

### Span Prediction (Optional for Text Prompting)

We also provide support for automatically predicting the spans based on the text description, which is especially helpful for separating non-ambience sound events.  You can enable this by adding `predict_spans=True` in your call to `separate`

```python
with torch.inference_mode()
   outputs = model.separate(batch, predict_spans=True)

# To further improve performance (at the expense of latency), you can add candidate re-ranking
with torch.inference_mode():
   outputs = model.separate(batch, predict_spans=True, reranking_candidates=8)
```

### Re-Ranking

We provide the following models to assess the quality of the separated audio:

- [CLAP](https://github.com/LAION-AI/CLAP): measures the similarity between the target audio and text description
- [Judge](https://huggingface.co/facebook/sam-audio-judge): measures the overall separation quality across 3 axes: precision, recall, and faithfulness (see the [model card](https://huggingface.co/facebook/sam-audio-judge#output-format) for more details)
- [ImageBind](https://github.com/facebookresearch/ImageBind): for visual prompting, we measure the imagebind embedding similarity between the separated audio and the masked input video

We provide support for generating multiple candidates (by setting `reranking_candidates=<k>` in your call to `separate`), which will generate `k` audios, and choose the best one based on the ranking models mentioned above

# Models

Below is a table of each of the models we released along with their overall subjective evaluation scores

| Model    | General SFX | Speech | Speaker | Music | Instr(wild) | Instr(pro) |
|----------|-------------|--------|---------|-------|-------------|------------|
| [`sam-audio-small`](https://huggingface.co/facebook/sam-audio-small) | 3.62        | 3.99   | 3.12    | 4.11  | 3.56        | 4.24       |
| [`sam-audio-base`](https://huggingface.co/facebook/sam-audio-base)   | 3.28        | 4.25   | 3.57    | 3.87  | 3.66        | 4.27       |
| [`sam-audio-large`](https://huggingface.co/facebook/sam-audio-large) | 3.50        | 4.03   | 3.60    | 4.22  | 3.66        | 4.49       |

We additional release another variant (in each size) that is better specifically on correctness of target sound as well as visual prompting:
- [`sam-audio-small-tv`](https://huggingface.co/facebook/sam-audio-small-tv)
- [`sam-audio-base-tv`](https://huggingface.co/facebook/sam-audio-base-tv)
- [`sam-audio-large-tv`](https://huggingface.co/facebook/sam-audio-large-tv)

## Evaluation

See the [eval](eval) directory for instructions and scripts to reproduce results from the paper

## Contributing

See [contributing](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.
