# Automated Audio Intelligence Pipeline

An **offline, automated data-engineering architecture** designed to solve the "cold start" dataset bottleneck for Generative AI music models (such as Latent Diffusion and LoRA training).

## 🚀 Key Features & Innovations

1. **Hybrid Neural-DSP Architecture:** Seamlessly combines a state-of-the-art Neural Transformer network (`beat-this`) with classical Digital Signal Processing (DSP) algorithms (IOI Histograms, Autocorrelation) to guarantee maximum extraction accuracy for rhythmically complex audio.
2. **Advanced Feature Extraction:** Autonomously extracts metadata that traditionally requires human engineers:
   * **Exact BPM**
   * **Time Signature (e.g. 4/4, 3/4)**
   * **Musical Scale & Key (Krumhansl-Schmuckler algorithm)**
   * **Deep DSP Parameters (Harmonic-Percussive Source Separation and Transient Density)**
3. **Automated Audio Engineering:** 
   * Precisely trims digital silence from the start and end of tracks.
   * Algorithmically stitches short audio segments using **equal-power crossfades** to generate seamless, continuous loops that perfectly fit the AI model's context window (e.g., 45s lengths).
4. **Smart Metatada Generation:** 
   * Intelligently renames physical audio files to permanently embed extracted intelligence.
   * Automatically generates heavily structured `.json` metadata and `.txt` caption sidecar files, which serve as direct instructional text inputs for AI model training.
5. **Offline Privacy & Scale:** Executes 100% locally on CPU, ensuring total data privacy while processing thousands of files in bulk without cloud API rate limits.

## ⚙️ How It Works

* **Tier 1:** A pre-trained Neural Transformer natively extracts beats, downbeats, and Time Signatures.
* **Tier 2/3:** Inter-Onset-Interval (IOI) Histograms and Autocorrelation serve as deterministic physical fallbacks if neural extraction fails.
* **Tier 4:** Standard librosa dynamic programming fallback.

## 🛠 Usage
Launch the interactive Gradio UI to configure directories and deep DSP parameters:
```bash
uv run python3 dataset_gui.py
```
* **BPM Control:** Accepts manual BPM overrides, filename hints (e.g., "100bpm"), or fully autonomous Neural/DSP extraction.
* **Advanced Deep DSP Checkbox:** Extracts `Type: Percussive/Harmonic` and `Density: Sparse/Dense` parameters into the final training captions.

## 💼 Commercial Impact
By solving the data-bottleneck, this pipeline accelerates the research and development lifecycle of foundational AI audio models from months of manual human audio-engineering labor down to hours of unattended, automated computation.
