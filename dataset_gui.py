import gradio as gr
from prepare_lora_dataset import process_directory
import os


def run_pipeline(input_dir, output_dir, quarantine_dir, dataset_name, max_duration, manual_bpm_raw, enable_advanced_ui):
    if not all([input_dir, output_dir, quarantine_dir, dataset_name]):
        yield "❌ Error: Please fill in all directory and dataset name fields."
        return

    if not os.path.exists(input_dir):
        yield f"❌ Error: Source directory does not exist:\n  {input_dir}"
        return

    # Parse manual BPM — None means auto-detect
    manual_bpm = None
    try:
        v = float(str(manual_bpm_raw).strip()) if manual_bpm_raw else 0
        if 40 <= v <= 300:
            manual_bpm = v
    except (ValueError, TypeError):
        pass

    if manual_bpm:
        yield f"🎯 Manual BPM Override active: {manual_bpm:.0f} BPM will be applied to ALL files.\n"
    else:
        yield "🔍 BPM Override: OFF — auto-detection will be used (filename hints + DSP).\n"

    log_output = "🚀 Starting Audio Intelligence Pipeline...\n"
    yield log_output

    for msg in process_directory(
        input_dir, output_dir, quarantine_dir, dataset_name, max_duration,
        manual_bpm=manual_bpm, enable_advanced=enable_advanced_ui
    ):
        log_output += msg
        yield log_output


with gr.Blocks(title="Audio Intelligence DSP Pipeline — Boobalan Arjunan") as demo:

    gr.Markdown("""
# 🎛️ Audio Intelligence DSP Pipeline
### Product Owner & Lead Architect: **Boobalan Arjunan**

> *Enterprise-grade automated data engineering for AI music models.*
> *Uses Krumhansl-Schmuckler Key Profiles, IOI Beat Tracking, Onset-based One-Shot Classification,
> and Equal-Power Crossfade Loop Stitching to prepare production-ready LoRA training datasets.*
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Section 1: Directory Configuration")
            gr.Markdown("*Paste the absolute paths for your local folders below.*")
            input_dir    = gr.Textbox(label="Source Audio Directory (Raw WAVs)",
                                      placeholder="/Users/mymac/Desktop/Raw_Audio")
            output_dir   = gr.Textbox(label="Processed Output Directory (AI-Ready Dataset)",
                                      placeholder="/Users/mymac/Desktop/Processed_Dataset")
            quarantine_dir = gr.Textbox(label="🚨 Quarantine Directory (Bad / Corrupted Files)",
                                        placeholder="/Users/mymac/Desktop/Quarantine_Vault")

            gr.Markdown("### ⚙️ Section 2: Model Configuration")
            dataset_name = gr.Textbox(label="Dataset Name",
                                      placeholder="e.g. BollyHood Beats",
                                      value="BollyHood Beats")
            max_duration = gr.Slider(
                minimum=5, maximum=180, value=45, step=1,
                label="Target Context Window (Seconds)",
                info="Max audio duration fed into the DiT. 45s is required for ACE-Step 4B to avoid OOM."
            )

            gr.Markdown("### 🎵 Section 3: BPM Control")
            manual_bpm_input = gr.Number(
                label="Manual BPM Override (leave 0 or blank for auto-detect)",
                value=0,
                minimum=0,
                maximum=300,
                step=1,
                info="⚡ Set this if you KNOW the BPM of your sample pack (e.g. 100, 120, 85). "
                     "This overrides ALL auto-detection and gives 100% accurate BPM labels. "
                     "Leave as 0 to use smart auto-detection from filename hints + DSP."
            )
            
            gr.Markdown("### 🧠 Section 4: Advanced AI Parameters")
            enable_advanced_ui = gr.Checkbox(
                label="Enable Advanced Deep DSP (HPSS & Density)", 
                value=False, 
                info="Adds 'Percussive/Harmonic' and 'Dense/Sparse' tags to your dataset. Takes slightly longer to compute."
            )

            run_btn = gr.Button("🚀 Initialize DSP Engine", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Section 4: Execution Engine & Live Console")
            gr.Markdown(
                "**BPM Detection Priority:** Manual Override → Filename Hint (e.g. `100bpm`) → "
                "Trailing number (e.g. `tabla_100.wav`) → DSP Auto-Detect. "
                "If your files have no BPM label in the name, use the **Manual BPM Override** above."
            )
            console_output = gr.Textbox(
                label="Pipeline Log",
                lines=28, max_lines=32,
                interactive=False
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_dir, output_dir, quarantine_dir, dataset_name, max_duration, manual_bpm_input, enable_advanced_ui],
        outputs=[console_output]
    )


if __name__ == "__main__":
    print("Launching Premium Audio Intelligence Pipeline UI...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7865,
        theme=gr.themes.Monochrome(primary_hue="zinc")
    )
