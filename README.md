# HyFlexPIM  
**HyFlexPIM Noise-Aware Simulator**

This is the **noise-aware simulator** for **HyFlexPIM**, developed to evaluate the accuracy results shown in **Figure 12** and **Figure 13** of our ISCA 2025 submission.

The simulation flow consists of the following five steps:
- Model Preparation
- Original model training 
- SVD Preprocessing   
- Fine-tuning & Gradient Redistribution  
- Noise-injected Inference

Repository URL: [https://github.com/songchangeun96/HyFlexPIM](https://github.com/songchangeun96/HyFlexPIM)

---

📁 File Structure
HyFlexPIM_llama3.ipynb: Main simulation notebook (with labeled steps)

hyflex_utils.py: Utility functions

requirements.txt: List of required Python packages

## 🔧 Installation

First, clone the repository:

```bash
git clone https://github.com/songchangeun96/HyFlexPIM.git
cd HyFlexPIM
```


Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

🚀 How to Run
Launch the HyFlexPIM_llama3.ipynb notebook using Jupyter and follow each block labeled Step X.
The steps are sequential and can be executed one by one.

🧪 Simulation Steps
✅ Step 1: Model Preparation
By default, the notebook uses the LLaMA-3.2 1B model and the PTP dataset.
You can replace them with your own quantized model and dataset.

The function evaluate_model() can be used to evaluate accuracy (optional).

✅ Step 2: Pre-SVD Fine-tuning (Recommended)
Before applying SVD, we recommend fine-tuning the model for 2–3 epochs.
Better pre-trained performance usually improves noise robustness.

After training, merge all adapter paths using:
```bash
model = model.merge_and_unload()
```
This step consolidates model weights and prepares them for SVD decomposition.

✅ Step 3: Singular Value Decomposition (SVD)
Apply SVD and truncate the Linear layers via:
```bash
replace_linear_layer_llama(model, rank_ratio)
```
Converts all Linear layers into low-rank U Σ Vᵀ format.

rank_ratio controls the truncation level while maintaining parameter and OPS counts.

✅ Step 4: Fine-tuning & Gradient Redistribution
To recover any accuracy degradation from SVD:

Fine-tune the model for 1–3 epochs.

Perform gradient redistribution based on gradient sensitivity.

Gradient files are saved in ./gradient/ and used for selective noise injection.
Save the final model after this step.

✅ Step 5: Noise-Injection Inference
Reload the gradients and apply noise:

```bash
load_gradients(...)
apply_noise_to_llama(model, std=..., threshold=...)
```

std: Controls noise magnitude.

th: Determines the SLC rate via gradient sensitivity ranking.

Then, run inference to evaluate noise-aware model accuracy.

