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

HyFlexPIM_llama3.ipynb: Main simulation notebook for Llama 3 model (with labeled steps)
HyFlexPIM_BERT.ipynb: Main simulation notebook BERT-Large model(with labeled steps)

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

## 🚀 How to Run
Launch the HyFlexPIM_llama3.ipynb (or HyFlexPIM_BERT.ipynb) notebook using Jupyter and follow each block labeled Step X.
The steps are sequential and can be executed one by one.

## 🧪 Simulation Steps 

## ✅ Step 1: Model Preparation 

## Llama3

By default, the notebook uses the LLaMA-3.2 1B model and the PTB dataset.

To use the LLaMA 3.2 1B (or BERT) model, you must have **access to the gated repository** on Hugging Face.

1. **Log in to Hugging Face:**
   - Visit [https://huggingface.co/login](https://huggingface.co/login) and log in with your account.

2. **Request access to the LLaMA 3.2 1B model:**
   - Go to [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
   - Click the **"Access repository"** button and agree to Meta’s license.

3. **Create a read-only access token:**
   - Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click **"New token"**, give it a name like `llama3-token`, set the role to **read**, and create the token.
   - Copy the token that looks like: `hf_...`
   - Use this token in cell 3 of HyFlexPIM_llama3.ipynb
  
## BERT Large

1. You don't need specific access to run, just run all the cells in step 1.

You can replace them with your own quantized model and dataset.

The function evaluate_model() can be used to evaluate loss (optional).

## ✅ Step 2: Pre-SVD Fine-tuning (Recommended)
Before applying SVD, we recommend fine-tuning the model for 2–3 epochs.
Better pre-trained performance usually improves noise robustness.

After training, merge all adapter paths using:
```bash
model = model.merge_and_unload()
```
This step consolidates model weights and prepares them for SVD decomposition. (This is only for Llama3)

## ✅ Step 3: Singular Value Decomposition (SVD)
Apply SVD and truncate all linear layers via:

```bash
replace_linear_layer_{your model}(model)
```
Converts all Linear layers into low-rank U Σ Vᵀ format.

## ✅ Step 4: Fine-tuning & Gradient Redistribution
To recover any accuracy degradation from SVD:

Fine-tune the model for 1–3 epochs.

Perform gradient redistribution based on gradient sensitivity.

Gradient files are saved in ./gradient/ and used for selective noise injection.
Save the final model after this step.

## ✅ Step 5: Noise-Injection Inference to generate plots
First, set the standard deviation (std) for noise injection (default is 0.025), and change the lists inside thresholds
e.g., thresholds = [0, 20] --> Simulation for 0% of SLC rate, and 20% of SLC rate
      thresholds = [0, 20, 40] --> Simulation for 0% of SLC rate, 20% of SLC rate, 40% of SLC rate

   - std: Controls noise magnitude.

   - thresholds: Determines the proportion of weights stored in SLC by adjusting a threshold parameter.



Second, set the path of your best-trained model after SVD.

Then, run inference to evaluate the noise-aware model loss. (This will generate the plots based on your SLC (thresholds) rate)






## 📝 License
This project is licensed under the [MIT License](./LICENSE).


