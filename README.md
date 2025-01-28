# HalluShift

This repository contains the source code for [**HalluShift: Measuring Distribution Shifts towards Hallucination Detection**].

---

## **Model Preparation**

1. **Setup environment**  
   - Install `Python 3.10.12` and the necessary packages from `requirements.txt`.
   - For easily managing different python versions, we recommend using [conda](https://docs.anaconda.com/miniconda/install/).
   - Create a new environment in conda and install necessary python packages:
     ```bash
     conda create -n hallushift python=3.10.12 -y
     conda activate hallushift
     pip install -r requirements.txt
     ```
2. **Setup Language Models**
   - You can choose between [
Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)(default) and [
opt-6.7b](https://huggingface.co/facebook/opt-6.7b).
   - Login to `huggingface` or create an account if you don't have already.
   - From the [settings](https://huggingface.co/settings/tokens) create a new access token with WRITE access.
   - Open the `hal_detection.py` and paste your access token at `line 15, hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"`
3. **Setup Directory**  
   Create a folder to store the downloaded models:
   ```bash
   mkdir models
   ```
   Place the model checkpoints inside the `models` folder.

---

## **Generating Responses and Hallucination Detection**

1. **Setup Results Directory**  
   Create a folder to save:
   - LLM-generated answers and Ground truth labels for model-generated content
   - Features for training classifiers
   ```bash
   mkdir results
   ```

2. **Ground Truth Evaluation**  
   Since generated answers lack explicit ground truth, we use [BleuRT](https://arxiv.org/abs/2004.04696) to evaluate truthfulness.

3. **Download BleuRT Models**  
   Refer to the [BleuRT repository](https://github.com/google-research/bleurt) and save the models in the `./models` folder.

4. **Hallucination Detection for TruthfulQA**  
   To perform hallucination detection on the **TruthfulQA** dataset run the following command:
   ```bash
   python hal_detection.py --dataset_name truthfulqa --model_name llama2_7B 
   ```
   - `dataset_name`: Choose from `truthfulqa`, `triviaqa`, `tydiqa`, `coqa`, `haluevalqa`, `haluevaldia`, `haluevalsum`.
   - `model_name`: Choose from `llama2_7B`, `llama3_8B`, or `opt6.7B`.

   **Note:** If you encounter memory errors, consider reducing the number of workers using the `--num_workers` parameter. For example:
   ```bash
   python hal_detection.py --dataset_name truthfulqa --model_name llama2_7B --num_workers 1
   ```
   Refer to Section IV of the paper for implementation details.

---
* **Demo**
   To quickly evaluate the pre-trained model on the TruthfulQA dataset (which has already been processed and inferred using the LLaMA-2 7B model), run the following command:
   ```bash
   python demo/demo.py
   ```
   The pre-trained model and the processed dataset are provided in the `demo` folder for easy evaluation.
