# HalluShift

This repository contains the source code for [**HalluShift: Measuring Distribution Shifts towards Hallucination Detection**].

---

## **Model Preparation**

1. **Setup environment and Download Models**  
   - Install `Python 3.10.12` and the necessary packages from `requirment.txt`.
   - Get the [LLaMA-2 7B](https://huggingface.co/meta-llama) and [OPT 6.7B](https://huggingface.co/facebook/opt-6.7b) models.
   
2. **Setup Directory**  
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
