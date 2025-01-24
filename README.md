# HalluShift

This repository contains the source code for [**HalluShift: Measuring Distribution Shifts towards Hallucination Detection**] by some random people.
---

## **Model Preparation**

1. **Download Models**  
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
   - LLM-generated answers
   - Ground truth labels for model-generated content
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

   Refer to Section IV of the paper for implementation details.

---


## **Citation**

```plaintext
(Include citation details for your paper here)
