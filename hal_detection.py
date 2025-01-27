import functions, classifier
from datasets import load_dataset, Dataset
import json
import time
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings
warnings.filterwarnings("ignore")

# Mapping of model names to their identifiers
MODELS_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b"
}

def seed_everything(seed: int):
    """Sets seeds for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_dataset_by_name(dataset_name):
    """Loads a dataset based on the provided name.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        Dataset: The loaded dataset.
    """
    
    # Load the TruthfulQA dataset's validation split
    if dataset_name == "truthfulqa":
        return load_dataset("truthful_qa", 'generation')['validation']
    
    # Load the TriviaQA dataset and remove duplicate questions
    elif dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        return dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    
    # Load the TyDiQA dataset and filter for English questions
    elif dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        return dataset.filter(lambda row: "english" in row["id"])
    
    # Load the CoQA dataset
    elif dataset_name == 'coqa':
        return load_coqa_dataset()
    
    # Load a specific subset of the HaluEval dataset
    elif dataset_name == 'haluevaldia':
        return load_dataset("pminervini/HaluEval", "dialogue")['data']
    elif dataset_name == 'haluevalqa':
        return load_dataset("pminervini/HaluEval", "qa")['data']
    elif dataset_name == 'haluevalsum':
        return load_dataset("pminervini/HaluEval", "summarization")['data']
    else:
        raise ValueError("Invalid dataset name")

def load_coqa_dataset():
    """
    Downloads and processes the CoQA dataset.
    Returns:
        Dataset: The processed CoQA dataset.
    """
    import urllib.request
    save_path = './coqa_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"{save_path}/coqa-dev-v1.0.json"):
        # Download the CoQA dataset if not already present
        url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
        try:
            urllib.request.urlretrieve(url, f"{save_path}/coqa-dev-v1.0.json")
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    # Load and process the dataset
    with open('./coqa_dataset/coqa-dev-v1.0.json', 'r') as infile:
        data = json.load(infile)['data']
        dataset = {
            'story': [],
            'question': [],
            'answer': [],
            'additional_answers': [],
            'id': []
        }
        for sample in data:
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))
                additional_answers_list = [
                    additional_answers[str(i)][question_index]['input_text'] for i in range(3)
                ]
                dataset['additional_answers'].append(additional_answers_list)
                story += f' Q: {question["input_text"]} A: {answers[question_index]["input_text"]}'
                if story[-1] != '.':
                    story += '.'
        return Dataset.from_dict(dataset)

def process_with_threads(args, dataset, process_func, max_workers):
    """Processes a dataset in parallel using threading.
    Args:
        dataset (Dataset): The dataset to process.
        process_func (callable): The function to apply to each dataset entry.
        max_workers (int): The maximum number of threads to use.
    Returns:
        list: Processed dataset entries.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(process_func, dataset), total=len(dataset), desc=f"Generating responses for {args.dataset_name} dataset ..."))

def main():
    """
    Main function to perform the following tasks:
    - Parse command-line arguments.
    - Download and preprocess datasets.
    - Load and configure language models.
    - Generate responses using the language model.
    - Evaluate generated responses using BLEURT.
    - Train classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', help='Name of the model to use.')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa', help='Name of the dataset to use.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpu threads to use.')
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(42)

    # Determine model path (local or remote)
    if args.model_name not in MODELS_NAMES:
        raise ValueError("Invalid model name")
    MODEL = MODELS_NAMES[args.model_name]
    
    print(f"""
    =========================================================================
                        HalluShift Execution Started
    =========================================================================
    Dataset: {args.dataset_name}    Model: {args.model_name}

    Workflow:
    1. LLM Response Generation and feature collection :
    - Estimated Time: Varies by dataset size, context length, and hardware
    - Example: TruthfulQA on llama2-7B takes ~45-60 min on NVIDIA GeForce RTX 3090

    2. Ground Truth Evaluation :
    - Method: BleuRT evaluation of LLM generated responses
    - Estimated Time: Varies by dataset size and answer length
    - Example: TruthfulQA takes ~30-60 sec on NVIDIA GeForce RTX 3090

    3. Feature Processing and Classifier Training :
    - Estimated Time: Varies by dataset size
    - Example: TruthfulQA takes ~60-90 sec on NVIDIA GeForce RTX 3090

    Output: 
    - Various evaluation metrics are displayed on the screen
    - LLM responses and processed dataset stored in 'result' folder
    =========================================================================\n
    """)
    time.sleep(10)
    
    print("Downloading Dataset...\n")
    dataset = load_dataset_by_name(args.dataset_name)
    print("Dataset successfully downloaded.\n")

    print("Initializing  LLM...\n")
    os.makedirs("./models", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="/home/iplab/LLM/models/",
        attn_implementation="eager").to("cuda")
    print("\nLLM successfully initialized.\n")

    # Configure prompt templates for different datasets
    os.makedirs(f'./results/{args.dataset_name}_processed/', exist_ok=True)
    base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A:",
        'triviaqa': "Answer the question concisely. Q: {question} A:",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:",
        'haluevaldia': "You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message.\nKnowledge: {knowledge}.\nConversations: {dialogue_history} [Assistant]:",
        'haluevalqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'haluevalsum': "{document} \n Please summarize the above article concisely. A:"
    }

    base_prompt = base_prompts.get(args.dataset_name, "")

    def process_row(row):
        if args.dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'tydiqa':
            prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'coqa':
            prompt = tokenizer(base_prompt.format(story=functions.truncate_after_words(row['story']), question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevaldia':
            prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevalqa':
            prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevalsum':
            prompt = tokenizer(base_prompt.format(document=functions.truncate_after_words(row['document'])),padding=True, return_tensors='pt').to("cuda") 

        generated = model.generate(**prompt,
                                    do_sample=False,
                                    max_new_tokens=64,
                                    pad_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_hidden_states=True,     
                                    output_attentions=True,      
                                    output_logits=True)

        decoded = tokenizer.decode(generated.sequences[0, prompt["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        return (
            functions.plot_internal_state_2(generated)
            + functions.plot_internal_state_2(generated, state="attention")
            + functions.probability_function(generated)
            + [decoded]
        )
    
    if args.num_workers == 1:
        result = []
        for index, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Generating responses for {args.dataset_name} dataset ..."):
            if args.dataset_name in ['truthfulqa', 'triviaqa']:
                prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'tydiqa':
                prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'coqa':
                prompt = tokenizer(base_prompt.format(story=functions.truncate_after_words(row['story']), question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevaldia':
                prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevalqa':
                prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevalsum':
                prompt = tokenizer(base_prompt.format(document=functions.truncate_after_words(row['document'])), return_tensors='pt').to("cuda") 

            generated = model.generate(**prompt,
                                        do_sample=False,
                                        max_new_tokens=64,
                                        pad_token_id=tokenizer.eos_token_id,
                                        return_dict_in_generate=True,
                                        output_hidden_states=True,     
                                        output_attentions=True,      
                                        output_logits=True)

            decoded = tokenizer.decode(generated.sequences[0, prompt["input_ids"].shape[-1]:],
                                        skip_special_tokens=True)
            result.append(
                functions.plot_internal_state_2(generated)
                + functions.plot_internal_state_2(generated, state="attention")
                + functions.probability_function(generated)
                + [decoded]
            )
        
    else : 
        result = process_with_threads(args, dataset, process_row, max_workers=args.num_workers)

    # Save the results to a DataFrame
    df = pd.DataFrame(result)
    answers = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    print("\nResponses successfully generated.\n")
    print("=========================================================================\n")
    
    time.sleep(5)
    
    print("Starting the BLEURT setup for evaluation...\n")
    # correct answers for questions 
    answer_mapping = {
        'truthfulqa': ['best_answer', 'correct_answers','question'],
        'triviaqa': ['answer','question'],
        'coqa': ['answer','question'],
        'tydiqa': ['answers','question'],
        'haluevaldia': ['right_response','dialogue_history'],
        'haluevalqa': ['right_answer','question'],
        'haluevalsum': ['right_summary','document']
    }

    if args.dataset_name in answer_mapping:
        keys = answer_mapping[args.dataset_name][:-1]
        result_dataset = pd.DataFrame([{key: d[key] for key in keys} for d in dataset])
        if args.dataset_name == 'truthfulqa':
            result_dataset['all_answers'] = result_dataset['best_answer'].apply(lambda row: [row])
        elif args.dataset_name == 'triviaqa':
            result_dataset['all_answers'] = result_dataset['answer'].apply(lambda row: row['aliases'])
        elif args.dataset_name == 'tydiqa':
            result_dataset['all_answers'] = result_dataset['answers'].apply(lambda row: row['text'])
        elif args.dataset_name == 'coqa':
            result_dataset['all_answers'] = result_dataset['answer'].apply(lambda row: [row['text']])
        else:
            result_dataset['all_answers'] = result_dataset[keys[0]].apply(lambda row: [row])
    result_dataset = pd.DataFrame({
        'answers': result_dataset['all_answers'],
        'llm_answer': answers.values,
        'id': [str(i) for i in range(len(result_dataset))]
    })
    result_dataset = result_dataset.explode('answers', ignore_index=True)

    # Installing BLEURT model
    print("Downloading BLEURT model...\n")
    if not os.path.exists("./models/BLEURT-20-D12"):
        os.system("wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip -O ./models/BLEURT-20-D12.zip")
        os.system("unzip -o ./models/BLEURT-20-D12.zip -d ./models")
    functions.column_to_txt(result_dataset, 'answers', 'answers')
    functions.column_to_txt(result_dataset, 'id', 'id')
    functions.column_to_txt(result_dataset, 'llm_answer', 'llm_answer')
    print("BLEURT model successfully downloaded\n")

    print("Running BLEURT scoring for response evaluation...\n")
    os.system(
        "python -m bleurt.score_files "
        "-candidate_file=llm_answer "
        "-reference_file=answers "
        "-bleurt_batch_size=100 "
        "-batch_same_length=True "
        "-bleurt_checkpoint=models/BLEURT-20-D12 "
        "-scores_file=scores"
    )
    print("=========================================================================\n")
    
    time.sleep(5)
    
    # Preparing data for training the classifier
    print("Starting Data Processing...\n")
    df_bleurt = functions.bleurt_processing("id", "scores", 0.5)
    
    # Save LLM responses with bleurt 
    pd.DataFrame({
        'questions': dataset[answer_mapping[args.dataset_name][-1]][:],
        'llm_answer': answers.values,
        'bleurt_score': df_bleurt['bleurt_score'], 
        'hallucination' : df_bleurt['hallucination']
    }).to_csv(f'./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_responses_with_bleurt.csv')
    
    data = functions.data_preparation(df, df_bleurt)
    data.to_parquet(f'./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_dataset.pq')
    
    # Remove unnecessary files
    if os.path.exists("llm_answer") and os.path.exists("answers") and os.path.exists("scores") and os.path.exists("id") :
        os.remove("llm_answer")
        os.remove("answers")
        os.remove("id")
        os.remove("scores")
    else:
        raise ValueError("BLEURT Score files not found")
    print("=========================================================================\n")
    
    time.sleep(5)
    
    print("Starting classifier training with the processed dataset...\n")
    if args.dataset_name in ['truthfulqa', 'triviaqa', 'tydiqa', 'coqa']:
        trained_model = classifier.train_combined_model(data, test_size=0.25)
    elif args.dataset_name in ['haluevaldia', 'haluevalqa', 'haluevalsum']:
        trained_model = classifier.train_combined_model(data, test_size=0.9)
    torch.save(trained_model.state_dict(), f"./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_model.pth")
    
    print("\nHalluShift execution completed successfully.\n")
    print("All results and trained model have been saved in the 'result' folder.\n")    
    print("=========================================================================\n")

if __name__ == '__main__':
    main()
