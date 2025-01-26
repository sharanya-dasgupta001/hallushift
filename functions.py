import itertools
import torch 
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np

def normalize_as_distribution(tensor):
    """Normalize the input tensor as a probability distribution.
    This function reshapes the input tensor and then applies the softmax
    function to it, effectively normalizing it into a probability distribution.
    Args:
        tensor: The input tensor.
    Returns:
        The normalized tensor (probability distribution).
    """
    tensor = tensor.view(-1)
    return F.softmax(tensor, dim=-1)


def wasserstein_dist(p, q):
    """Calculate the Wasserstein distance between two distributions.
    This function computes the Wasserstein distance, a measure of the distance
    between two probability distributions.
    Args:
        p: The first distribution.
        q: The second distribution.
    Returns:
        The Wasserstein distance between p and q.
    """
    p = p.to(torch.float32).cpu().numpy()
    q = q.to(torch.float32).cpu().numpy()
    return wasserstein_distance(p, q)

def cosine_similarity(tensor1, tensor2):
    """Calculate the cosine similarity between two tensors.
    This function computes the cosine similarity, a measure of similarity between two non-zero tensors.
    Args:
        tensor1: The first tensor.        
        tensor2: The second tensor.
    Returns:
        The cosine similarity between the two tensors.
    """
    tensor1 = tensor1.view(-1, tensor1.size(-1)).to(torch.float32)  
    tensor2 = tensor2.view(-1, tensor2.size(-1)).to(torch.float32)
    return F.cosine_similarity(tensor1, tensor2).item()

def plot_internal_state_2(outputs, state="hidden"):
    """Analyze internal model states and calculate various metrics.

    This function processes hidden states or attentions from model outputs,
    calculates Wasserstein distances and cosine similarities between consecutive states,
    and averages these metrics across all tokens.

    Args:
        outputs: Model outputs containing hidden states or attentions.
        state:  The type of state to analyze, either "hidden" or "attentions". Defaults to "hidden".

    Returns:
        A list of average Wasserstein distances and cosine similarities.
    """
    results = []
    index_sums = [0] * 30
    if state == "hidden":
        for tup in outputs.hidden_states:
            vec = [normalize_as_distribution(tup[i]) for i in range(2,33,2)]
            div = [wasserstein_dist(vec[i], vec[i+1]) for i in range(len(vec)-1)]
            div.extend(cosine_similarity(vec[i], vec[i+1]) for i in range(len(vec)-1))
            results.append(div)


    else:
        for tup in outputs.attentions:
            vec = [normalize_as_distribution(tup[i]) for i in range(1,32,2)]
            div = [wasserstein_dist(vec[i], vec[i+1]) for i in range(len(vec)-1)]
            div.extend(cosine_similarity(vec[i], vec[i+1]) for i in range(len(vec)-1))
            results.append(div)


    for res, i in itertools.product(results, range(30)):
        index_sums[i] += res[i]

    return [sum_val / len(results) for sum_val in index_sums]
        

def probability_function(output):
    """Calculate the maximum and minimum probabilities from model logits.
    This function iterates through the logits in the model output, 
    calculates the softmax probabilities for each logit, 
    and then determines the maximum and minimum probability values.
    Args:
        output: The model output containing logits.

    Returns:
        A tuple containing two lists: the maximum probabilities and the minimum probabilities.
    """
    max_prob_results = []
    min_prob_results = []
    spread_results = []
    for logit in output.logits:
        probabilities = F.softmax(logit[0], dim=0)
        max_prob = probabilities.max().item()
        min_prob = probabilities.min().item()
        max_prob_results.append(max_prob)
        min_prob_results.append(min_prob)
        spread_results.append(max_prob - min_prob)
    return [max_prob_results, min_prob_results]

def truncate_after_words(text, num_words=128):
        words = text.split()
        return " ".join(words[:num_words])

def column_to_txt(dataset, column_name, txt_file):
    """Writes the contents of a specified column in a dataset to a text file.

    This function iterates through each row of the dataset and writes the content
    of the specified column to a text file. Newlines and carriage returns are
    replaced with spaces to ensure each entry is on a single line.

    Args:
        dataset: The input dataset (list of dictionaries or pandas DataFrame).
        column_name (str): The name of the column to extract.
        txt_file (str): The path to the output text file.
    """
    try:
        with open(txt_file, mode='w', encoding='utf-8') as txtfile:
            for text in dataset[column_name]:
                sanitized_text = text.replace('\n', ' ').replace('\r', ' ')
                txtfile.write(sanitized_text + '\n')

    except Exception as e:
        print(f"An error occurred while creating txt files: {e}")

def bleurt_processing(file1, file2, threshold=0.5):
    """Processes BLEURT scores to determine hallucinations.

    Reads BLEURT scores from a file, groups them by ID, and assigns a hallucination
    label based on a threshold.  If the maximum BLEURT score for an ID is above the
    threshold, it's considered not a hallucination (0), otherwise it is (1).

    Args:
        file1 (str): Path to the file containing IDs.
        file2 (str): Path to the file containing BLEURT scores.
        threshold (float, optional): The threshold for BLEURT score. Defaults to 0.5.

    Returns:
        pandas.DataFrame: DataFrame with 'id', 'bleurt_score', and 'hallucination' columns.
        Returns None if the input files have different lengths.
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f3:
            column1 = [line.strip() for line in f3.readlines()]
        with open(file2, 'r', encoding='utf-8') as f4:
            column2 = [line.strip() for line in f4.readlines()]

        if len(column1) == len(column2) :
            df = pd.DataFrame({
                'id' : column1,
                'bleurt_score': column2
            })
            df = df.groupby('id', as_index=False, sort=False)['bleurt_score'].max()
            df['hallucination'] = df['bleurt_score'].astype(float).apply(lambda x: 0 if x > threshold else 1)
            return df
        else :
            raise ValueError("All columns are not of same length during bleurt processing")
    except Exception as e:
        print(f"An error occurred while bleurt processing: {e}")

def normalized_entropy(prob_list):
    """Calculates the normalized entropy of a probability distribution.

    This function computes the entropy of a probability distribution represented
    by a list of probabilities and normalizes it by the maximum possible entropy
    for a distribution with the same number of elements.

    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The normalized entropy, a value between 0 and 1.
        Returns 0 if the input list is empty or contains only zeros.
    """
    entropy = -np.sum([p * np.log(p) for p in prob_list if p > 0]) 
    max_entropy = np.log(len(prob_list))
    return entropy / max_entropy if max_entropy > 0 else 0

def count_low_probs(prob_list, threshold=0.1):
    """Counts the number of probabilities below a threshold.
    This function iterates through a list of probabilities and counts how many
    fall below a specified threshold.

    Args:
        prob_list: A list of probabilities.
        threshold (float, optional): The threshold value. Defaults to 0.1.
    Returns:
        int: The number of probabilities below the threshold.
    """
    return sum(p < threshold for p in prob_list)

def count_high_probs(prob_list, threshold=0.9):
    """Counts the number of probabilities above a threshold.
    This function iterates through a list of probabilities and counts how many
    are above a specified threshold.

    Args:
        prob_list: A list of probabilities.
        threshold (float, optional): The threshold value. Defaults to 0.9.

    Returns:
        int: The number of probabilities above the threshold.
    """
    return sum(p > threshold for p in prob_list)

def probability_gradients(prob_list):
    """Calculates the absolute differences between consecutive probabilities.

    This function computes the absolute gradients of a list of probabilities by taking the
    absolute difference between consecutive elements.

    Args:
        prob_list: A list of probabilities.

    Returns:
        list: A list of absolute probability gradients.
        Returns an empty list if the input list has fewer than two elements.
    """
    return [abs(prob_list[i+1] - prob_list[i]) for i in range(len(prob_list) - 1)]

def mean_gradient(prob_list):
    """Calculates the mean absolute difference between consecutive probabilities.

    This function computes the mean absolute gradient of a list of probabilities by taking the
    absolute difference between consecutive elements and calculating the average of these differences.

    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The mean absolute probability gradient.
        Returns 0 if the input list has fewer than two elements.
    """
    gradients = probability_gradients(prob_list)
    return np.mean(gradients) if gradients else 0

def max_gradient(prob_list):
    """Calculates the maximum absolute difference between consecutive probabilities.

    This function computes the maximum absolute gradient of a list of probabilities by taking the
    absolute difference between consecutive elements and returning the maximum of these differences.

    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The maximum absolute probability gradient.
        Returns 0 if the input list has fewer than two elements.
    """
    gradients = probability_gradients(prob_list)
    return max(gradients) if gradients else 0

def percentile(prob_list, q):
    """Calculates the q-th percentile of a list of probabilities.

    This function computes the q-th percentile of the given probability list using NumPy.

    Args:
        prob_list: A list of probabilities.
        q (float or int): Percentile to compute, which must be between 0 and 100 inclusive.

    Returns:
        float: The q-th percentile of the probability list.
    """
    return np.percentile(prob_list, q)

def data_preparation(df_1, df_2):
    """Prepares data for hallucination detection model training.

    This function engineers features from divergence measures and probability distributions, merges them with hallucination labels,
    and prepares the dataset for training a hallucination detection model.

    Args:
        df_1 (pandas.DataFrame): DataFrame containing divergence measures and probability distribution features.
        df_2 (pandas.DataFrame): DataFrame containing hallucination labels.

    Returns:
        pandas.DataFrame: The merged and feature-engineered DataFrame.

    Raises:
        ValueError: If the input DataFrames have different lengths.
    """

    # Creating Probabilistic Features
    temp_120 = df_1[60].copy()
    temp_121 = df_1[61].copy()

    df_1[61] = df_1.apply(
        lambda row: max(a - b for a, b in zip(row[60], row[61])), axis=1 )
    df_1[60] = df_1[60].apply(lambda x : min(x))

    # Add new features
    df_1['norm_entropy_max'] = temp_120.apply(normalized_entropy)
    df_1['norm_entropy_min'] = temp_121.apply(normalized_entropy)

    df_1['low_prob_count_max'] = temp_120.apply(lambda x: count_low_probs(x, threshold=0.1))
    df_1['low_prob_count_min'] = temp_121.apply(lambda x: count_low_probs(x, threshold=0.1))

    df_1['mean_grad_max'] = temp_120.apply(mean_gradient)
    df_1['mean_grad_min'] = temp_121.apply(mean_gradient)

    df_1['p25_max'] = temp_120.apply(lambda x: percentile(x, 25))
    df_1['p50_max'] = temp_120.apply(lambda x: percentile(x, 50))
    df_1['p75_max'] = temp_120.apply(lambda x: percentile(x, 75))

    if df_1.shape[0] != df_2.shape[0] :
        raise ValueError("Lengths of dataframes are not same")

    merged_df = pd.concat([df_1, df_2['hallucination']], axis=1)
    
    return merged_df