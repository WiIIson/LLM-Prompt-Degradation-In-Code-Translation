import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import numpy as np
from openai import OpenAI
import pandas as pd
import random
import replicate
import tensorflow_hub as hub
import torch

# -----------------------------------------------------------------------------

# LLM tokens
API = pd.read_excel('API.xlsx')
llama_token = list(API['Token'])[0]
gpt_token = list(API['Token'])[1]

# LLM versions
v_llama = "meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736"
v_gpt = "gpt-4"

# Load prompts
prompts = pd.read_excel("prompts.xlsx")

# Load Universal Sentence Encoder
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# -----------------------------------------------------------------------------

def gpt_response(message):
    """
    Generate a response from ChatGPT

    Parameters
    ----------
    message : str
        A message to send to ChatGPT.

    Returns
    -------
    str
        A response message recieved from ChatGPT.

    """
    client_gpt = OpenAI(api_key=gpt_token)
    response = client_gpt.chat.completions.create(
        messages = [
            {"role": "user", "content": message}
        ],
        model = v_gpt
    )
    return response.choices[0].message.content

def llama_response(message):
    """
    Generate a response from CodeLLAMA

    Parameters
    ----------
    message : str
        A message to send to CodeLLAMA.

    Returns
    -------
    str
        A response message recieved from CodeLLAMA.

    """
    client_llama = replicate.Client(api_token=llama_token)
    llama_input = {
        "top_k": 250,
        "prompt": message,
        "temperature": 0.95
    }
    response = client_llama.run(v_llama, input=llama_input)
    return "".join(response)

def get_random_prompt(prompts):
    """
    Returns a random prompt from a list of prompts

    Parameters
    ----------
    prompts : list[str]
        A list of prompts to choose from.

    Returns
    -------
    str
        A prompt to use with an LLM.

    """
    num = random.randint(0, len(prompts.index)-1)
    return prompts.iloc[[num]]

# -----------------------------------------------------------------------------

def run_simulation(num_loops=10, num_repeats=5, reverse=False):
    """
    Runs a simulation loop.

    Parameters
    ----------
    num_loops : int, optional
        Number of times to pass input between LLMs. The default is 10.
    num_repeats : int, optional
        Number of times to repeat the operation to get an average,. The default
        is 5.
    reverse : bool, optional
        Toggle to swap roles of LLMs. The default is False.

    Returns
    -------
    df_log : pd.DataFrame
        Log of output from LLMs.

    """
    
    df_log = pd.DataFrame()
    
    # Pre-prompt message to describe LLM task
    pre_desc = "You are tasked with creating a Python function based on the following description. Only write code, do not include any descriptions in your response. "
    pre_code = "You are tasked with describing the following Python function. Structure the description in a single paragraph and do not provide example code. "
    
    # Repeat the test multiple times to get the average
    for i in range(num_repeats):
        print('repeat: ' + str(i))
        prompt = get_random_prompt(prompts)
        desc_msg = prompt['prompt'].to_string(index=False)
        prompt_num = prompt['task_id'].to_string(index=False)[-3:]
        
        # Construct initial description message
        gpt_log = []
        llama_log = []
        
        # Run loop, create dataframe log of runs
        for j in range(num_loops):
            print('\tloop: ' + str(j))
            
            # Account for if roles are reversed
            if not reverse:
                code_msg = gpt_response(pre_desc + desc_msg)
                gpt_log.append(code_msg)
                desc_msg = llama_response(pre_code + code_msg)
                llama_log.append(desc_msg)
            else:
                code_msg = llama_response(pre_desc + desc_msg)
                llama_log.append(code_msg)
                desc_msg = gpt_response(pre_code + code_msg)
                gpt_log.append(desc_msg)
        
        # Append logs to the dataframe - account for if roles are reversed
        if not reverse:
            df_log[str(i) + '_' + prompt_num + '_' + 'gpt'] = gpt_log
            df_log[str(i) + '_' + prompt_num + '_' + 'llama'] = llama_log
        else:
            df_log[str(i) + '_' + prompt_num + '_' + 'llama'] = llama_log
            df_log[str(i) + '_' + prompt_num + '_' + 'gpt'] = gpt_log
    
    return df_log

# -----------------------------------------------------------------------------

def lev_dist(s1, s2):
    """
    Calculate the Levenshtein distance between two inputs

    :param s1: First input
    :param s2: Second input
    :return: Levenshtein distance
    """
    len_s1, len_s2 = len(s1), len(s2)
    
    # Initialize a matrix to store distances
    dp = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
    
    # Base cases
    for i in range(len_s1 + 1):
        dp[i][0] = i  # Cost of deleting characters
    for j in range(len_s2 + 1):
        dp[0][j] = j  # Cost of inserting characters
    
    # Fill the matrix
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0  # No operation needed
            else:
                cost = 1  # Substitution cost
            
            dp[i][j] = min(
                dp[i - 1][j] + 1,    # Deletion
                dp[i][j - 1] + 1,    # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    
    return dp[len_s1][len_s2]

def cosine_sim(first, last):
    """
    Calculate the cosine similarity of two strings using the Universal Sentence
    Encoder for semantic intent.

    Parameters
    ----------
    first : str
        First string to compare, the baseline string.
    last : str
        Second string to compare, the modified string.

    Returns
    -------
    float
        Percentage cosine similarity.

    """
    
    # Get embeddings for both texts
    embedding1 = embed([first])
    embedding2 = embed([last])
    
    # Convert embeddings to arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Convert numpy arrays to PyTorch tensors
    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)
    
    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2)
    return cosine_sim.item()

def bleu_score(first, last):
    """
    Calculate the BLEU score for two strings.

    Parameters
    ----------
    first : str
        First string to compare.
    last : str
        Second string to compare.

    Returns
    -------
    score : float
        BLEU score.

    """
    hypothesis = last.strip().split()
    reference = first.strip().split()
    
    weights = (0.25, 0.25, 0, 0)

    score = sentence_bleu([reference], hypothesis, weights)
    return score

# -----------------------------------------------------------------------------

def summary_score(filepath):
    """
    Calculates the summary metrics for an output file

    Parameters
    ----------
    filepath : str
        String filepath to an output file from the simulation.

    Returns
    -------
    results : pd.DataFrame
        A dataframe containing summary metrics.

    """
    
    df = pd.read_excel(filepath)
    results = pd.DataFrame()
    results['Metrics'] = ['Levenshtein', 'Cosine', 'BLEU']
    for col in df.columns:
        # Caluclate metrics
        dat = list(df[col])
        first = dat[0]
        last = dat[-1]
        cLEV = lev_dist(first, last)
        cCOS = cosine_sim(first, last)
        cBLEU = bleu_score(first, last)
        results[col] = [cLEV, cCOS, cBLEU]
    return results

def baseavg_score(filepath):
    """
    Calculate the baseline average from an output file.
    This measures the amount of change at each iteration from the baseline
    (first iteration).

    Parameters
    ----------
    filepath : str
        String filepath to an output file from the simulation.

    Returns
    -------
    df_lev : pd.DataFrame
        Pandas Dataframe containing Levenshtein distances.
    df_cos : pd.DataFrame
        Pandas Dataframe containing Cosine similarities.
    df_bleu : pd.DataFrame
        Pandas Dataframe containing BLEU scores.

    """
    
    df = pd.read_excel(filepath)
    df_lev = pd.DataFrame()
    df_cos = pd.DataFrame()
    df_bleu = pd.DataFrame()
    for col in df.columns:
        # Caluclate cumulative metrics
        dat = list(df[col])
        levs = []
        coss = []
        bleus = []
        for i in range(len(dat) - 1):
            cLEV = lev_dist(dat[0], dat[i+1])
            cCOS = cosine_sim(dat[0], dat[i+1])
            cBLEU = bleu_score(dat[0], dat[i+1])
            levs.append(cLEV)
            coss.append(cCOS)
            bleus.append(cBLEU)
        df_lev[col] = levs
        df_cos[col] = coss
        df_bleu[col] = bleus
    return df_lev, df_cos, df_bleu

def avgvar_score(filepath):
    """
    Calculate the average variance from an output file.
    This measures the amount of change between iterations.

    Parameters
    ----------
    filepath : str
        String filepath to an output file from the simulation.

    Returns
    -------
    df_lev : pd.DataFrame
        Pandas Dataframe containing Levenshtein distances.
    df_cos : pd.DataFrame
        Pandas Dataframe containing Cosine similarities.
    df_bleu : pd.DataFrame
        Pandas Dataframe containing BLEU scores.

    """
    
    df = pd.read_excel(filepath)
    df_lev = pd.DataFrame()
    df_cos = pd.DataFrame()
    df_bleu = pd.DataFrame()
    for col in df.columns:
        # Caluclate cumulative metrics
        dat = list(df[col])
        levs = []
        coss = []
        bleus = []
        for i in range(len(dat) - 1):
            cLEV = lev_dist(dat[i], dat[i+1])
            cCOS = cosine_sim(dat[i], dat[i+1])
            cBLEU = bleu_score(dat[i], dat[i+1])
            levs.append(cLEV)
            coss.append(cCOS)
            bleus.append(cBLEU)
        df_lev[col] = levs
        df_cos[col] = coss
        df_bleu[col] = bleus
    return df_lev, df_cos, df_bleu

# -----------------------------------------------------------------------------

def plot_cumsum(filepath, metric, llm):
    """
    Generates a plot from an output file

    Parameters
    ----------
    filepath : str
        String filepath to an output file.
    metric : str
        Metric being analyzed, should be 'Lev', 'Cos' or 'BLEU.
    llm : Str
        Name of the LLM being analyzed, should be 'GPT' or 'LLAMA'.

    Returns
    -------
    None.

    """
    
    # Read in file
    df = pd.read_excel(filepath)
    cols = df.columns
    colGPT = []
    colLLAMA = []
    
    # Split columns by LLM
    for col in cols:
        if col.split("_")[2]==llm.lower():
            colGPT.append(col)
        else:
            colLLAMA.append(col)
    
    # Plot lines and set colors
    colors = ['#ccece6','#99d8c9','#66c2a4','#2ca25f','#006d2c']
    xs = []
    ys = []
    for i in range(len(colGPT)):
        col = colGPT[i]
        plt.plot(list(df[col]), color=colors[i])
        # Record points for average line
        ys += list(df[col])
        xs += list(range(2,11))
    
    # Fit average line
    xs = np.array(xs)
    ys = np.array(ys)
    a, b = np.polyfit(np.array(xs), np.array(ys), 1)
    plt.plot(np.array(range(2,11)), a*np.array(range(2,11))+b, color='red')
    
    # Set plot info based on metric
    mlabel = ''
    if metric == 'Lev':
        mlabel = 'Levenshtein Distance'
    elif metric == 'Cos':
        mlabel = 'Cosine Similarity'
    elif metric == 'BLEU':
        mlabel = 'BLEU Score'
    if metric != 'Lev':
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 1000)
    plt.xlabel('Loop number')
    plt.ylabel(mlabel)
    plt.legend(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5'])
    plt.title(mlabel + ' Over Each Trial of ' + llm.upper())
    plt.show()
    return

# -----------------------------------------------------------------------------

# Run loop
#df = run_simulation(reverse=True)
#df.to_excel('dLLAMA_cGPT.xlsx', sheet_name='results', index=False)

#df = summary_score('Outputs/cLLAMA_dGPT.xlsx')
#df.to_excel('cLLAMA_dGPT_Results.xlsx', index=False)

#df = summary_score('Outputs/cLLAMA_dGPT.xlsx')
#df.to_excel('cLLAMA_dGPT_Results.xlsx', index=False)

#convert_file('Outputs/cGPT_dLLAMA.xlsx')
#df_lev, df_cos, df_bleu = avgvar_score('Outputs/cLLAMA_dGPT.xlsx')
#df_lev.to_excel('Metrics/cLLAMA_dGPT_Lev.xlsx',index=False)
#df_cos.to_excel('Metrics/cLLAMA_dGPT_Cos.xlsx',index=False)
#df_bleu.to_excel('Metrics/cLLAMA_dGPT_BLEU.xlsx',index=False)

#plot_cumsum('Metrics/cLLAMA_dGPT_BLEU.xlsx', 'BLEU', 'gpt')

#df_lev, df_cos, df_bleu = avgvar_score('Outputs/cLLAMA_dGPT.xlsx')
#df_cos.to_excel('cLLAMA_dGPT_cos_thorough.xlsx')

#df_lev, df_cos, df_bleu = avgvar_score('Outputs/cGPT_dLLAMA.xlsx')
#df_cos.to_excel('cGPT_dLLAMA_cos_thorough.xlsx')