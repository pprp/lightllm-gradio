import time
import requests
import json
import argparse
import json 
import re 
# compute kendall tau
from scipy.stats import kendalltau
from tqdm import tqdm

def extract_before_second_dot(text):
    # Find the indices of all occurrences of '.'
    dot_indices = [i for i, char in enumerate(text) if char == '.']

    # Check if there are at least two dots
    if len(dot_indices) >= 2:
        # Extract the text up to the second dot
        return text[:dot_indices[1] + 1]
    else:
        # If there are fewer than two dots, return the original text
        return text

def extract_number(text):
    # Define a regular expression pattern to match a number following "valid accuracy"
    pattern = r"(\d+\.\d+)"
    match = re.search(pattern, text)

    if match:
        # Extract the number
        number = match.group(1)
        return float(number)
    else:
        # Return None if no match is found
        return None

def query_and_get_results(prompt):
    System = "As an expert in Neural Architecture Search, your task is to evaluate the performance of a given neural architecture. " \
         "Please focus solely on providing detailed estimates of its accuracy, flops, params, and loss. " \
         "Answer the question in just one sentence and just stop. Avoid addressing any content beyond these performance metrics. " \
         "Do not include any analysis, discussion, or other information beyond the requested metrics. "
    headers = {'Content-Type': 'application/json'}
    if not prompt.endswith('### Agent:'):
        prompt += '### Agent:'
    data = {
        'inputs': System + prompt,
        "parameters": {
            'do_sample': False,
            'ignore_eos': False,
            'max_new_tokens': 100,
            'stop_sequences': '###',
        }
    }

    response = requests.post(args.model_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        res = response.json()['generated_text'][0].split('###')[0]
        res = extract_before_second_dot(res)
        return res
    else:
        return None 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-url",
                        type=str,
                        default="http://localhost:8080/generate")
    args = parser.parse_args()

    json_file = './nb201_finetune.json'
    with open(json_file) as f:
        data = json.load(f)["instances"]
    
    gt_list, pd_list = [], []

    keys = "valid accuracy"
    for _dict in tqdm(data):
        if keys[0] in _dict['input']:
            # just conduct one experiment
            prompt = _dict['input']
            res = query_and_get_results(prompt)
            if res is None:
                continue 
            _pred = extract_number(res)
            _gt = extract_number(_dict['output'])
            if _pred is None or _gt is None:
                continue
            gt_list.append(_gt)
            pd_list.append(_pred)

    # compute kendall tau
    tau, p_value = kendalltau(gt_list, pd_list)
    print('Kendall tau: ', tau)
    print('p-value: ', p_value)
    
    # save the results
    re_dict = {}
    re_dict['gt'] = gt_list
    re_dict['pd'] = pd_list
    with open('results.json', 'w') as f:
        json.dump(re_dict, f)
    
    # plot correlation 
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    ax = sns.violinplot(x=gt_list, y=pd_list)
    plt.savefig('violinplot.png')



            


