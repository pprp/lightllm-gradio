import argparse
import json
import requests
import gradio as gr

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

def http_bot(prompt):
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
        yield res
    else:
        yield 'Error: ' + str(response.status_code) + ' - ' + response.text


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# NASBenchGPT text completion demo\n")
        inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")
        outputbox = gr.Textbox(label="Output", placeholder="Generated result from the model")
        inputbox.submit(http_bot, inputs=inputbox, outputs=outputbox)
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-url",
                        type=str,
                        default="http://localhost:8080/generate")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)
