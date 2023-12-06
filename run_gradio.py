import argparse
import json
import requests
import gradio as gr

def http_bot(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {
        'inputs': prompt,
        "parameters": {
            'do_sample': False,
            'ignore_eos': False,
            'max_new_tokens': 100,
        }
    }

    response = requests.post(args.model_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        yield response.json()['generated_text'][0]
    else:
        yield 'Error: ' + str(response.status_code) + ' - ' + response.text


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM text completion demo\n")
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
