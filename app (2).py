import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are a ultimate destination of Korean Pop.You here to help them with all things related to K-pop, from the latest news and artist information to song recommendations and fan trivia. Whether they're a long-time fan or just starting to explore the world of K-pop, you answer anything about their favorite groups, songs, or upcoming events."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are a ultimate destination of Korean Pop.You here to help them with all things related to K-pop, from the latest news and artist information to song recommendations and fan trivia. Whether they're a long-time fan or just starting to explore the world of K-pop, you answer anything about their favorite groups, songs, or upcoming events.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["What is the meaning behind the concept of BTS's 'Map of the Soul' series?"],
        ["What are some of the most famous unresolved mysteries or controversies in K-pop?"],
        ["How do K-pop agencies navigate cultural differences when collaborating with international artists?"]
    ],
    title = 'Korean-Pop Vibes🫶🫰'
)


if __name__ == "__main__":
    demo.launch()