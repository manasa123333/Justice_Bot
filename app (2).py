from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_llm():
    """
    Loads the GPT-2 model and tokenizer using the Hugging Face `transformers` library.
    """
    try:
        # Load pre-trained model and tokenizer from Hugging Face
        print("Downloading or loading the GPT-2 model and tokenizer...")
        model_name = 'gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("Model and tokenizer successfully loaded!")
        return model, tokenizer
    
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None

def generate_response(model, tokenizer, user_input):
    """
    Generates a response using the GPT-2 model and tokenizer.
    
    Args:
    - model: The loaded GPT-2 model.
    - tokenizer: The tokenizer corresponding to the GPT-2 model.
    - user_input (str): The input question from the user.

    Returns:
    - response (str): The generated response.
    """
    try:
        inputs = tokenizer.encode(user_input, return_tensors='pt')
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        return f"An error occurred during response generation: {e}"

# Load the model and tokenizer
model, tokenizer = load_llm()

if model is None or tokenizer is None:
    print("Model and/or tokenizer loading failed.")
else:
    print("Model and tokenizer are ready for use.")

import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face API client
# The InferenceClient does not take an api_key argument
client = InferenceClient()

def retrieval_QA_chain(llm, prompt, db):
    # Placeholder function - ensure it integrates as needed
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Handles interaction with the chatbot by sending the conversation history
    and system message to the Hugging Face Inference API.
    """
    print("Starting respond function")
    print("Received message:", message)
    print("Conversation history:", history)

    messages = [{"role": "system", "content": system_message}]
    
    for user_msg, assistant_msg in history:
        if user_msg:
            print("Adding user message to messages:", user_msg)
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            print("Adding assistant message to messages:", assistant_msg)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    print("Final message list for the model:", messages)

    response = ""
    try:
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message['choices'][0]['delta']['content']
            response += token
            print("Token received:", token)
            yield response
    except Exception as e:
        print("An error occurred:", e)
        yield f"An error occurred: {e}"

    print("Response generation completed")

# Set up the Gradio ChatInterface
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are an AI specializing in human rights issues. Provide detailed and passionate answers about the importance of human rights..", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    title="JusticeBot",
    description="Ask questions about human rights, and get informed, passionate answers!"
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()