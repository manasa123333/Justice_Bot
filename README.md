
# Project Title

JusticeBot is an AI chatbot designed to provide detailed and passionate responses about human rights issues. Built using GPT-2 and fine-tuned on human rights-related texts, it aims to educate and advocate for the importance of human rights through intelligent conversations.


## Demo

[Justice Bot Demo](https://huggingface.co/spaces/Manasa1/Justice_Bot)


## Features

- **Human Rights Advocacy:** Provides insightful responses about human rights topics.
- **Interactive Chat Interface:** Engage in conversations using Gradio.
- **Customizable Settings:** Adjust parameters such as max tokens, temperature, and top-p.


## Installation

1. **Clone the Repository**

```bash
  git clone https://github.com/manasa123333/Justice_Bot.py
  cd Justice_Bot
```

2. **Install Dependencies**

Ensure you have Python installed. Install the required packages using pip:

```bash
pip install -r requirements.txt
```


    
## Set Up

1. **Prepare Your PDF Dataset**

Place your PDF file in the project directory and name it human_rights.pdf.

2. **Create the Vector Database**

Run the create_vector_db.py script to process the PDF and create a FAISS vector store:

```bash
python create_vector_db.py
```

This will load the PDF, split the text, generate embeddings, and save the FAISS vector store to vectorstores/db_faiss.

3. **Fine-Tuning GPT-2 (Optional)**

If you wish to fine-tune GPT-2 on your dataset, use the fine_tune_gpt2.py script (not provided here but recommended for fine-tuning).

4. **Run the Chatbot**

Launch the Gradio interface to interact with the chatbot:

```bash
python app.py
```
## Usage

1. **System Message**: Customize the behavior of the chatbot by modifying the system message in the Gradio interface.
2. **Adjust Settings**: Fine-tune response generation by adjusting parameters such as max tokens, temperature, and top-p in the Gradio interface.




## Deployment

1. **Deploy to Hugging Face Spaces**

Follow these steps to deploy your chatbot to Hugging Face Spaces:

Create a new Space on Hugging Face.

Push the code along with requirements.txt and README.md:

```bash
  git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://huggingface.co/spaces/your_space_name
git push -u origin main

```
2. **Run the App on Hugging Face**

Once deployed, you can access and interact with JusticeBot directly on Hugging Face Spaces.

## Contributing

Contributions to enhance JusticeBot are welcome! Please submit issues and pull requests via GitHub.



