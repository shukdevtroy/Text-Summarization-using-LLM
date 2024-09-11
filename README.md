# Text-Summarization-using-LLM

This repository provides a tool for summarizing text using various LLaMA models with Hugging Face's Transformers library. It utilizes state-of-the-art models for natural language processing to generate concise summaries from provided text.

## Features

- **Text Summarization**: Generates a summary of input text using language models.
- **Flexible Model Selection**: Supports various pre-trained models from Hugging Face.
- **Customizable Generation Parameters**: Allows configuration of temperature, max length, top-k sampling, and more.

## Requirements

- Python 3.x
- PyTorch
- Transformers
- LangChain
- AutoGPTQ
- Other dependencies

You can install the required packages with the following commands:

```bash
pip install -q -U transformers peft accelerate optimum
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
pip install langchain
pip install einops
pip install langchain-community
```

## Setup

1. **Import Necessary Libraries**

   The script begins by importing required libraries and modules:

   ```python
   import time
   import numpy as np
   import math
   from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
   import torch
   import transformers
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from langchain import LLMChain, HuggingFacePipeline, PromptTemplate
   ```

2. **Configuration**

   Set up your configuration for the models and parameters:

   ```python
   config = {
       "model_id": [
           "TheBloke/Llama-2-7b-Chat-GPTQ",
           "TheBloke/Llama-2-7B-AWQ",
           "TheBloke/Llama-2-7B-GGUF",
           "TheBloke/Llama-2-7B-GGML",
           "TheBloke/Llama-2-7B-fp16",
           "TheBloke/Llama-2-7B-GPTQ",
           "TheBloke/llama-2-7B-Guanaco-QLoRA-AWQ",
           "TheBloke/Llama-2-7B-AWQ"
       ],
       "hf_token": "...",
       "model": {
           "temperature": 0.7,
           "max_length": 3000,
           "top_k": 10,
           "num_return": 1
       }
   }
   ```

3. **Model Initialization**

   Load the model and tokenizer:

   ```python
   def generate_model(model_id, config):
       print(f"Setting up model {model_id}")
       model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True,
                                 device_map='auto', trust_remote_code=True)
       tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                 device_map='auto', trust_remote_code=True)
       return model, tokenizer

   model, tokenizer = generate_model(config['model_id'][0], config)
   ```

4. **Parameter Count**

   Display the number of parameters in the model:

   ```python
   def call_parameter(model):
       pytorch_total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       untrainable_params = pytorch_total_params - trainable_params
       print(f'Model {model.__class__.__name__} has {pytorch_total_params} parameters in total\n'\
             f'Trainable parameters: {trainable_params}\nUntrainable parameters: {untrainable_params}')
       return pytorch_total_params

   no_params = call_parameter(model)
   ```

5. **Agent and Generator Classes**

   Define classes for the agent and text generation:

   ```python
   class Agent:
       def __init__(self, model, tokenizer):
           self.model = model
           self.tokenizer = tokenizer

       def __repr__(self):
           return f'Model {self.model.__class__.__name__}'

   class Generator:
       def __init__(self, config, agent, template):
           self.agent = agent
           pipeline = transformers.pipeline(
               "text-generation",
               model=self.agent.model,
               tokenizer=self.agent.tokenizer,
               torch_dtype=torch.bfloat16,
               trust_remote_code=True,
               device_map="auto",
               max_length=config['model']['max_length'],
               do_sample=True,
               top_k=config['model']['top_k'],
               num_return_sequences=config['model']['num_return'],
               pad_token_id=tokenizer.eos_token_id
           )
           llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': config['model']['temperature']})
           prompt = PromptTemplate(template=template, input_variables=["text"])
           self.llm_chain = LLMChain(prompt=prompt, llm=llm)

       def generate(self, text):
           result = self.llm_chain.invoke(text)
           return result
   ```

6. **Define Template and Generate Summary**

   Create a summary template and generate a summary for sample text:

   ```python
   template = """
       Write a summary of the following text delimited by triple backticks.
       Return your response which covers the key points of the text.
       ```{text}```
       SUMMARY:
   """

   agent = Agent(model, tokenizer)
   llm_agent = Generator(config, agent, template)

   text = """
     AI is also playing an increasingly important role in sustainability efforts, with some tech giants like Microsoft, Google, and IBM leveraging AI to reduce climate harms.
     ...
   """

   summary = llm_agent.generate(text)
   print(summary)
   ```

## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shukdevtroy/Text-Summarization-using-LLM.git
   cd Text-Summarization-using-LLM
   ```

2. **Install Dependencies**

   ```bash
   pip install -q -U transformers peft accelerate optimum
   pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
   pip install langchain
   pip install einops
   pip install langchain-community
   ```

3. **Run the Script**

   Modify the script with your desired configuration and run it to generate summaries.

## Contributing

Feel free to submit issues or pull requests. Please ensure your contributions adhere to the coding standards and include appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This `README.md` provides a comprehensive overview of the project, installation instructions, and usage details to help users understand and work with the code effectively.
