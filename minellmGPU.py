### IMPORTS
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

### LOAD BASE LLM MODEL
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

### LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

### MODEL PARAMETRIZATION
generation_config = GenerationConfig(max_new_tokens=500, temperature=1.0)

### QA LOOP
while (True):
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    query = "Please answer the following question: " + query

    inputs = tokenizer(query, return_tensors='pt').to(device)
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )[0], 
        skip_special_tokens=True
    )

    print(f'Response:\n{output}')












