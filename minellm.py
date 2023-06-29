### IMPORTS
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

### LOAD BASE LLM MODEL
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

### LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

### MODEL PARAMETRIZATION
generation_config = GenerationConfig(max_new_tokens=500, temperature=1.0)

### QA LOOP
while (True):
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    query = "Please answer the following question:\n" + query

    inputs = tokenizer(query, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )[0], 
        skip_special_tokens=True
    )

    print(f'Response:\n{output}')












