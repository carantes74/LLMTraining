
# %pip install --upgrade pip
# %pip install --disable-pip-version-check \
#     torch==1.13.1 \
#     torchdata==0.5.1 --quiet

# %pip install \
#     transformers==4.27.2 \
#     datasets==2.11.0  --quiet

### IMPORTS
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

###LOAD HUGGINGFACE SAMPLE DATASETS
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

### SET EXAMPLE INDEXES
example_indices = [40, 200]

### LOAD BASE LLM MODEL
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

### LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

### PROMPT FORMATTER - ADDS STOP SEQUENCE ]\n\n\n
def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        
        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""
    
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    
    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""
        
    return prompt


### N SHOTS PROMPT FORMATT
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200
few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
#print(few_shot_prompt)

### MODEL PARAMETRIZATION
#generation_config = GenerationConfig(max_new_tokens=50)
#generation_config = GenerationConfig(max_new_tokens=10)
generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
#generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
#generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0], 
    skip_special_tokens=True
)

print(f'MODEL GENERATION - FEW SHOT:\n{output}')











