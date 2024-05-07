# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC ### PEFT (Parameter Efficient Fine Tuning Example)

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install --upgrade pip
# MAGIC %pip install --disable-pip-version-check \
# MAGIC     torch==2.0.1 \
# MAGIC     torchdata==0.6.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# MAGIC
# MAGIC %pip install \
# MAGIC     transformers==4.40.0 \
# MAGIC     datasets==2.19.0 \
# MAGIC     evaluate==0.4.0 \
# MAGIC     rouge_score==0.1.2 \
# MAGIC     loralib==0.1.1 \
# MAGIC     peft==0.3.0 \
# MAGIC     accelerate==0.27.2 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python for Databricks
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Dataset and Model Initialization
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can find the complete dataset on HuggingFace [NebulaByte/E-Commerce_Customer_Support_Conversations](https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations)

# COMMAND ----------

# DBTITLE 1,Read Datasets
# Read training and Test Datasets

train_df = spark.read.table("main.vikas_demo.ecomm_train_data")
validate_df = spark.read.table("main.vikas_demo.ecomm_validate_data")

# COMMAND ----------

display(train_df.select("conversation", "summary").take(1))

# COMMAND ----------

# DBTITLE 1,Convert Spark Dataframe to Transformer Dataset
from datasets import Dataset

train_ds = Dataset.from_spark(train_df)
validate_ds = Dataset.from_spark(validate_df)

# COMMAND ----------

train_ds

# COMMAND ----------

# DBTITLE 1,Load the original base model
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Let's find out the number of trainable parameters in the model we have chosen. 
# MAGIC
# MAGIC This could be achieved by looping through all the named paramers and counting the params that **require gradients**.
# MAGIC Below is a utility function that implements this.
# MAGIC

# COMMAND ----------

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"Total Number of model parameters: {all_model_params} \nTrainable model parameters: {trainable_model_params} \nPercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"""

# COMMAND ----------

print(print_number_of_trainable_model_parameters(original_model))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Do we really need to train all model parameters?
# MAGIC
# MAGIC ### NOT NECESSARILY !
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **Parameter Efficient Fine Tuning (PEFT)** is a method used in machine learning, particularly with large-scale pre-trained models like those in natural language processing (NLP), to adapt these models to new tasks without significantly increasing the number of parameters. Unlike traditional fine-tuning approaches that adjust all parameters of a model, PEFT focuses on modifying only a small subset of the model's parameters. This approach identifies and updates the most crucial parameters for the new task, thereby reducing computational and memory requirements, speeding up training times, and minimizing the risk of overfitting. PEFT enables the efficient adaptation of large models to specific tasks with less resource expenditure, making it a practical solution for improving model performance on downstream tasks
# MAGIC

# COMMAND ----------

# DBTITLE 1,Tokenize the training dataset
def tokenize_function(example):
    start_prompt = 'Categorize the sentiment of the customer into one of the following in the below conversation - (neutral, frustrated, positive, negative) \n\n'
    end_prompt = '\n\nSentiment: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["conversation"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=512).input_ids.to('cuda')
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids.to('cuda')
    
    return example

tokenized_datasets = train_ds.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['issue_area', 'issue_category', 'issue_sub_category', 'issue_category_sub_category', 'customer_sentiment', 'summary', 'product_category', 'product_sub_category', 'issue_complexity', 'agent_experience_level', 'agent_experience_level_desc', 'conversation'])

# COMMAND ----------

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets.shape}")

print(tokenized_datasets)

# COMMAND ----------

# MAGIC %md ####LoRA Tunable Knobs####
# MAGIC
# MAGIC Low Rank Adaptation (LoRA) is a library implemented in HUggingFace transformer library that substantially reduces the amount of processing power needed during the training process. 
# MAGIC
# MAGIC Two parameters in the LoraConfig below are quite critical for the training process:
# MAGIC
# MAGIC - **r**: represents the rank of the low rank matrices learned during the finetuning process. As this value is increased, the number of parameters needed to be updated during the low-rank adaptation increases. Low value yields quicker results but the quality of training is compromized while increasing it beyond a certain point also does not give better results. 
# MAGIC
# MAGIC - **target_modules**: The target_modules array in a LoRAConfig object specifies which parts of the model's architecture are to be adapted using low-rank matrices. The more modules you decide to train, the more computationally expensive it becomes.

# COMMAND ----------

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=40, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

# COMMAND ----------

peft_model = get_peft_model(original_model, 
                            lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

# COMMAND ----------

output_dir = f'/local_disk0/peft-conversation-categorize-training-{str(int(time.time()))}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=3,
    logging_steps=1,
    max_steps=20,
    per_device_train_batch_size=1, # Reduce batch size
    gradient_accumulation_steps=2, # Effectively increase overall batch size    
    weight_decay=0.01,
    save_total_limit=2,
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Start Model Training with PEFT Trainer ####
# MAGIC
# MAGIC A gradual decrease in the training loss during the fine-tuning of a Large Language Model (LLM) indicates that the model is learning and improving its performance on the training dataset over time. It means that the model's predictions are getting closer to the actual target values, leading to a reduction in the error between the predicted and true outputs. 

# COMMAND ----------

# MAGIC %sh rm -rf /local_disk0/peft-conversation-categorize-checkpoint-local

# COMMAND ----------

peft_trainer.train()

peft_model_path="/local_disk0/peft-conversation-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# COMMAND ----------

def dash_line():
  return "==========================================================================="

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Evaluating model with ROUGE metric
# MAGIC
# MAGIC ROUGE metric is a set of evaluation tools used primarily to assess the quality of text generated by models. It measures the similarity between the machine-generated text and a set of reference texts, which are typically human-generated.

# COMMAND ----------


dialogues = validate_ds['conversation'][0:30]
human_baseline_summary = validate_ds['summary'][0:30]

original_model_summary = []
peft_model_summary = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the below conversation

conversation::
{dialogue}

Summary:: 
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    human_baseline_text_output = human_baseline_summary[idx]
    
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=512))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=512))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_summary.append(original_model_text_output)
    peft_model_summary.append(peft_model_text_output)

zipped_summaries = list(zip(human_baseline_summary, original_model_summary, peft_model_summary))
 
evaluation = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summary', 'original_model_summary', 'peft_model_summary'])
evaluation

# COMMAND ----------

rouge = evaluate.load('rouge')

original_model_results = rouge.compute(
    predictions=original_model_summary,
    references=human_baseline_summary[0:len(original_model_summary)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summary,
    references=human_baseline_summary[0:len(peft_model_summary)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)

print('PEFT MODEL:')
print(peft_model_results)

# COMMAND ----------

print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

improvement = np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values()))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {(value*100):.2f}%')

# COMMAND ----------

index =120

dialogue = validate_ds['conversation'][index]
summary = validate_ds['summary'][index]

prompt = f"""
Summarize the below conversation between a customer service agenat and a customer


conversation::
{dialogue}

Summary:: 
"""

print(prompt)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
original_model = original_model.to('cuda')

original_model_outputs = original_model.generate(input_ids=input_ids.to('cuda'), generation_config=GenerationConfig(max_new_tokens=200, temperature = 0.0)) #num_beams=1
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.to('cuda').generate(input_ids=input_ids.to('cuda'), generation_config=GenerationConfig(max_new_tokens=200, temperature = 0.0))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(dash_line())
print(f'BASELINE HUMAN SENTIMENT:\n{summary}')
print(dash_line())
print(f'ORIGINAL MODEL:\n{original_model_text_output}')

print(dash_line())
print(f'PEFT MODEL: {peft_model_text_output}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### How to register the model to Unity Catalog ###

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig

peft_model_id = "/local_disk0/peft-conversation-summary-checkpoint-local"
config = PeftConfig.from_pretrained(peft_model_id)

from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)



# COMMAND ----------

import mlflow
import transformers

class FlanT5FineTuned(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto")

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [512])[0]

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda")
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0])

        return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["Translate the complete sentence to German:  My name is Vikas and I live in Mumbai."], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=FlanT5FineTuned(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "einops","sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Register model to Unity Catalog

registered_name = "main.vikas_demo.diday-flan-t5-finetuned"  
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Choose the right model version registered in the above cell.
client.set_registered_model_alias(name=registered_name, alias="beta", version=result.version)

# COMMAND ----------


import mlflow

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@beta")
loaded_model.predict(
    {
        "prompt": "Translate complete sentence to German: My name is Vikas and I live in Mumbai",
        "temperature": 0.5,
        "max_tokens": 100,  # Corrected parameter name from 'max_new_tokens' to 'max_tokens'
    }
)

# COMMAND ----------


