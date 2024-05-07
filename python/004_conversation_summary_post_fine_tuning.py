# Databricks notebook source
# MAGIC %pip install transformers gradio --quiet
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, "", text)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Get base model
base_model_name = "main.vikas_demo.diday-flan-t5-base"

original_model = mlflow.pyfunc.load_model(f"models:/{base_model_name}@beta")

# COMMAND ----------



finetuned_model_name = "main.vikas_demo.diday-finetuned"
finetuned_model = mlflow.pyfunc.load_model(f"models:/{finetuned_model_name}@beta")

# COMMAND ----------

def prompting_demo(input_text):
    response = []
    original_model_result = original_model.predict(
        {
        "prompt": input_text,
        "temperature": 1.0,
        "max_tokens": 512, 
        }
    )
    original_model_response = remove_html_tags(original_model_result)
    
    finetuned_model_result = finetuned_model.predict(
    {
        "prompt": input_text,
        "temperature": 1.0,
        "max_tokens": 512, 
    }
    )
    finetuned_model_response = remove_html_tags(finetuned_model_result)

    response.append(original_model_response)
    response.append(finetuned_model_response)
    
    print(f"original_model_response:: {original_model_response}" )
    print(f"finetuned_model_response:: {finetuned_model_response}")

    return response

# COMMAND ----------

print(prompting_demo("""
               Agent: Hello, thank you for contacting BrownBox customer support. My name is John. How may I assist you today?

Customer: Hi, I recently placed an order for a smartwatch on your website, but I need to cancel it. Can you help me with that?

Agent: Sure, I can help you with that. May I have your order number, please?

Customer: Yes, it's #BB789012.

Agent: Thank you for providing your order number. Let me check the details for you. Please hold for a moment.

[Agent puts the customer on hold for a few minutes]

Agent: Thank you for waiting. I have checked your order details, and I can see that the order is still in processing status. I can cancel the order for you right away. However, I would like to inform you that it may take up to 24 hours for the cancellation to reflect in your account. Is that okay with you?

Customer: Yes, that's fine.

Agent: Alright. I have initiated the cancellation process for your order. You will receive an email confirmation shortly. Is there anything else I can assist you with?

Customer: No, that's all. Thank you for your help.

Agent: You're welcome. Please feel free to contact us if you have any further queries or issues. Have a great day!

Customer: You too. Goodbye.

Agent: Goodbye!
               """))

# COMMAND ----------

# gradio app

import gradio as gr

demo = gr.Interface(fn=prompting_demo,
                    inputs=[gr.Textbox(label="Input Conversation", lines = 20)],
                    outputs=[gr.Textbox(label="Original Model Summary"), gr.Textbox(label="Finetuned Model Summary")],
                    title="Demo Finetuned Model", 
                    allow_flagging="never"
                    )
demo.launch(share=True)

# COMMAND ----------

 # demo.close()

# COMMAND ----------


