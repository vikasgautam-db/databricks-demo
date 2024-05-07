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
registered_name = "main.vikas_demo.diday-flan-t5-base"

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@beta")

# COMMAND ----------

def prompting_demo(input_text, human_sentiment):

    model_result = loaded_model.predict(
    {
        "prompt": input_text,
        "temperature": 0.01,
        "max_tokens": 512, 
    }
    )
    response = remove_html_tags(model_result)

    return response

# COMMAND ----------

# gradio app

import gradio as gr

demo = gr.Interface(fn=prompting_demo,
                    inputs=[gr.Textbox(label="Input Conversation"), gr.Textbox(label="Human Assigned Sentiment")],
                    outputs=[gr.Textbox(label="Model Generated Sentiment")],
                    title="Prompting with Base Model", 
                    allow_flagging="never"
                    )
demo.launch(share=True)

# COMMAND ----------

demo.close()
