import io
import re
import sys
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from openai import OpenAI

''' Example usage
file_path = 'electronic_sales_data.csv'
description = "This dataset contains sales transaction records for an electronics company."
key_features = """
Customer ID: Unique identifier for each customer.
Age: Age of the customer (numeric)
Gender: Gender of the customer (Male or Female)
Loyalty Member: (Yes/No) (Values change by time, so pay attention to who cancelled and who signed up)
Product Type: Type of electronic product sold (e.g., Smartphone, Laptop, Tablet)
SKU: a unique code for each product.
Rating: Customer rating of the product (1-5 stars) (Should have no Null Ratings)
Order Status: Status of the order (Completed, Cancelled)
Payment Method: Method used for payment (e.g., Cash, Credit Card, Paypal)
Total Price: Total price of the transaction (numeric)
Unit Price: Price per unit of the product (numeric)
Quantity: Number of units purchased (numeric)
Purchase Date: Date of the purchase (format: YYYY-MM-DD)
Shipping Type: Type of shipping chosen (e.g., Standard, Overnight, Express)
Add-ons Purchased: List of any additional items purchased (e.g., Accessories, Extended Warranty)
Add-on Total: Total price of add-ons purchased (numeric)
"""

question = "What is the average sales amount for each month?"

assistant = DataScienceAssistant(file_path, description, key_features)
assistant.ask_question(question)
'''

class DatasetAnalyzer:
    def __init__(self, file_path, description=None, key_features=None):
        self.file_path = file_path
        self.description = description if description else "No description provided."
        self.key_features = key_features if key_features else "No key features provided."
        self.df = pd.read_csv(file_path)
        
    def generate_description(self):
        columns = f"**Column Names:**\n\n- " + "\n- ".join(list(self.df.columns))
        head = f"**First 2 Rows of the DataFrame:**\n\n" + tabulate(self.df.head(2), headers='keys', tablefmt='pipe')
        
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        info = f"**DataFrame Info:**\n\n```\n{info_str}\n```"
        
        types = f"**Data Types:**\n\n" + tabulate(self.df.dtypes.reset_index(), headers=['Column', 'Data Type'], tablefmt='pipe')
        summary = f"**Summary Statistics:**\n\n" + tabulate(self.df.describe(), headers='keys', tablefmt='pipe')
        null_sum = "**Number of Nulls in Columns:**\n\n" + "\n".join([f"- **{col}**: {count}" for col, count in self.df.isnull().sum().items()])
        dupe_sum = f"**Number of Duplicate Rows:** {self.df.duplicated().sum()}"
        unique_sum = "**Number of Unique Values in Columns:**\n\n" + "\n".join([f"- **{col}**: {count}" for col, count in self.df.nunique().items()])
        
        system = '''
        You are a data scientist tasked to explore a dataset and provide meaningful insights.
        To do this you will use the pandas Python library. Based on the questions you are asked, you will write Python code that should answer that question when you run it.

        Below are details about the dataset that will help you write the code. Consider the filename and column names when writing your response so the code works for this dataset.
        Do not make up column names, only use details that are provided.

        DATASET DETAILS
        ---
        FILENAME:
        {file_path}

        DESCRIPTION:
        {description}

        COLUMN NAMES:
        {columns}

        KEY COLUMNS:
        {key_features}

        HEAD:
        {head}

        SUMMARY:
        {summary}

        INFO:
        {info}

        NUMBER OF NULLS IN COLUMNS:
        {null_sum}

        NUMBER OF DUPLICATES IN COLUMNS:
        {dupe_sum}

        NUMBER OF UNIQUE VALUES IN COLUMNS:
        {unique_sum}

        If you are unable to write code for the given question, do the following:
           1. Explain what would be needed to answer this question.
           2. Give a suggestion on a better question to ask.
        '''
        
        return system.format(
            file_path=self.file_path,
            description=self.description,
            columns=columns,
            key_features=self.key_features,
            head=head,
            summary=summary,
            info=info,
            null_sum=null_sum,
            dupe_sum=dupe_sum,
            unique_sum=unique_sum
        )

class CodeExecutor:
    @staticmethod
    def execute_code(response):
        code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            
            # Capture the output of the code execution
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                exec(code, globals())
                output = sys.stdout.getvalue()
                
                # Check if there are any plots generated
                plt.show()
                
                display(Markdown(f"**Code Executed:**\n\n```python\n{code}\n```"))
                if output:
                    display(Markdown(f"**Output:**\n\n```\n{output}\n```"))
            except Exception as e:
                output = f"Error executing code: {e}"
                display(Markdown(f"**Error:**\n\n```\n{output}\n```"))
            finally:
                sys.stdout = old_stdout
        else:
            display(Markdown("**No Python code found in the response.**"))

class AIHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_response(self, system_message, question):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content

class DataScienceAssistant:
    def __init__(self, file_path, description=None, key_features=None):
        self.analyzer = DatasetAnalyzer(file_path, description, key_features)
        self.ai_handler = AIHandler(api_key=os.environ.get("OPENAI_API_KEY"))

    def ask_question(self, question):
        system_message = self.analyzer.generate_description()
        response_content = self.ai_handler.get_response(system_message, question)
        
        display(Markdown(f"**Question:** {question}"))
        display(Markdown(f"**AI Response:**\n\n{response_content}"))
        
        CodeExecutor.execute_code(response_content)

