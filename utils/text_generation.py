from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


def generate(input_str):
    prompt = PromptTemplate.from_template("{input_str}")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    chain = LLMChain(prompt=prompt, llm=llm)
    message = chain.run(input_str)
    return message.strip()


def get_rating(x):
    """
    Extracts a rating from a string.
    
    Args:
    - x (str): The string to extract the rating from.
    
    Returns:
    - int: The rating extracted from the string, or None if no rating is found.
    """
    nums = [int(i) for i in re.findall(r'\d+', x)]
    if len(nums) > 0:
        return min(nums)
    else:
        return None
