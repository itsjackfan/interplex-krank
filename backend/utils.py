# utils.py

from typing import Dict, Any
import re
import json
import logging
from functools import lru_cache
from config import Config

logger = logging.getLogger(__name__)

class LightweightTextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.punct_pattern = re.compile(r'[^\w\s]')

    def clean_text(self, text: str) -> str:
        text = self.url_pattern.sub('[URL]', text)
        text = ' '.join(text.split())
        return text.strip()

    def process_text(self, text: str) -> str:
        cleaned_text = self.clean_text(text)
        processed_text = self.punct_pattern.sub(' ', cleaned_text)
        processed_text = ' '.join(processed_text.split())
        return processed_text

preprocessor = LightweightTextPreprocessor()

@lru_cache(maxsize=10000)
def preprocess_for_embedding(text: str) -> str:
    return preprocessor.process_text(text)

def clean_llm_output(output: str) -> str:
    output = re.sub(r'```json\s*', '', output)
    output = re.sub(r'```', '', output)
    output = output.strip()
    output = output.replace('\\n', '\n')
    return output

def fix_common_json_errors(json_str: str) -> str:
    json_str = re.sub(r'}\s*{', '}, {', json_str)
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str

def parse_tree_of_thoughts(output: str) -> Dict[str, Any]:
    try:
        cleaned_output = clean_llm_output(output)
        corrected_output = fix_common_json_errors(cleaned_output)
        tree_of_thoughts = json.loads(corrected_output)
        return tree_of_thoughts
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.debug(f"Failed JSON string: {corrected_output}")
        raise ValueError(f"Unable to parse JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        raise ValueError(f"Unexpected error: {e}")
