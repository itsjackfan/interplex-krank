# utils.py

from typing import Dict, Any
import re
import json
import logging
from functools import lru_cache
from config import Config
from passlib.context import CryptContext

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

# Removed parse_tree_of_thoughts function as RAG functionality is no longer needed

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

from bson import ObjectId
from typing import Any, Dict

def convert_objectid_to_str(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert ObjectId fields in a MongoDB document to string.
    """
    if not isinstance(data, dict):
        return data
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, ObjectId):
            converted_data[key] = str(value)
        elif isinstance(value, list):
            converted_data[key] = [convert_objectid_to_str(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            converted_data[key] = convert_objectid_to_str(value)
        else:
            converted_data[key] = value
    return converted_data
