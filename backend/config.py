# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    # -------------------- Logging Configuration --------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # e.g., DEBUG, INFO, WARNING, ERROR

    # -------------------- FastAPI Configuration --------------------
    APP_TITLE = os.getenv("APP_TITLE", "Article Management API with LLM Features")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", 8000))

    # -------------------- MongoDB Configuration --------------------
    MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "mongodb")   
    MONGODB_PASS = os.getenv("MONGODB_PASS", "password")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "articles")  # Updated from 'notes' to 'articles'

    # -------------------- LLM Configuration --------------------
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")  # Ensure this is set in .env
    GENIUS_MODEL = os.getenv("GENIUS_MODEL", "gemini-1.5-flash-8b")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # -------------------- Similarity Thresholds --------------------
    SIMILARITY_THRESHOLD_ARTICLE = float(os.getenv("SIMILARITY_THRESHOLD_ARTICLE", 0.1))  # Renamed from 'SIMILARITY_THRESHOLD_NOTE'
    SIMILARITY_THRESHOLD_CLUSTER = float(os.getenv("SIMILARITY_THRESHOLD_CLUSTER", 0.2))
    SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS = float(os.getenv("SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS", 0.3875))

    # -------------------- DBSCAN Parameters --------------------
    DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", 1.1725))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 2))

    # -------------------- PageRank Configuration --------------------
    PAGERANK_ALPHA = float(os.getenv("PAGERANK_ALPHA", 0.85))

    # -------------------- SentenceTransformer Model --------------------
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

    # *** Prompt Templates ***
    REFINE_QUERY_PROMPT_TEMPLATE = os.getenv("REFINE_QUERY_PROMPT_TEMPLATE",
        """
You are assisting in refining a search query to generate more accurate and relevant search results. The goal is to explore multiple strategies for query refinement, each focusing on different aspects such as specificity, broader scope, and different keyword combinations. You will apply the Tree of Thoughts (ToT) technique to provide multiple branches of query refinement, allowing the user to select the most suitable version or combine elements from various approaches.

#ROLE:
Your role is to act as a search refinement strategist, using Tree of Thoughts to break down the search query into different thought pathways. Each pathway will represent a unique approach to improving the query, either by focusing on a different search intention, varying keyword specificity, or introducing Boolean operators and modifiers.

#RESPONSE GUIDELINES:

Generate distinct branches refining the query, thinking about:
- narrowing the query by making it more specific, using detailed keywords and refining the scope.
- broadening the query to explore more diverse or general results, potentially increasing the range of topics covered.
- keyword optimization and advanced search techniques, such as Boolean operators (AND, OR, NOT), quotations, or search filters (e.g., date ranges, geographic regions).
In each branch, explain the reasoning behind the refinement and how it impacts the search results:

Highlight why specific terms were added, removed, or modified.
Discuss how each branch addresses a different aspect of the searcher's goals (e.g., relevance, comprehensiveness, or filtering results).
Provide a final refined query for each branch and recommend which branch might be best for achieving specific objectives.

#TASK CRITERIA:

A refined version of the original search query, representing a thorough explanation of the Tree of Thoughts.
Ensure each query remains concise, with a focus on improving search relevance, precision, or scope.

#OUTPUT:
The AI will provide:

A refined version of the search query exploring the branches of thought.
        """)

    ARTICLE_SUMMARY_PROMPT_TEMPLATE = os.getenv("ARTICLE_SUMMARY_PROMPT_TEMPLATE",
        "Summarize the following content clearly and concisely, so that the most important points can be gleaned at a glance of this summary:\n{content}")

    CLUSTER_TITLE_PROMPT_TEMPLATE = os.getenv("CLUSTER_TITLE_PROMPT_TEMPLATE",
        "Generate a concise and descriptive title for the following content:\n{content}")

    CLUSTER_SUMMARY_PROMPT_TEMPLATE = os.getenv("CLUSTER_SUMMARY_PROMPT_TEMPLATE",
        "Summarize the following content in a few sentences clearly and concisely, so that the most important points can be gleaned at a glance of this summary:\n{content}")
