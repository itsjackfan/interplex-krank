# main.py

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Path
from pydantic import BaseModel, Field

from config import Config
from models import (
    ArticleInput, ArticleOutput, QueryInput, UserCreate, UserRead, UserInDB,
    QueryResponse, QueryResultItem
)
from db import MongoDBConnection
from llm_client import get_llm_client
from utils import preprocess_for_embedding, hash_password, verify_password, convert_objectid_to_str
from sentence_transformers import SentenceTransformer

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import numpy as np
import pymongo
import networkx as nx

# For user authentication
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

# Initialize Logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title=Config.APP_TITLE)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB Connection
mongodb_conn = MongoDBConnection()

# Initialize SentenceTransformer Model
model_cache = {}

def get_sentence_transformer():
    if Config.SENTENCE_TRANSFORMER_MODEL not in model_cache:
        model_cache[Config.SENTENCE_TRANSFORMER_MODEL] = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        logger.info(f"SentenceTransformer model '{Config.SENTENCE_TRANSFORMER_MODEL}' loaded.")
    return model_cache[Config.SENTENCE_TRANSFORMER_MODEL]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def generate_article_id() -> str:
    return os.urandom(8).hex()

def create_article_in_mongodb(article_id: str, title: str, content: str, processed_content: str,
                              embedding: List[float], timestamp: datetime, summary: str = "", author_username: str = ""):
    try:
        mongodb_conn.create_article(
            article_id=article_id,
            title=title,
            content=content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=timestamp,
            summary=summary,
            author_username=author_username
        )
        logger.debug(f"Article {article_id} created in MongoDB.")
    except Exception as e:
        logger.error(f"Error creating article {article_id}: {e}")
        raise

# Authentication setup
SECRET_KEY = Config.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=180))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = mongodb_conn.get_user_by_username(username)
        if user is None:
            raise credentials_exception
        return UserInDB(**user)
    except JWTError:
        raise credentials_exception

# -----------------------------------------------------------------------------
# Graph and PageRank Functions
# -----------------------------------------------------------------------------

def compute_pagerank():
    """
    Computes PageRank using NetworkX and updates the scores in MongoDB.
    Includes authors and articles as nodes.
    """
    logger.info("Starting PageRank computation including authors and articles...")
    start_time = time.time()
    try:
        # Fetch all edges from the graph collection
        edges = mongodb_conn.get_graph_edges()

        # Build the graph
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(edge['from'], edge['to'])

        if G.number_of_edges() == 0:
            logger.warning("No edges found for PageRank computation.")
            return

        # Compute PageRank
        pagerank_scores = nx.pagerank(G, alpha=Config.PAGERANK_ALPHA, max_iter=100, tol=1e-06)

        # Separate author and article nodes
        user_nodes = mongodb_conn.get_all_user_nodes()
        article_nodes = mongodb_conn.get_all_articles()

        # Update PageRank scores for authors
        bulk_user_updates = [
            pymongo.UpdateOne(
                {"username": user['username']},
                {"$set": {"pagerank": pagerank_scores.get(user['username'], 0.0)}}
            ) for user in user_nodes
        ]
        if bulk_user_updates:
            mongodb_conn.users.bulk_write(bulk_user_updates)
            logger.info("PageRank scores updated for authors in MongoDB.")

        # Update PageRank scores for articles
        bulk_article_updates = [
            pymongo.UpdateOne(
                {"_id": article['_id']},
                {"$set": {"pagerank": pagerank_scores.get(article['_id'], 0.0)}}
            ) for article in article_nodes
        ]
        if bulk_article_updates:
            mongodb_conn.articles.bulk_write(bulk_article_updates)
            logger.info("PageRank scores updated for articles in MongoDB.")

        elapsed_time = time.time() - start_time
        logger.info(f"PageRank computation and update completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during PageRank computation: {e}")

# -----------------------------------------------------------------------------
# Scheduler and Startup Events
# -----------------------------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the compute_pagerank function to run every 10 minutes
scheduler.add_job(
    func=compute_pagerank,
    trigger=IntervalTrigger(minutes=10),
    id='compute_pagerank_job',
    name='Compute PageRank every 10 minutes',
    replace_existing=True
)

# Shut down the scheduler when exiting the app
import atexit
atexit.register(lambda: scheduler.shutdown())

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete. Scheduled background tasks.")

@app.on_event("shutdown")
async def shutdown_event():
    mongodb_conn.client.close()
    logger.info("Application shutdown complete.")

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/")
async def default_post():
    return {"message": "Welcome to the ArticleRank API"}

# User registration endpoint
@app.post("/register", response_model=UserRead)
async def register_user(user_input: UserCreate):
    if mongodb_conn.get_user_by_username(user_input.username):
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = hash_password(user_input.password)
    user_in_db = UserInDB(
        username=user_input.username,
        email=user_input.email,
        hashed_password=hashed_password,
        is_active=True,
        created_at=datetime.utcnow()
    )

    # Convert to dict, excluding unset and None values
    user_data = user_in_db.dict(by_alias=True, exclude_unset=True, exclude_none=True)

    mongodb_conn.create_user(user_data)

    user = mongodb_conn.get_user_by_username(user_input.username)
    if not user:
        raise HTTPException(status_code=500, detail="User creation failed.")
    return UserRead(**user)

# User login endpoint
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = mongodb_conn.get_user_by_username(form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/articles/{article_id}", response_model=ArticleOutput)
async def get_article_endpoint(
    article_id: str,
    current_user: UserInDB = Depends(get_current_user)
):

    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    article = mongodb_conn.get_article_by_author(article_id, current_user.username)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return ArticleOutput(**article)

@app.get("/articles/", response_model=List[ArticleOutput])
async def get_all_articles_endpoint(
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    articles = mongodb_conn.get_all_articles_by_author(current_user.username)
    if not articles:
        raise HTTPException(status_code=404, detail="No articles found")

    return [ArticleOutput(**article) for article in articles]

@app.put("/articles/{article_id}", response_model=ArticleOutput)
async def update_article_endpoint(
    article_input: ArticleInput,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user),
    article_id: str = Path(..., description="The ID of the article to update")
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    # Verify that the article exists and is authored by the current user
    existing_article = mongodb_conn.get_article_by_author(article_id, current_user.username)
    if not existing_article:
        raise HTTPException(status_code=404, detail="Article not found or access denied")

    try:
        # Process the updated content
        processed_content = preprocess_for_embedding(article_input.content)
        model = get_sentence_transformer()
        embedding = model.encode([processed_content], show_progress_bar=False)[0].tolist()

        # Generate new summary using LLM
        llm_client = get_llm_client()
        summary_prompt = Config.ARTICLE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
        summary = await llm_client.generate_content(summary_prompt)

        # Update the article in MongoDB
        mongodb_conn.update_article(
            article_id=article_id,
            title=article_input.title,
            content=article_input.content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=article_input.timestamp,
            summary=summary
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(
            mongodb_conn.update_relationships,
            article_id,
            Config.SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS
        )
        background_tasks.add_task(compute_pagerank)
        background_tasks.add_task(mongodb_conn.update_user_topics, current_user.username)

        # Retrieve the updated article to include all fields
        article = mongodb_conn.get_article(article_id)
        if not article:
            raise HTTPException(status_code=500, detail="Failed to retrieve the updated article.")
        article_output = ArticleOutput(**article)
        return article_output

    except Exception as e:
        logger.error(f"Error updating article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/articles", response_model=ArticleOutput)
async def create_article_endpoint(
    article_input: ArticleInput,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        article_id = generate_article_id()
        processed_content = preprocess_for_embedding(article_input.content)
        model = get_sentence_transformer()
        embedding = model.encode([processed_content], show_progress_bar=False)[0].tolist()

        # Generate summary using LLM
        llm_client = get_llm_client()
        summary_prompt = Config.ARTICLE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
        summary = await llm_client.generate_content(summary_prompt)

        # Create article in MongoDB
        create_article_in_mongodb(
            article_id=article_id,
            title=article_input.title,
            content=article_input.content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=article_input.timestamp,
            summary=summary,
            author_username=current_user.username
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(
            mongodb_conn.update_relationships,
            article_id,
            Config.SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS
        )
        background_tasks.add_task(compute_pagerank)
        background_tasks.add_task(mongodb_conn.update_user_topics, current_user.username)

        # Retrieve the created article to include all fields
        article = mongodb_conn.get_article(article_id)
        if not article:
            raise HTTPException(status_code=500, detail="Failed to retrieve the created article.")
        article_output = ArticleOutput(**article)
        return article_output

    except Exception as e:
        logger.error(f"Error creating article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_articles(
    query_input: QueryInput,
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        processed_query = preprocess_for_embedding(query_input.query)
        model = get_sentence_transformer()
        query_embedding = model.encode([processed_query], show_progress_bar=False)[0].tolist()

        # Find similar articles
        similar_articles = mongodb_conn.get_similar_articles(
            query_embedding=query_embedding,
            similarity_threshold=Config.SIMILARITY_THRESHOLD_ARTICLE,
            limit=query_input.limit
        )

        # From the articles, find associated authors
        author_usernames = {article['author_username'] for article in similar_articles}
        authors = mongodb_conn.get_users_by_usernames(list(author_usernames))

        # Prepare results
        results = []

        for article in similar_articles:
            results.append(QueryResultItem(
                type='article',
                id=article['id'],
                title=article['title'],
                content=article['content'],
                pagerank=article.get('pagerank', 0.0),
                similarity=article.get('similarity', 0.0),
                username=article['author_username']
            ))

        for author in authors:
            results.append(QueryResultItem(
                type='author',
                username=author['username'],
                profile_info=author.get('profile_info', ''),
                topics_of_interest=author.get('topics_of_interest', []),
                pagerank=author.get('pagerank', 0.0)
            ))

        # Sort results based on combined score of PageRank and similarity
        def ranking_score(item):
            if item.type == 'article':
                return item.pagerank + item.similarity
            else:
                return item.pagerank

        sorted_results = sorted(results, key=ranking_score, reverse=True)
        return QueryResponse(results=sorted_results)

    except Exception as e:
        logger.error(f"Error querying articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/test/mongodb")
async def test_mongodb():
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="MongoDB connection failed.")
    return {"mongodb": "Connection successful."}

@app.get("/test/llm")
async def test_llm():
    
    try:
        llm_client = get_llm_client()
        response = await llm_client.generate_content("Write a test summary.")
        return {"llm": "Connection successful.", "response": response}
    except Exception as e:
        logger.error(f"LLM API test failed: {e}")
        raise HTTPException(status_code=500, detail="LLM API test failed.")

@app.get("/test/model")
async def test_model():
    try:
        model = get_sentence_transformer()
        sample_text = "Test encoding"
        embedding = model.encode([sample_text], show_progress_bar=False)[0]
        return {
            "model": "Loaded successfully.",
            "embedding_length": len(embedding),
            "sample_embedding": embedding[:5].tolist()  # Return first 5 elements as a sample
        }
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise HTTPException(status_code=500, detail="SentenceTransformer model test failed.")

@app.head("/")
async def root():
    return {"message": "Hello from FastAPI!"}

# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------

import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=10000)
