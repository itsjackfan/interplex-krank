# db.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pymongo
from pymongo import MongoClient, UpdateOne
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from utils import convert_objectid_to_str  # Import the helper function

# Initialize Logging
logger = logging.getLogger(__name__)

class MongoDBConnection:
    def __init__(self):
        try:
            self.client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            self.db = self.client[Config.MONGODB_DB_NAME]
            self.articles = self.db.articles
            self.users = self.db.users  # Authors collection
            self.graph = self.db.graph  # Collection for graph edges
            # Ensure indexes for performance
            self._ensure_indexes()
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _ensure_indexes(self):
        """
        Ensure that necessary indexes are created for optimal query performance.
        """
        try:
            # Unique index on username and email to prevent duplicates
            self.users.create_index("username", unique=True)
            self.users.create_index("email", unique=True)
            # Index on 'similar_articles' for PageRank computations
            self.articles.create_index("similar_articles")
            # Index on 'author_username' to quickly retrieve author's articles
            self.articles.create_index("author_username")
            # Index on 'type' for graph edges
            self.graph.create_index("type")
            logger.info("MongoDB indexes ensured.")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    def verify_connectivity(self) -> bool:
        """
        Verify if the MongoDB connection is alive.
        """
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB connectivity verification failed: {e}")
            return False

    # User (Author) Management Methods

    def create_user(self, user_data: Dict[str, Any]):
        """
        Insert a new user into the 'users' collection.
        """
        try:
            user_data['pagerank'] = 0.0  # Initialize pagerank
            self.users.insert_one(user_data)
            logger.debug(f"User {user_data['username']} created.")
            # Create a node in the graph
            self.graph.insert_one({
                'from': user_data['username'],
                'to': user_data['username'],
                'type': 'author_node'
            })
        except pymongo.errors.DuplicateKeyError as e:
            logger.error(f"Duplicate user detected: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by their username.
        """
        try:
            user = self.users.find_one({"username": username})
            if user:
                user = convert_objectid_to_str(user)  # Convert ObjectId to string
            return user
        except Exception as e:
            logger.error(f"Error retrieving user {username}: {e}")
            raise

    def get_users_by_usernames(self, usernames: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple users by their usernames.
        """
        try:
            users = list(self.users.find({"username": {"$in": usernames}}))
            return users
        except Exception as e:
            logger.error(f"Error retrieving users {usernames}: {e}")
            raise

    def update_user_profile(self, username: str, profile_info: str):
        """
        Update a user's profile information.
        """
        try:
            self.users.update_one(
                {"username": username},
                {"$set": {"profile_info": profile_info}}
            )
            logger.debug(f"Updated profile for user {username}.")
        except Exception as e:
            logger.error(f"Error updating profile for user {username}: {e}")
            raise

    # Article Management Methods

    def create_article(self, article_id: str, title: str, content: str, processed_content: str,
                       embedding: List[float], timestamp: datetime, summary: str,
                       author_username: str):
        """
        Insert a new article into the 'articles' collection.
        """
        try:
            article_document = {
                "_id": article_id,  # Custom string ID
                "title": title,
                "content": content,
                "processed_content": processed_content,
                "embedding": embedding,
                "timestamp": timestamp,  # Stored as datetime
                "summary": summary,
                "pagerank": 0.0,
                "similar_articles": [],
                "author_username": author_username
            }
            self.articles.insert_one(article_document)
            logger.debug(f"Article {article_id} inserted into MongoDB.")
            # Add edge between author and article in the graph
            self.graph.insert_one({
                'from': author_username,
                'to': article_id,
                'type': 'authored'
            })
        except pymongo.errors.DuplicateKeyError:
            logger.error(f"Duplicate article ID detected: {article_id}")
            raise
        except Exception as e:
            logger.error(f"Error inserting article {article_id}: {e}")
            raise

    def update_article(self, article_id: str, title: str, content: str, processed_content: str,
                       embedding: List[float], timestamp: datetime, summary: str):
        try:
            result = self.articles.update_one(
                {"_id": article_id},
                {"$set": {
                    "title": title,
                    "content": content,
                    "processed_content": processed_content,
                    "embedding": embedding,
                    "timestamp": timestamp,
                    "summary": summary
                }}
            )
            if result.matched_count == 0:
                raise Exception("Article not found.")
        except Exception as e:
            logger.error(f"Error updating article {article_id}: {e}")
            raise

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single article by its ID.
        """
        try:
            article = self.articles.find_one({"_id": article_id})
            return article
        except Exception as e:
            logger.error(f"Error retrieving article {article_id}: {e}")
            raise

    def get_articles(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple articles by their IDs.
        """
        try:
            articles_cursor = self.articles.find({"_id": {"$in": article_ids}})
            articles = list(articles_cursor)
            return articles
        except Exception as e:
            logger.error(f"Error retrieving articles {article_ids}: {e}")
            raise

    def get_all_articles_by_author(self, username: str) -> List[Dict[str, Any]]:
        """
        Retrieve all articles authored by a specific user.
        """
        try:
            articles_cursor = self.articles.find({"author_username": username})
            articles = list(articles_cursor)
            return articles
        except Exception as e:
            logger.error(f"Error retrieving articles for user {username}: {e}")
            raise

    def get_all_articles(self) -> List[Dict[str, Any]]:
        """
        Retrieve all articles in the database.
        """
        try:
            articles_cursor = self.articles.find()
            articles = list(articles_cursor)
            return articles
        except Exception as e:
            logger.error(f"Error retrieving all articles: {e}")
            raise

    def update_relationships(self, article_id: str, similarity_threshold: float):
        """
        Update the 'similar_articles' field for a given article based on embedding similarity.
        """
        try:
            # Fetch the embedding of the current article
            current_article = self.get_article(article_id)
            if not current_article:
                logger.error(f"Article {article_id} not found for updating relationships.")
                return

            current_embedding = np.array(current_article["embedding"]).reshape(1, -1)

            # Fetch embeddings of all other articles
            other_articles_cursor = self.articles.find(
                {"_id": {"$ne": article_id}, "embedding": {"$exists": True, "$ne": []}},
                {"_id": 1, "embedding": 1}
            )

            similar_article_ids = []
            for article in other_articles_cursor:
                other_article_id = article["_id"]
                other_embedding = np.array(article["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(current_embedding, other_embedding)[0][0]
                if similarity >= similarity_threshold:
                    similar_article_ids.append(other_article_id)
                    # Update the 'similar_articles' field of the similar article to include the current article
                    self.articles.update_one(
                        {"_id": other_article_id},
                        {"$addToSet": {"similar_articles": article_id}}
                    )

            # Update the 'similar_articles' field of the current article
            self.articles.update_one(
                {"_id": article_id},
                {"$set": {"similar_articles": similar_article_ids}}
            )
            logger.debug(f"Updated similar_articles for article {article_id} with {len(similar_article_ids)} similar articles.")

        except Exception as e:
            logger.error(f"Error updating relationships for article {article_id}: {e}")
            raise

    def get_similar_articles(
        self,
        query_embedding: List[float],
        similarity_threshold: float,
        limit: int,
        use_pagerank_weighting: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve articles similar to the query_embedding based on cosine similarity.
        Optionally weight the similarity by PageRank scores.
        """
        try:
            # Convert query_embedding to numpy array
            query_emb = np.array(query_embedding).reshape(1, -1)

            # Fetch articles with embeddings
            articles_cursor = self.articles.find({"embedding": {"$exists": True, "$ne": []}})
            similar_articles = []
            for article in articles_cursor:
                article_id = article["_id"]
                embedding = np.array(article["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(query_emb, embedding)[0][0]
                if similarity >= similarity_threshold:
                    pagerank = article.get("pagerank", 0.0)
                    if use_pagerank_weighting:
                        weighted_similarity = similarity * pagerank
                    else:
                        weighted_similarity = similarity
                    # Include all article fields in the result
                    article_data = article.copy()
                    article_data["id"] = article_data.pop("_id")
                    article_data["similarity"] = similarity
                    article_data["weighted_similarity"] = weighted_similarity
                    similar_articles.append(article_data)

            # Sort articles based on similarity or weighted_similarity
            if use_pagerank_weighting:
                similar_articles.sort(key=lambda x: x["weighted_similarity"], reverse=True)
            else:
                similar_articles.sort(key=lambda x: x["similarity"], reverse=True)

            # Limit the number of results
            limited_articles = similar_articles[:limit]
            return limited_articles

        except Exception as e:
            logger.error(f"Error retrieving similar articles: {e}")
            raise

    def get_article_by_author(self, article_id: str, author_username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single article by its ID and author.
        """
        try:
            article = self.articles.find_one({"_id": article_id, "author_username": author_username})
            return article
        except Exception as e:
            logger.error(f"Error retrieving article {article_id}: {e}")
            raise

    def get_all_user_nodes(self) -> List[Dict[str, Any]]:
        """
        Retrieve all users to be included as nodes in the graph.
        """
        try:
            users = list(self.users.find())
            return users
        except Exception as e:
            logger.error(f"Error retrieving all user nodes: {e}")
            raise

    def get_graph_edges(self) -> List[Dict[str, Any]]:
        """
        Retrieve all edges from the graph collection.
        """
        try:
            edges = list(self.graph.find())
            return edges
        except Exception as e:
            logger.error(f"Error retrieving graph edges: {e}")
            raise

    def extract_topics_from_articles(self, articles: List[Dict[str, Any]]) -> List[str]:
        """
        Extract topics from articles using a placeholder function.
        """
        # Placeholder implementation
        topics = []
        for article in articles:
            # For demonstration, we'll just split the content into words and pick unique words
            content = article.get('content', '')
            words = content.split()
            topics.extend(words)
        # Return unique topics
        return list(set(topics))

    def update_user_topics(self, username: str):
        """
        Update the user's topics of interest based on their articles.
        """
        try:
            articles = self.get_all_articles_by_author(username)
            topics = self.extract_topics_from_articles(articles)
            self.users.update_one(
                {'username': username},
                {'$set': {'topics_of_interest': topics}}
            )
            logger.debug(f"Updated topics of interest for user {username}.")
        except Exception as e:
            logger.error(f"Error updating topics for user {username}: {e}")
            raise

    def close(self):
        self.client.close()
