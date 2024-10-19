# models.py

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr

# Model for user registration (input)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str  # Plain text password input

# Model for user data stored in the database (internal use)
class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str
    is_active: bool
    created_at: datetime
    profile_info: Optional[str] = None  # Profile information
    topics_of_interest: List[str] = []  # Topics extracted from articles
    pagerank: float = 0.0  # PageRank score

# Model for user data returned in responses (output)
class UserRead(BaseModel):
    id: str = Field(..., alias='_id')
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    profile_info: Optional[str] = None
    topics_of_interest: List[str] = []
    pagerank: float = 0.0

    class Config:
        populate_by_name = True
        from_attributes = True

# Model for article input
class ArticleInput(BaseModel):
    title: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Model for article data stored in the database (internal use)
class ArticleInDB(BaseModel):
    id: str
    title: str
    content: str
    processed_content: str
    embedding: List[float]
    timestamp: datetime
    summary: str
    author_username: str  # Reference to the user
    pagerank: float = 0.0  # PageRank score

# Model for article output
class ArticleOutput(BaseModel):
    id: str = Field(..., alias='_id')
    title: str
    content: str
    processed_content: str
    timestamp: datetime
    summary: str
    pagerank: float = 0.0
    similar_articles: List[str] = Field(default_factory=list)
    author_username: str  # Author's username

    class Config:
        populate_by_name = True
        from_attributes = True

# Model for query input
class QueryInput(BaseModel):
    query: str
    limit: int

# Model for query result item
class QueryResultItem(BaseModel):
    type: str  # 'article' or 'author'
    id: Optional[str] = None  # Article ID
    username: Optional[str] = None  # Username for authors
    title: Optional[str] = None  # Article title
    content: Optional[str] = None  # Article content
    profile_info: Optional[str] = None  # Author's profile info
    topics_of_interest: Optional[List[str]] = None  # Author's topics
    pagerank: float = 0.0
    similarity: Optional[float] = None  # Similarity score for articles

# Model for query response
class QueryResponse(BaseModel):
    results: List[QueryResultItem]
