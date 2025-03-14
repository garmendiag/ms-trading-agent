"""
News Sentiment Analysis Module

This module is responsible for collecting and analyzing news sentiment
using the NewsAPI and natural language processing techniques. It provides
functions to fetch news articles related to financial markets, analyze
sentiment, and generate sentiment scores for use by the trading agent.
"""

import os
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json

class NewsSentimentAnalyzer:
    """
    Class for collecting and analyzing news sentiment from financial news sources.
    """
    
    # List of financial news sources
    FINANCIAL_SOURCES = [
        'bloomberg', 'cnbc', 'financial-times', 'the-wall-street-journal',
        'business-insider', 'fortune', 'reuters', 'the-economist'
    ]
    
    # List of relevant keywords for market sentiment
    MARKET_KEYWORDS = [
        'stock market', 'S&P 500', 'Nasdaq', 'Federal Reserve', 'Fed',
        'interest rates', 'inflation', 'recession', 'economic growth',
        'market volatility', 'bull market', 'bear market', 'market crash',
        'market rally', 'earnings', 'GDP', 'unemployment', 'treasury yields'
    ]
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the news sentiment analyzer.
        
        Args:
            api_key (str, optional): NewsAPI key. If None, will look for
                                    NEWS_API_KEY environment variable.
            cache_dir (str, optional): Directory to cache news data.
                                      Defaults to None (no caching).
        """
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.environ.get('NEWS_API_KEY')
        
        if not self.api_key:
            print("Warning: No NewsAPI key provided. Using demo mode with limited functionality.")
            # In demo mode, we'll use pre-cached data or mock data
        
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize NLTK components
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            print("Downloading NLTK vader_lexicon...")
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_financial_news(self, query=None, sources=None, days=1):
        """
        Get financial news articles from NewsAPI.
        
        Args:
            query (str, optional): Search query. If None, uses a default market query.
            sources (list, optional): List of news sources. If None, uses FINANCIAL_SOURCES.
            days (int, optional): Number of days to look back. Defaults to 1.
        
        Returns:
            list: List of news articles.
        """
        if query is None:
            # Use a default query that captures market-related news
            query = 'stock market OR S&P 500 OR Nasdaq OR Federal Reserve'
        
        if sources is None:
            sources = self.FINANCIAL_SOURCES
        
        # Check if cached data exists and is recent enough
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f"news_{query.replace(' ', '_')}_{days}d.json"
            if cache_file.exists():
                # For news, data that's less than 3 hours old is considered fresh
                file_time = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.datetime.now() - file_time).seconds < 10800:  # 3 hours in seconds
                    print(f"Loading cached news data from {cache_file}")
                    with open(cache_file, 'r') as f:
                        return json.load(f)
        
        # If no API key or in demo mode, use mock data
        if not self.api_key:
            print(f"No API key available. Using mock news data.")
            return self._get_mock_news(query, days)
        
        try:
            # Calculate date range
            to_date = datetime.datetime.now()
            from_date = to_date - datetime.timedelta(days=days)
            
            # Format dates for API
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            
            # Prepare API request
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'sources': ','.join(sources),
                'from': from_date_str,
                'to': to_date_str,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.api_key
            }
            
            # Make API request
            print(f"Fetching news for query: {query}")
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code != 200:
                print(f"Error fetching news: {data.get('message', 'Unknown error')}")
                return self._get_mock_news(query, days)
            
            articles = data.get('articles', [])
            
            # Cache the data if cache_dir is specified
            if self.cache_dir:
                with open(cache_file, 'w') as f:
                    json.dump(articles, f)
            
            return articles
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_mock_news(query, days)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a text using VADER sentiment analyzer.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            dict: Dictionary containing sentiment scores.
        """
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        return self.sentiment_analyzer.polarity_scores(text)
    
    def get_market_sentiment(self, days=1):
        """
        Get overall market sentiment based on recent financial news.
        
        Args:
            days (int, optional): Number of days to look back. Defaults to 1.
        
        Returns:
            dict: Dictionary containing market sentiment information.
        """
        # Get news for each market keyword
        all_articles = []
        for keyword in self.MARKET_KEYWORDS:
            articles = self.get_financial_news(query=keyword, days=days)
            all_articles.extend(articles)
        
        # Remove duplicates based on title
        unique_articles = []
        seen_titles = set()
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        if not unique_articles:
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'Neutral',
                'article_count': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'top_positive': None,
                'top_negative': None,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Analyze sentiment for each article
        sentiments = []
        for article in unique_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description for better sentiment analysis
            content = f"{title}. {description}"
            
            sentiment = self.analyze_sentiment(content)
            article['sentiment'] = sentiment
            sentiments.append(sentiment['compound'])
        
        # Calculate overall sentiment
        overall_sentiment = np.mean(sentiments)
        
        # Determine sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = 'Very Positive'
        elif overall_sentiment > 0.05:
            sentiment_label = 'Positive'
        elif overall_sentiment < -0.2:
            sentiment_label = 'Very Negative'
        elif overall_sentiment < -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        # Calculate sentiment ratios
        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        positive_ratio = positive_count / len(sentiments) if sentiments else 0
        negative_ratio = negative_count / len(sentiments) if sentiments else 0
        neutral_ratio = neutral_count / len(sentiments) if sentiments else 0
        
        # Find top positive and negative articles
        sorted_articles = sorted(unique_articles, key=lambda x: x['sentiment']['compound'], reverse=True)
        top_positive = sorted_articles[0] if sorted_articles and sorted_articles[0]['sentiment']['compound'] > 0 else None
        top_negative = sorted_articles[-1] if sorted_articles and sorted_articles[-1]['sentiment']['compound'] < 0 else None
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'article_count': len(unique_articles),
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_keyword_sentiments(self, days=1):
        """
        Get sentiment for each market keyword.
        
        Args:
            days (int, optional): Number of days to look back. Defaults to 1.
        
        Returns:
            dict: Dictionary containing sentiment for each keyword.
        """
        result = {}
        
        for keyword in self.MARKET_KEYWORDS:
            articles = self.get_financial_news(query=keyword, days=days)
            
            if not articles:
                result[keyword] = {
                    'sentiment': 0,
                    'article_count': 0,
                    'sentiment_label': 'No Data'
                }
                continue
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}"
                sentiment = self.analyze_sentiment(content)
                sentiments.append(sentiment['compound'])
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments)
            
            # Determine sentiment label
            if avg_sentiment > 0.2:
                sentiment_label = 'Very Positive'
            elif avg_sentiment > 0.05:
                sentiment_label = 'Positive'
            elif avg_sentiment < -0.2:
                sentiment_label = 'Very Negative'
            elif avg_sentiment < -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            result[keyword] = {
                'sentiment': avg_sentiment,
                'article_count': len(articles),
                'sentiment_label': sentiment_label
            }
        
        return result
    
    def _get_mock_news(self, query, days):
        """
        Generate mock news data for demonstration purposes when API key is not available.
        
        Args:
            query (str): Search query.
            days (int): Number of days to look back.
        
        Returns:
            list: List of mock news articles.
        """
        # Create date range for the past days
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Generate random dates within the range
        dates = [start_date + datetime.timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds())))
            for _ in range(10)]
        
        # Sort dates in descending order (newest first)
        dates.sort(reverse=True)
        
        # Sample headlines based on query
        headlines = {
            'stock market': [
                "Stock Market Rallies on Strong Economic Data",
                "Market Dips as Investors Weigh Inflation Concerns",
                "Stocks Close Higher After Fed Comments",
                "Market Volatility Increases Amid Global Tensions",
                "Investors Cautious as Earnings Season Approaches"
            ],
            'S&P 500': [
                "S&P 500 Reaches New All-Time High",
                "S&P 500 Drops on Tech Sector Weakness",
                "S&P 500 Gains for Fifth Straight Session",
                "Analysts Predict Continued Growth for S&P 500",
                "S&P 500 Volatility Index Spikes on Economic Uncertainty"
            ],
            'Nasdaq': [
                "Nasdaq Surges on Tech Earnings",
                "Nasdaq Falls as Investors Rotate to Value Stocks",
                "Nasdaq Outperforms Other Indices This Quarter",
                "Tech-Heavy Nasdaq Faces Pressure from Rising Rates",
                "Nasdaq Composite Breaks Through Resistance Level"
            ],
            'Federal Reserve': [
                "Fed Signals Potential Rate Cut in Coming Months",
                "Federal Reserve Holds Rates Steady, Cites Economic Strength",
                "Markets React to Fed Chair's Comments on Inflation",
                "Fed Minutes Reveal Divided Opinion on Monetary Policy",
                "Federal Reserve Faces Pressure as Inflation Persists"
            ]
        }
        
        # Default to stock market headlines if query not in our samples
        query_key = next((k for k in headlines.keys() if k in query), 'stock market')
        
        # Generate mock articles
        articles = []
        for i in range(min(10, len(dates))):
            headline = headlines[query_key][i % len(headlines[query_key])]
            
            # Generate a description that's somewhat related to the headline
            description = f"Analysis of recent market movements shows that {headline.lower()[:-1]} due to various economic factors and investor sentiment."
            
            articles.append({
                'source': {'id': 'mock-source', 'name': 'Mock Financial News'},
                'author': f"Mock Author {i+1}",
                'title': headline,
                'description': description,
                'url': f"https://mock-financial-news.com/article{i+1}",
                'urlToImage': f"https://mock-financial-news.com/images/article{i+1}.jpg",
                'publishedAt': dates[i].isoformat(),
                'content': f"{description} This is mock content for demonstration purposes."
            })
        
        return articles

# Example usage
if __name__ == "__main__":
    # Create a news sentiment analyzer with caching
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    analyzer = NewsSentimentAnalyzer(cache_dir=cache_dir)
    
    # Get financial news
    news = analyzer.get_financial_news(query="S&P 500", days=1)
    print(f"Retrieved {len(news)} news articles")
    
    if news:
        print("\nSample article:")
        article = news[0]
        print(f"Title: {article.get('title')}")
        print(f"Source: {article.get('source', {}).get('name')}")
        print(f"Published: {article.get('publishedAt')}")
        
    <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>