import tweepy
from nba_api.stats.endpoints import boxscoresummaryv2, scoreboardv2
from nba_api.stats.static import players
import requests
import json
import redis
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from openai import OpenAI  # Import OpenAI SDK for Grok API calls
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Redis for caching and pick tracking
redis_client = redis.Redis(host='localhost', port=6379, db=0)

load_dotenv()


X_API_KEY = os.environ.get("X_API_KEY")
X_API_BEARER_TOKEN = os.environ.get("X_API_BEARER_TOKEN")
X_API_SECRET = os.environ.get("X_API_SECRET")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

# X_API_KEY="t9DYzS4ipO9Qi4Y9gQV8EhPeu"
# X_API_SECRET="OAa1hnDmrEaCOKFGqiCpDbNdyQvrknEe5nqW44BqNpAXYdWUjb"
# X_ACCESS_TOKEN="1900025310823014400-sUh83PEIDN2soeT1GJJFnJMjFeKoop"
# X_ACCESS_TOKEN_SECRET="mbYwR2hzzj0Id7CIO3DbcDzf21DCBnTrU01YbnXX7trOm"
# X_API_BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAACqezwEAAAAAuPU28toxvRy51Jze03N1ir8ADD4%3DuPjsmruZLzCeFgSJS6KCcrdn12NioHQ2e6H3I9HrLdN4MAkISY"
# GROK_API_KEY="xai-D9dTaDyRbdPkDhgkZRzl52rlgwdxpDX5JMGtFdMPvcI95GMKFIXyBA9wsK51IIrASHVKf0syAdRNSYqf"

# Tweepy authentication for staging with v2 compatibility (using Bearer Token)
client = tweepy.Client(
    bearer_token=X_API_BEARER_TOKEN,  # Use Bearer Token for OAuth 2.0
    consumer_key=X_API_KEY,
    consumer_secret=X_API_SECRET,
    access_token=X_ACCESS_TOKEN,
    access_token_secret=X_ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

# Initialize OpenAI client for Grok API
grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

def image_to_base64(image_url):
    """Convert an image URL to a base64-encoded string."""
    try:
        logger.info(f"Staging - Fetching image from URL: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    except (requests.RequestException, IOError) as e:
        logger.error(f"Staging - Error fetching or processing image: {e}")
        return None

def extract_pick(tweet_text=None, image_base64=None, use_vision=False):
    """Extract picks from text and/or image using Grok's API."""
    safe_text = str(tweet_text).replace("{", "{{").replace("}", "}}") if tweet_text else ""
    logger.info(f"Staging - Input tweet_text: {safe_text}")
    if image_base64 and use_vision:
        logger.info("Staging - Image provided for analysis")

    messages = [
        {
            "role": "system",
            "content": """
You are an AI designed to analyze sports betting content from text and/or images. Your task is to extract any sports betting *pick* (a prediction for a future bet) from the provided text and/or image and return it as a JSON string or JSON array based on the following rules. Do NOT extract results (completed bets, whether wins or losses). Use these formats:

- For a single prop bet: {"type": "prop", "player": "player_name", "bet_type": "over" or "under", "value": "numeric_value", "stat": "PTS|AST|REB|PA|RA|PRA|G|SA|A|PTS|HR|RBI|SB|K" (PTS/AST/REB/PA/RA/PRA for NBA/NCAAB, G/SA/A/PTS for NHL, HR/RBI/SB/K for MLB, total_games for tennis)}
- For a single team bet: {"type": "team", "team1": "team_name", "team2": "opponent_name", "bet_type": "spread|moneyline|puck_line|run_line", "value": "numeric_value"} or {"type": "team", "player1": "player_name", "player2": "opponent_name", "bet_type": "moneyline", "value": "numeric_value"} for tennis
- For a single tennis over/under: {"type": "over_under", "team1": "player_name", "team2": "opponent_name", "bet_type": "over" or "under", "value": "numeric_value", "stat": "total_games"}
- For a parlay: {"type": "parlay", "picks": [pick1, pick2, ...], "odds": "optional_odds_value"} where each pick is a prop, team bet, or tennis over/under object, and "odds" is a string like "+1000" if present, detected only when the text includes 'parlay', '+', or similar combinative indicators.

Rules:
1. A *pick* is a prediction for a bet yet to be resolved (e.g., "Michigan State Money Line" or "Middle Tennessee +5.5").
2. A *result* is a completed bet outcome (win or loss). Filter out results by checking for:
   - Success indicators like "✅", "CASH", "💰", "won", "hit", or "tailed".
   - Failure indicators like "💔", "❌", "lost", or "missed".
   - Past-tense language like "was", "already", or records like "1-0" or "0-1".
   - Phrases implying completion (e.g., "First Half CASH").
3. For team bets:
   - Infer 'team2' from 'vs' clauses (e.g., 'MSU vs ORE' implies ORE as opponent).
   - Recognize team abbreviations:
     - NBA: 'LAL' (Lakers), 'BOS' (Celtics), 'GS' (Warriors), etc.
     - NCAAB: 'MSU' (Michigan State), 'UK' (Kentucky), 'DUKE' (Duke), etc.
     - NHL: 'TOR' (Maple Leafs), 'VGK' (Golden Knights), 'NYR' (Rangers), etc.
     - MLB: 'NYY' (Yankees), 'BOS' (Red Sox), 'LAD' (Dodgers), etc.
     - Tennis: Player names (e.g., 'Nadal', 'Djokovic') with 'vs' for opponents.
   - 'bet_type' is:
     - 'moneyline' for outright win bets (e.g., "+150" or "-120").
     - 'spread' for point spreads (e.g., "+5.5" or "-1.5") for NBA/NCAAB/MLB.
     - 'puck_line' for NHL spreads (typically +1.5 or -1.5).
     - 'run_line' for MLB spreads (typically +1.5 or -1.5).
   - If the line includes '(ALT)', treat it as the respective bet type with an alternative line (e.g., 'spread', 'puck_line', 'run_line').
   - 'value' is a string like "0" for moneyline or a number like "5.5" for spreads/puck_lines/run_lines.
4. For prop bets:
   - 'stat' field specifies the statistic:
     - NBA/NCAAB: PTS (points), AST (assists), REB (rebounds), PA (points+assists), RA (rebounds+assists), PRA (points+rebounds+assists).
     - NHL: G (goals), SA (shots against), A (assists), PTS (points, goals + assists).
     - MLB: HR (home runs), RBI (runs batted in), SB (stolen bases), K (strikeouts).
   - Infer the sport context from team names, player names, or league mentions (e.g., "NHL", "MLB", "ATP").
5. For tennis bets:
   - Use 'player1' and 'player2' instead of 'team1' and 'team2' for moneyline bets.
   - Use 'over_under' type for bets on total games, with 'stat': "total_games" and a numeric 'value' (e.g., "38.5").
   - Infer tennis context from player names or terms like "ATP", "WTA", or "total games".
6. For multiple picks:
   - If the text contains multiple independent picks without parlay indicators (e.g., no '+', 'parlay', or combined odds), return them as a JSON array: [pick1, pick2, ...].
   - If the text indicates a parlay (e.g., "Lakers -5.5 + Nuggets +3.5 +1000" or "NBA Parlay"), return a single object: {"type": "parlay", "picks": [pick1, pick2, ...], "odds": "optional_odds_value"}.
7. If the text or image contains a result instead of a pick, or no clear pick is found, return "null".
8. When analyzing an image, extract the betting information directly from the image content (e.g., team names, spreads, odds).

Examples:
- Single Pick (NBA): "LAL -5.5 vs BOS"
  Output: {"type": "team", "team1": "Lakers", "team2": "Celtics", "bet_type": "spread", "value": "-5.5"}
- Multiple Single Picks: "LAL -5.5, DEN +3.5"
  Output: [{"type": "team", "team1": "Lakers", "team2": "Celtics", "bet_type": "spread", "value": "-5.5"}, {"type": "team", "team1": "Nuggets", "team2": "implied_opponent", "bet_type": "spread", "value": "3.5"}]
- Parlay: "LAL -5.5 + DEN +3.5 +1000"
  Output: {"type": "parlay", "picks": [{"type": "team", "team1": "Lakers", "team2": "Celtics", "bet_type": "spread", "value": "-5.5"}, {"type": "team", "team1": "Nuggets", "team2": "implied_opponent", "bet_type": "spread", "value": "3.5"}], "odds": "+1000"}
- NBA Prop: "LeBron James Over 25.5 PTS"
  Output: {"type": "prop", "player": "LeBron James", "bet_type": "over", "value": "25.5", "stat": "PTS"}

Return your answer as a JSON string or JSON array only, wrapped in ```json ... ``` markers to clearly indicate the JSON content, with no additional text.
"""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze this text: '{safe_text}'"},
                *(
                    [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]
                    if image_base64 and use_vision else []
                )
            ]
        }
    ]

    try:
        # Use grok-2-vision-1212 for vision requests, grok-2 for text-only
        model = "grok-2-vision-1212" if use_vision else "grok-2"
        logger.info(f"Staging - Using model: {model}")
        response = grok_client.chat.completions.create(
            model=model,
            messages=messages
        )
        result = response.choices[0].message.content
        logger.info(f"Staging - Raw response from Grok: {result}")
    except Exception as e:
        logger.error(f"Staging - Error in grok_client.chat.completions.create: {e}")
        # Fallback: If vision model fails, try text-only with grok-2
        if "does not support image input" in str(e).lower() or "invalid model" in str(e).lower():
            logger.warning("Staging - Vision model not supported or unavailable, falling back to text-only analysis")
            messages[1]["content"] = [{"type": "text", "text": f"Analyze this text: '{safe_text}'"}]
            try:
                response = grok_client.chat.completions.create(
                    model="grok-2",
                    messages=messages
                )
                result = response.choices[0].message.content
                logger.info(f"Staging - Raw response from Grok (text-only fallback): {result}")
            except Exception as e:
                logger.error(f"Staging - Error in fallback grok_client.chat.completions.create: {e}")
                return None
        else:
            return None

    if result is None:
        logger.warning("Staging - Result is None, returning None")
        return None
    
    # Extract JSON content within ```json ... ``` markers, removing any preceding text
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result.strip())
    if json_match:
        cleaned_result = json_match.group(1).strip()
    else:
        cleaned_result = re.sub(r'^```json\s*|\s*```$', '', result.strip())  # Fallback to old method
    logger.info(f"Staging - Cleaned result for JSON parsing: {cleaned_result}")
    
    try:
        # Parse the JSON result
        parsed = json.loads(cleaned_result)
        if parsed == "null":  # Check if Grok returned "null" as a string
            logger.info("Staging - Grok returned 'null', returning None")
            return None
        # Return the parsed JSON as a string if it's a valid pick or array
        return json.dumps(parsed)
    except json.JSONDecodeError as e:
        logger.error(f"Staging - Invalid JSON from Grok: {cleaned_result} (Error: {e})")
        return None
    
def detect_pick_grok(tweet_text=None, image_url=None):
    """Detect picks from text and/or image using Grok's API, prioritizing text if a pick is found, handling single picks, parlays, and multiple single picks."""
    if not tweet_text and not image_url:
        logger.warning("Staging - No tweet text or image URL provided, returning None")
        return None

    all_picks = []

    # Step 1: Try to detect a pick from the text only
    if tweet_text:
        result = extract_pick(tweet_text=tweet_text, use_vision=False)
        logger.info(f"Staging - Result from text-only analysis: {result}")
        if result:
            try:
                # Remove Markdown code block syntax (e.g., ```json ... ```) and extra whitespace
                cleaned_result = re.sub(r'^```json\s*|\s*```$', '', result.strip())
                logger.info(f"Staging - Cleaned result for JSON parsing (text): {cleaned_result}")
                parsed = json.loads(cleaned_result)
                # Handle case where parsed is None (e.g., json.loads("null"))
                if parsed is None:
                    logger.info("Staging - Parsed result from text is None (likely 'null' JSON), proceeding to image if available")
                elif isinstance(parsed, dict):
                    if parsed.get("type") == "parlay":
                        all_picks.append(parsed)  # Single parlay object
                    else:
                        all_picks.append(parsed)  # Single pick
                elif isinstance(parsed, list):
                    all_picks.extend(parsed)  # Multiple single picks
                else:
                    logger.warning(f"Staging - Unexpected JSON format from text: {parsed}")
            except json.JSONDecodeError as e:
                logger.error(f"Staging - Invalid JSON from text: {result} (Error: {e})")

    # Step 2: If a pick was found in the text, skip image analysis
    if all_picks:
        logger.info("Staging - Pick detected in text, skipping image analysis")
    elif image_url:
        # Convert image URL to base64
        image_base64 = image_to_base64(image_url)
        if not image_base64:
            logger.warning("Staging - Failed to convert image to base64, no pick detected")
            return None

        # Step 3: Analyze the image if no pick was found in the text
        result = extract_pick(tweet_text=tweet_text, image_base64=image_base64, use_vision=True)
        logger.info(f"Staging - Result from image analysis: {result}")
        if result:
            try:
                # Remove Markdown code block syntax (e.g., ```json ... ```) and extra whitespace
                cleaned_result = re.sub(r'^```json\s*|\s*```$', '', result.strip())
                logger.info(f"Staging - Cleaned result for JSON parsing (image): {cleaned_result}")
                parsed = json.loads(cleaned_result)
                # Handle case where parsed is None (e.g., json.loads("null"))
                if parsed is None:
                    logger.info("Staging - Parsed result from image is None (likely 'null' JSON), skipping")
                elif isinstance(parsed, dict):
                    if parsed.get("type") == "parlay":
                        all_picks.append(parsed)  # Single parlay object
                    else:
                        all_picks.append(parsed)  # Single pick
                elif isinstance(parsed, list):
                    all_picks.extend(parsed)  # Multiple single picks
                else:
                    logger.warning(f"Staging - Unexpected JSON format from image: {parsed}")
            except json.JSONDecodeError as e:
                logger.error(f"Staging - Invalid JSON from image: {result} (Error: {e})")

    if not all_picks:
        logger.info("Staging - No picks detected from text or image")
        return None
    elif len(all_picks) == 1:
        logger.info(f"Staging - Single pick or parlay detected: {all_picks[0]}")
        return json.dumps(all_picks[0])  # Single pick or parlay
    else:
        # Multiple picks (either independent or incorrectly grouped parlay picks)
        logger.info(f"Staging - Multiple picks detected: {all_picks}")
        # Check if any pick is a parlay; if so, return it as the primary result
        for pick in all_picks:
            if isinstance(pick, dict) and pick.get("type") == "parlay":
                return json.dumps(pick)
        # Otherwise, return as an array of single picks
        return json.dumps(all_picks)
    

def track_pick(tweet_id, action):
    """
    Track retweets and comments for a tweet in Redis (staging simulation).
    
    Args:
        tweet_id (int): The ID of the tweet.
        action (str): The action to track ('pick', 'retweet', or 'comment').
    """
    key = f"tweet:{tweet_id}"
    try:
        # Simulate tracking in Redis (no actual set in staging)
        logger.info(f"Staging - Simulated tracking {action} for tweet ID {tweet_id}")
    except redis.RedisError as e:
        logger.error(f"Staging - Error simulating tracking {action} for tweet ID {tweet_id}: {e}")

def has_been_actioned(tweet_id, action):
    """
    Check if a specific action (pick, retweet, or comment) has been performed on the tweet (staging simulation).
    
    Args:
        tweet_id (int): The ID of the tweet.
        action (str): The action to check ('pick', 'retweet', or 'comment').
    
    Returns:
        bool: True if the action has been performed (simulated), False otherwise.
    """
    key = f"tweet:{tweet_id}"
    try:
        # Simulate checking in Redis (always return False in staging for testing)
        return False
    except redis.RedisError as e:
        logger.error(f"Staging - Error simulating checking {action} status for tweet ID {tweet_id}: {e}")
        return False  # Assume not actioned if there's an error

def has_commented(tweet_id, my_username):
    """
    Check if the authenticated user has already commented on the tweet (staging simulation).
    
    Args:
        tweet_id (int): The ID of the tweet.
        my_username (str): The username of the authenticated user.
    
    Returns:
        bool: True if a comment exists (simulated), False otherwise.
    """
    try:
        # Simulate checking comments (always return False in staging for testing)
        logger.info(f"Staging - Simulated checking comments for tweet ID {tweet_id}")
        return False
    except tweepy.TweepyException as e:
        logger.error(f"Staging - Error simulating checking comments for tweet ID {tweet_id}: {e}")
        return False  # Assume no comment if there's an error

def poll_tweets():
    """Poll for up to 10 tweets per account from the last 24 hours from Twitter/X accounts in staging mode using Twitter/X v2, filter results to exclude replies, comments, and track picks. Retweets are processed if they contain unique picks. Skips tweets already processed in production with a 1-day TTL for Redis."""
    max_tweets_per_account = 10  # Limit to 10 tweets per account
    total_tweets_processed = 0
    seen_picks = {}  # Dictionary to track unique picks per account: {screen_name: set(pick_str)}

    # Initialize Redis client (assuming it's set up globally or in a config)
    redis_client = redis.Redis(host='localhost', port=6379, db=0)  # Adjust host/port as needed
    PROCESSED_SET_KEY = "processed_tweets"  # Global set for processed tweet IDs
    TTL_SECONDS = 86400  # 1 day in seconds

    # Set or refresh TTL for the processed set
    redis_client.expire(PROCESSED_SET_KEY, TTL_SECONDS)

    # List of Twitter/X account usernames to poll
    accounts_to_poll = ["AlexCaruso"]

    # Define time range (e.g., last 24 hours)
    end_time = datetime.now(timezone.utc)  # Current UTC time
    start_time = end_time - timedelta(hours=24)  # 24 hours ago

    # Format times to RFC3339 (yyyy-MM-dd'T'HH:mm:ss+00:00 or yyyy-MM-dd'T'HH:mm:ssZ)
    start_time_str = start_time.replace(microsecond=0).isoformat()  # Remove microseconds, add UTC offset
    end_time_str = end_time.replace(microsecond=0).isoformat()  # Remove microseconds, add UTC offset

    # Get the authenticated user's ID and username
    try:
        me = client.get_me(user_fields=["id", "username"])
        my_user_id = me.data.id
        my_username = me.data.username
        logger.info(f"Staging - Authenticated user ID: {my_user_id}, Username: {my_username}")
    except tweepy.TweepyException as e:
        logger.error(f"Staging - Error fetching authenticated user info: {e}")
        return

    for screen_name in accounts_to_poll:
        if screen_name not in seen_picks:
            seen_picks[screen_name] = set()
        tweets_processed_for_account = 0
        # Get the user ID for the screen name (using v2 API)
        try:
            logger.info(f"Staging - Looking up user ID for @{screen_name}")
            user = client.get_users(usernames=[screen_name], user_fields=["id"])
            if not user.data or len(user.data) == 0:
                logger.warning(f"Staging - User not found: {screen_name}")
                continue
            user_id = user.data[0].id
            logger.info(f"Staging - Found user ID for @{screen_name}: {user_id}")
        except tweepy.TweepyException as e:
            logger.error(f"Staging - Error fetching user ID for {screen_name}: {e}")
            continue

        # Poll tweets using v2 endpoint for the user, filtering by time
      