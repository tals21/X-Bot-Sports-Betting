# System Design: Sports Betting Tweet Processing Bot

## Overview
This system is designed to monitor Twitter/X accounts for sports betting picks, process them using AI (Grok), and perform actions (retweet, comment) while avoiding duplicates and tracking processed tweets. It is built to run in a staging environment with plans for production deployment using CI/CD, pick tracking

## Architecture
### Components
1. **Twitter/X API Client**
   - **Library**: `tweepy`
   - **Purpose**: Fetches tweets from specified accounts using the Twitter API v2.
   - **Authentication**: Uses a bearer token stored in an environment variable (`X_API_BEARER_TOKEN`).
   - **Data**: Retrieves tweet text, metadata (e.g., `created_at`, `id`), and media (images).

2. **Grok AI Client**
   - **Library**: Custom `grok_client` (assumed)
   - **Purpose**: Analyzes tweet text and images to extract betting picks in JSON format.
   - **Authentication**: Uses an API key (`GROK_API_KEY`) from an environment variable.
   - **Output**: JSON objects for single picks, parlays, or arrays of multiple picks.

3. **Redis Store**
   - **Library**: `redis`
   - **Purpose**: Tracks processed tweet IDs, action history (retweets, comments), and unique picks.
   - **Configuration**: Connects via `REDIS_HOST` and `REDIS_PORT` from environment variables.
   - **TTL**: Processed tweet IDs expire after 1 day (86400 seconds).

4. **Main Script (`prototype.py`)**
   - **Language**: Python 3.9
   - **Functionality**:
     - Polls tweets from predefined accounts every 24 hours.
     - Filters out replies, pinned tweets (>24 hours old), and processed tweets (via Redis).
     - Detects picks using Grok and processes them (single, multiple, or parlay).
     - Simulates retweets and comments, avoiding duplicates.
   - **Dependencies**: `tweepy`, `redis`, `pillow` (for images), `openai`, `python-dotenv`.
   - **Configuration**: Uses environment variables from a `.env` file (not committed; `.env.example` provided for reference). Production secrets managed via GitHub Secrets or secure server config.

5. **CI/CD Pipeline**
   - **Platform**: GitHub Actions
   - **Stages**: Lint (flake8), Test (unittest), Deploy
   - **Environment**: Uses Redis service for testing, deploys to production on `main` push.

## Data Flow
1. **Tweet Fetching**
   - Twitter API polls up to 10 tweets per account from the last 24 hours.
   - Filters applied: Skip replies, pinned tweets (>24 hours old), and processed tweets (via Redis).

2. **Pick Detection**
   - Text and optional images are sent to Grok.
   - Grok returns:
     - Single pick: `{"type": "prop", ...}`
     - Parlay: `{"type": "parlay", "picks": [...], "odds": "..."}`
     - Multiple picks: `[pick1, pick2, ...]`
   - Invalid or result-based content returns `null`.

3. **Processing**
   - Checks for duplicate picks using an in-memory `seen_picks` set.
   - Tracks actions (retweet, comment) in Redis, skipping if already done.
   - Marks tweets as processed in Redis with a 1-day TTL.

4. **Actions**
   - Simulates retweet and posts a comment (e.g., "This pick was retweeted...").
   - Logs all operations for debugging.

## Design Decisions
- **24-Hour Window**: Matches polling frequency and Redis TTL for consistency.
- **Environment Variables**: Secures API keys using `.env` and `python-dotenv`.
- **Duplicate Avoidance**: Combines in-memory (`seen_picks`) and persistent (Redis) checks.
- **Scalability**: Limits to 10 tweets/account to manage API rate limits; extensible with pagination.

## Dependencies
- `tweepy`: Twitter API interaction
- `redis`: Persistent storage
- `pillow`: Image processing
- `openai`: Potential future AI integration
- `python-dotenv`: Environment variable management

## Future Improvements
- **Pagination**: Handle more than 10 tweets per account.
- **Per-Tweet TTL**: Use hashes for individual tweet expirations if needed.
- **Error Handling**: Add retries for API failures.
- **Monitoring**: Integrate logging to a central system (e.g., ELK stack).

## Deployment
- **Staging**: Local or test server with mock data and a generated `.env` for testing.
- **Production**: Deploy via CI/CD to a server, securing `.env` with a deployment script or vault.
- **CI/CD**: GitHub Actions with lint, test, and deploy stages.

## Contact
- **Author**: Akhil
- **Date**: March 18, 2025
