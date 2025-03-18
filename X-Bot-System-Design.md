# X Bot System Design

## 1. Overview
A Twitter/X bot for tracking sports betting picks from accounts like Alex Caruso and Dan Gamble AI, using NBA stats to validate picks in real-time.

## 2. Objectives
- Phase 1: Prototype with 5–10 accounts, real-time polling, and pick detection (regex, Grok, Grok vision).
- Phase 2: Scale to thousands of accounts, streaming API, and production-ready features.

## 3. Data Sources
- X API (Basic tier, $200/month).
- Grok API (vision and text, ~$0.46 for Phase 1, ~$1,291/month for Phase 2).
- NBA API (nba_api, free for Phase 1, potential switch to SportsDataIO for Phase 2).

## 4. Pick Detection and Processing
### Regex-Based Detection
- Patterns capture prop bets, spreads, moneyline, and parlays from text-based tweets (e.g., Alex Caruso, Dan Gamble’s text posts).
- Filters promotional content (e.g., "Join here," "VIP," "Drop Play #").

### Grok API Integration
- Falls back to Grok API if regex fails, using a structured prompt to extract picks from text.
- Phase 1 (Prototype): Uses Grok vision for image-based tweets (e.g., Dan Gamble’s graphics) after validating regex for text, testing with 5–10 images.
- Cost Estimate: ~$0.46 for Phase 1 (992,000 tokens), ~$1,291/month for Phase 2 (2.766M tokens).

### Image Processing (Phase 1, After Text Validation)
- Uses Grok vision to extract text and detect picks from image-based tweets (e.g., Dan Gamble’s graphics). Avoids OCR (e.g., `pytesseract`) to leverage Grok’s higher accuracy for stylized text and reduce setup overhead.
- Tests with 5–10 images in Phase 1, ensuring costs remain low (~$0.46 total for Phase 1).

## 5. Result Tracking
- NBA Stats: Uses `nba_api` for live player stats (e.g., free throws, PRA) during the prototype phase. Polls every 5 minutes for real-time tracking, caching results in Redis.
- Real-Time Tracking (Phase 1): Implements polling with `tweepy` (e.g., every 5 minutes) for 5–10 accounts, staying within Basic tier’s 15,000-read limit. Defers streaming to Phase 2 with Pro tier.
- Future Consideration: Switch to SportsDataIO’s NBA API for production to access real-time scores, odds, and historical data, improving scalability and data richness.

## 9. Cost and Efficiency
- API Costs:
  - X API: Basic tier ($200/month) for Phase 1, Pro tier ($5,000/month) for Phase 2.
  - Grok API: ~$0.46 for Phase 1 (992,000 tokens), ~$1,291/month for Phase 2 (2.766M tokens).
  - `nba_api`: Free for Phase 1, with potential switch to SportsDataIO (~$49–$99/month) in Phase 2.
- Optimization: Uses regex-first to reduce Grok usage, caching (Redis) to minimize API calls, and polling to simulate real-time tracking within Basic tier limits.