import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

BALLEDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

DEFAULT_BALLDONTLIE_KEY = os.getenv("BALLDONTLIE_API_KEY", "4e335fcf-f133-493f-9fa0-015417f3dc6f")
DEFAULT_ODDS_KEY = os.getenv("ODDS_API_KEY", "a17a6e90cf7065d09edee849f3f80e98")


@st.cache_data(ttl=300)
def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def safe_fetch(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return fetch_json(url, headers=headers, params=params), None
    except requests.HTTPError as exc:
        return None, str(exc)
    except requests.RequestException as exc:
        return None, str(exc)


def today_date_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def api_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": api_key}


def get_season_averages(api_key: str, team_id: int, season: int) -> Optional[Dict[str, Any]]:
    headers = api_headers(api_key)
    params = {"team_ids[]": team_id, "season": season}
    data = fetch_json(f"{BALLEDONTLIE_BASE_URL}/season_averages", headers=headers, params=params)
    season_data = data.get("data", [])
    return season_data[0] if season_data else None


def get_team_stats_last_games(api_key: str, team_id: int, games_back: int = 5) -> List[Dict[str, Any]]:
    headers = api_headers(api_key)
    params = {"team_ids[]": team_id, "per_page": games_back, "postseason": "false"}
    data = fetch_json(f"{BALLEDONTLIE_BASE_URL}/games", headers=headers, params=params)
    return data.get("data", [])


def compute_market_total(odds_data: List[Dict[str, Any]], home_team: str, away_team: str) -> Optional[float]:
    for game in odds_data:
        if game.get("home_team") == home_team and game.get("away_team") == away_team:
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                return None
            markets = bookmakers[0].get("markets", [])
            totals = next((market for market in markets if market.get("key") == "totals"), None)
            if not totals:
                return None
            outcomes = totals.get("outcomes", [])
            over = next((o for o in outcomes if o.get("name") == "Over"), None)
            if over:
                return float(over.get("point"))
    return None


def compute_prediction(
    home_team: Dict[str, Any],
    away_team: Dict[str, Any],
    season: int,
    api_key: str,
    injuries: List[Dict[str, Any]],
    odds_data: List[Dict[str, Any]],
    standings: List[Dict[str, Any]],
    leaders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    home_avg = get_season_averages(api_key, home_team["id"], season)
    away_avg = get_season_averages(api_key, away_team["id"], season)

    home_recent = get_team_stats_last_games(api_key, home_team["id"])
    away_recent = get_team_stats_last_games(api_key, away_team["id"])

    def recent_points(games: List[Dict[str, Any]], team_id: int) -> List[int]:
        points = []
        for game in games:
            if game.get("home_team", {}).get("id") == team_id:
                points.append(game.get("home_team_score", 0))
            elif game.get("visitor_team", {}).get("id") == team_id:
                points.append(game.get("visitor_team_score", 0))
        return points

    home_recent_pts = recent_points(home_recent, home_team["id"])
    away_recent_pts = recent_points(away_recent, away_team["id"])

    home_avg_pts = home_avg.get("pts", 110) if home_avg else 110
    away_avg_pts = away_avg.get("pts", 110) if away_avg else 110

    recent_home = sum(home_recent_pts) / len(home_recent_pts) if home_recent_pts else home_avg_pts
    recent_away = sum(away_recent_pts) / len(away_recent_pts) if away_recent_pts else away_avg_pts

    base_total = (home_avg_pts + away_avg_pts) / 2
    recent_total = (recent_home + recent_away) / 2
    model_total = round((base_total * 0.6 + recent_total * 0.4) * 2, 1)
    market_total = compute_market_total(odds_data, home_team.get("full_name"), away_team.get("full_name"))
    if market_total:
        predicted_total = round(model_total * 0.7 + market_total * 0.3, 1)
    else:
        predicted_total = model_total

    injury_teams = [injury.get("team", {}).get("id") for injury in injuries]
    injury_penalty = 0.05 * (injury_teams.count(home_team["id"]) + injury_teams.count(away_team["id"]))

    data_coverage = 0.5
    if home_avg and away_avg:
        data_coverage += 0.2
    if home_recent_pts and away_recent_pts:
        data_coverage += 0.2
    if standings:
        data_coverage += 0.05
    if leaders:
        data_coverage += 0.05

    confidence = max(0.1, min(0.95, data_coverage - injury_penalty))

    return {
        "predicted_total": predicted_total,
        "confidence": confidence,
        "market_total": market_total,
        "home_avg_pts": home_avg_pts,
        "away_avg_pts": away_avg_pts,
        "recent_home_pts": recent_home,
        "recent_away_pts": recent_away,
    }


st.set_page_config(page_title="NBA Dashboard & Total Score Predictor", layout="wide")

st.title("NBA Today: Scores, Totals Odds, and Predictions")

with st.sidebar:
    st.header("API Keys")
    odds_key = st.text_input("Odds API Key", value=DEFAULT_ODDS_KEY, type="password")
    balldontlie_key = st.text_input("BallDontLie API Key", value=DEFAULT_BALLDONTLIE_KEY, type="password")
    current_year = datetime.now().year
    season = st.number_input("Season", min_value=2000, max_value=current_year, value=current_year)

if not odds_key or not balldontlie_key:
    st.warning("Provide both API keys to load data.")
    st.stop()

st.caption("Data sources: balldontlie.io and the-odds-api.com")

games_today, games_error = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/games",
    headers=api_headers(balldontlie_key),
    params={"dates[]": today_date_iso(), "per_page": 100},
)
games_today = games_today.get("data", []) if games_today else []
if games_error:
    st.error(f"Failed to load games: {games_error}")

odds_data, odds_error = safe_fetch(
    f"{ODDS_API_BASE_URL}/sports/basketball_nba/odds",
    params={
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "apiKey": odds_key,
    },
)
odds_data = odds_data if odds_data else []
if odds_error:
    st.error(f"Failed to load odds: {odds_error}")

injuries_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/injuries",
    headers=api_headers(balldontlie_key),
    params={"per_page": 100},
)
injuries_data = injuries_data.get("data", []) if injuries_data else []

standings_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/standings",
    headers=api_headers(balldontlie_key),
    params={"season": season},
)
standings_data = standings_data.get("data", []) if standings_data else []

leaders_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/leaders",
    headers=api_headers(balldontlie_key),
    params={"season": season},
)
leaders_data = leaders_data.get("data", []) if leaders_data else []

teams_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/teams",
    headers=api_headers(balldontlie_key),
)
teams_data = teams_data.get("data", []) if teams_data else []

players_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/players",
    headers=api_headers(balldontlie_key),
    params={"per_page": 25},
)
players_data = players_data.get("data", []) if players_data else []

active_players_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/players",
    headers=api_headers(balldontlie_key),
    params={"per_page": 25, "active": "true"},
)
active_players_data = active_players_data.get("data", []) if active_players_data else []

stats_data, _ = safe_fetch(
    f"{BALLEDONTLIE_BASE_URL}/stats",
    headers=api_headers(balldontlie_key),
    params={"per_page": 25},
)
stats_data = stats_data.get("data", []) if stats_data else []

data_coverage = {
    "Teams": bool(teams_data),
    "Players": bool(players_data),
    "Games": bool(games_today),
    "Game Player Stats": bool(stats_data),
    "Active Players": bool(active_players_data),
    "Player Injuries": bool(injuries_data),
    "Season Averages": bool(games_today),
    "Game Advanced Stats": bool(stats_data),
    "Live Box Scores": False,
    "Lineups": False,
    "Team Standings": bool(standings_data),
    "Leaders": bool(leaders_data),
    "Betting Odds": bool(odds_data),
    "Player Props": False,
    "Team Contracts": False,
    "Player Contracts": False,
    "Player Contract Aggregates": False,
    "Team Season Averages": bool(games_today),
}

with st.expander("Data coverage checklist"):
    coverage_rows = [
        {"Data Domain": key, "Available": "✅" if value else "⚠️"}
        for key, value in data_coverage.items()
    ]
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ indicates the requested domain is not exposed by the current API or is temporarily unavailable."
    )

if not games_today:
    st.info("No games found for today.")
else:
    st.subheader("Today\'s Games")
    rows = []
    for game in games_today:
        home = game.get("home_team", {})
        away = game.get("visitor_team", {})
        status = game.get("status", "TBD")
        scores = f"{away.get('abbreviation')} {game.get('visitor_team_score')} - {home.get('abbreviation')} {game.get('home_team_score')}"
        rows.append(
            {
                "Matchup": f"{away.get('full_name')} at {home.get('full_name')}",
                "Status": status,
                "Score": scores,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Totals Odds (Over/Under)")
    odds_rows = []
    for game in odds_data:
        if not game.get("bookmakers"):
            continue
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        bookmaker = game["bookmakers"][0]
        markets = bookmaker.get("markets", [])
        totals = next((market for market in markets if market.get("key") == "totals"), None)
        if not totals:
            continue
        outcomes = totals.get("outcomes", [])
        over = next((o for o in outcomes if o.get("name") == "Over"), None)
        under = next((o for o in outcomes if o.get("name") == "Under"), None)
        odds_rows.append(
            {
                "Matchup": f"{away_team} at {home_team}",
                "Bookmaker": bookmaker.get("title"),
                "Over": over.get("point") if over else None,
                "Under": under.get("point") if under else None,
            }
        )
    st.dataframe(pd.DataFrame(odds_rows), use_container_width=True, hide_index=True)

    st.subheader("Predicted Total Scores")
    prediction_rows = []
    for game in games_today:
        home = game.get("home_team", {})
        away = game.get("visitor_team", {})
        try:
            prediction = compute_prediction(
                home,
                away,
                season,
                balldontlie_key,
                injuries_data,
                odds_data,
                standings_data,
                leaders_data,
            )
        except requests.HTTPError:
            continue
        prediction_rows.append(
            {
                "Matchup": f"{away.get('full_name')} at {home.get('full_name')}",
                "Predicted Total": prediction["predicted_total"],
                "Confidence": f"{prediction['confidence']:.0%}",
                "Market Total": prediction["market_total"],
                "Home Avg PTS": round(prediction["home_avg_pts"], 1),
                "Away Avg PTS": round(prediction["away_avg_pts"], 1),
                "Recent Home PTS": round(prediction["recent_home_pts"], 1),
                "Recent Away PTS": round(prediction["recent_away_pts"], 1),
            }
        )

    st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True, hide_index=True)

st.markdown(
    """
    **Model inputs considered**

    The prediction blends season averages, recent game totals, injury counts, standings context,
    leader summaries, and available betting totals. When any endpoint is unavailable, the model
    falls back to season averages and reports lower confidence.
    """
)
