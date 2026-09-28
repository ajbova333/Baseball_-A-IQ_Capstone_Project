# Baseball (A)IQ

Baseball (A)IQ predicts a player's **next-season batting average (AVG)** from their **current-season** batting stats, and lets you ask for that prediction in plain English instead of filling out a form. It's built for anyone curious about a real MLB player's projected performance, or about a hypothetical "what if" player profile — a fan, a fantasy-baseball player, or anyone wanting a lightweight, data-backed second opinion on a player's projected AVG.

This is a capstone project combining a trained regression model (scikit-learn, tracked with MLflow) with an LLM interface (Nebius AI Studio) that parses a user's question, runs it through the trained model, and explains the result in plain language.

## What it does

Ask about a real player, a completely hypothetical one, or a mix of both:

- **Real Player**: "What was Mike Trout's batting average last year?" — pulls his actual last qualifying season from the dataset and predicts his next season's AVG.
- **Hypothetical**: "I'm a 32-year-old left fielder with a 15% strikeout rate — what's my predicted average?" — builds a prediction from whatever you state, filling in anything unstated with a league-average default.
- **Hybrid**: "What would Mike Trout's average be if his OBP was .672?" — starts from a real player's actual stats, then overrides just the field(s) you specify.

Every response includes the predicted AVG and a typical error range, not a bare number, and the system asks a clarifying question rather than guessing when it doesn't have enough information (or the wrong player name).

## Setup

1. **Clone the repo and install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Get the data**: Download the [Lahman Baseball Database](https://www.kaggle.com/datasets/dalyas/lahman-baseball-database) from Kaggle and place `Batting.csv`, `People.csv`, and `Fielding.csv` into `data/raw/`.

3. **Configure your API key**: Copy `.env.example` to `.env` and fill in your Nebius AI Studio key:
   ```
   NEBIUS_API_KEY=your_key_here
   ```
   Never commit `.env` — it's already excluded via `.gitignore`.

## Usage

Run the pipeline in order the first time:

```
python -m src.build_panel        # builds the lagged training panel from raw data
python -m src.train               # trains 5 model configs, logs everything to MLflow
python -m src.compare_experiments # prints the current best run
```

Then launch the app:

```
PYTHONPATH=. streamlit run src/app.py
```

Pick a mode (Real Player / Hypothetical / Hybrid), describe what you want in the text box, and submit. If the name you typed is a close misspelling of a real player, the app will ask "did you mean X?" before proceeding.

To inspect the MLflow experiment tracking UI:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run the test suite with:
```
python -m pytest tests/ -v
```

## Architecture

```
Lahman CSVs → build_panel.py → panel.csv (lagged: season N stats → season N+1 AVG)
                                     │
                              preprocessing.py (fill/encode Position, scale numerics)
                                     │
                    train.py → 5 MLflow runs (LinearRegression, RandomForest,
                    GradientBoosting, best-of-3 on rates-only features,
                    GridSearchCV-tuned winner) → sqlite:///mlflow.db
                                     │
                    compare_experiments.py → picks current best run via
                    mlflow.search_runs()
                                     │
                    app.py (Streamlit) ─┬─ loads winning model + fitted scaler
                                         ├─ player_lookup.py (name matching)
                                         └─ Nebius LLM, two calls:
                                             1. low-temp extraction → structured JSON
                                             2. response generation, constrained to
                                                only the numbers the code hands it
```

The model and the LLM are deliberately kept separate: the LLM never predicts a batting average itself, and the model never touches natural language. The LLM's first call turns a free-text question into structured feature values (with a code-level `normalize_stats` safety net that catches common format mismatches, like a stated "15%" being extracted as the raw number `15` instead of `0.15`); the trained model produces the actual number; a second LLM call explains that number in plain language, explicitly forbidden from citing any number it wasn't handed. The predicted AVG and its uncertainty range are rendered directly by the app's own code, not narrated by the LLM, since testing showed the LLM would occasionally transcribe those specific numbers incorrectly even when explicitly instructed to relay them verbatim.

## Results

Five MLflow runs were compared (chronological train/test split, held constant across all five): Linear Regression and two ensemble methods on the full feature set, the best of those three re-run on a rates-only feature set, and a `GridSearchCV`-tuned variant of the overall winner.

**Best model: tuned Linear Regression, full feature set**
| Metric | Value |
|---|---|
| MAE | 0.0234 |
| RMSE | 0.0298 |
| R² | 0.2247 |

Linear Regression won outright — neither Random Forest nor Gradient Boosting beat it on the full feature set, and the rates-only feature set didn't beat the full one either. That's a real, interesting finding: the relationship between a player's current-season stats and their next-season AVG appears to be largely linear and mean-reverting, without much nonlinear structure for the ensemble methods to exploit. Tuning the winner's one adjustable hyperparameter (`fit_intercept`) produced an identical result to the untuned baseline, which makes sense in hindsight — there wasn't much room to tune in the first place.

An MAE of ~0.023 on a stat that typically ranges from .200–.350 is a reasonable result for this task; predicting next-season batting average from a single prior season is a well-known hard problem in sabermetrics, given how much regression to the mean affects individual seasons.

## Reflection

The hardest part of this project wasn't the model — it was making the LLM interface trustworthy. Repeated testing surfaced the same lesson in several different forms: **an LLM will not reliably transcribe or format numbers correctly, even under explicit instructions telling it not to invent or alter them.** Over the course of testing, the response-generation model fabricated a "95% confidence" label that was never computed, invented specific stat values it was never given, and once reported a completely different (and internally contradictory) prediction range than the one actually computed. The fix that actually worked wasn't better prompt wording — it was removing the opportunity for the mistake entirely: rendering the predicted number and its range directly from Python instead of asking the LLM to restate them, and defensively normalizing extracted values (e.g., percentage-vs-decimal, spelled-out positions vs. abbreviations) in code rather than trusting the model to always format them the way the pipeline expects.

A second, smaller lesson: a prompt rule that includes a concrete illustrative example with realistic-looking numbers runs the risk of the model copying those exact numbers verbatim, even when the rule's condition doesn't apply to the current input. Illustrative examples in a system prompt are safest when they're clearly generic, not numerically specific.

With more time, I'd want to: build genuine statistical prediction intervals instead of a simple `prediction +/- MAE` rule of thumb; extend the model beyond AVG to other counting/rate stats; properly handle the handful of real players who share the exact same name — right now the system just picks whichever one played most recently instead of asking the user which one they meant; and explore collapsing the three explicit input modes into a single free-text box with the LLM inferring intent, now that the underlying prediction/extraction pipeline is solid enough to build that on top of.
