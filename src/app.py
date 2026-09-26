import yaml
import joblib
import pandas as pd
import streamlit as st
import re
import os
import json
from openai import OpenAI
from dotenv import load_dotenv


from src.compare_experiments import get_all_runs, get_best_run
from src.evaluate import load_model_from_run, prediction_interval
from src.train import get_feature_sets
from src.preprocessing import preprocess, get_numeric_feature_cols, encode_position, apply_scaler
from src.player_lookup import load_people, build_name_lookup, find_player, get_latest_season


load_dotenv()

def load_app_config(path='configs/app_config.yaml'):
        with open(path, 'r') as f:
                return yaml.safe_load(f)



@st.cache_resource
def load_resources():
        config= load_app_config()

        runs= get_all_runs(config)
        best_run= get_best_run(runs)

        model= load_model_from_run(best_run['run_id'])
        scaler= joblib.load(config['data']['scaler_path'])


        panel= pd.read_csv(config['data']['panel_path'])
        processed_panel= preprocess(panel)
        people= load_people(config['data']['people_path'])
        name_lookup= build_name_lookup(people)

        feature_sets= get_feature_sets(processed_panel)
        feature_set_key= 'full feature set' if best_run['params.feature_set'] == 'full' else 'Rates Only columns'
        feature_cols= feature_sets[feature_set_key]

        return{
                'config': config,
                'model': model,
                'scaler': scaler,
                'panel': panel,
                'processed_panel': processed_panel,
                'name_lookup': name_lookup,
                'feature_cols': feature_cols,
                'mae': best_run['metrics.mae'],
        }

def build_feature_row(stats, resources):
        """Turn a dictionary of know stat values into a scaled, one-hot encoded
        correctly-ordered single-row DataFram ready for model.predict()."""

        panel= resources['processed_panel']
        scaler= resources['scaler']

        scale_cols= get_numeric_feature_cols(panel)
        defaults= dict(zip(scale_cols, scaler.mean_))

        row= {}
        caveats= []
        for col in scale_cols:
                if stats.get(col) is not None:
                        row[col]= stats[col]
                else:
                        row[col]= defaults[col]
                        caveats.append(col)

        row_df= pd.DataFrame([row])
        row_df['Position']= stats.get('Position') or 'Unknown'
        row_df= encode_position(row_df)
        row_df= apply_scaler(row_df, scaler, scale_cols)

        x= row_df[resources['feature_cols']]
        return x, caveats



required_fields=[
        'AVG',
        'AB',
        'Age'
]

stat_fields= [
        'AVG', 'OBP', 'SLG', 'BB%', 'K%', 'BABIP', 'PA/AB', 'Age',
    'AB', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO',
    'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'G', 'PA', 'Position',
]

real_player_prompt= """You extract a baseball player's name from a user's question.
Return ONLY a JSON object: {'player_name': '<name>'}
If no name is Mentioned, return{'player_name': null}."""

hypothetical_player_prompt= f"""You extract baseball statistics mentioned in a user's message.
Return ONLY a JSON object with these exact keys: {stat_fields}
used the value the user stated for any field the mentioned. Use null for any field not mentioned.
Do not guess or infer values the user did not state."""


hybrid_prompt= f"""You Extract a real baseball player's name and any stated override statistics from a user's message.
Return only a JSON object with this shapeL:
{{'player_name': '<name or null>', 'overrides': {{<any of these keys the user explicitly overrides: value, ...}}}}
Valid Override keys: {stat_fields}
Only include a key in 'overrides' if the user explicitly states a different value than the real player's actual stats."""

response_system_prompt ="""You are a baseball analytics assistant explaining a ML model's prediction.


Strict Rules:

-Only use the numeric values given to you in the user message below. Never use outside knowledge about any real player's actual stats.
-Never invent or guess a number that was not given to you.
-State the predicted next-season AVG and its uncertainty range clearly.
-If any inputs were defaulted(not given by the user), mention this briefly as a caveat.
-Keep the ton clear, conversational, and concise (2-4 sentences)."""


def get_client(config):
        return OpenAI(
    base_url=config['nebius']['base_url'],
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

def strip_reasoning(text):
        """Remove <think> blocks that reasoning models like DeepSeek-R1 may have."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_json(text):
        """Pull the first {...} JSON object out of a model response."""
        cleaned= strip_reasoning(text)
        match= re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if not match:
                raise ValueError(f'No JSON Object found in model output:{text}')
        return json.loads(match.group(0))

def extract_features(client, config, mode, user_text):
        prompts= {
                'real_player': real_player_prompt,
                'hypothetical': hypothetical_player_prompt,
                'hybrid': hybrid_prompt,
        }
        response= client.chat.completions.create(
                model=config['nebius']['extraction_model'],
                temperature= config['nebius']['extraction_temperature'],
                max_tokens= config['nebius']['max_tokens'],
                messages= [
                        {'role': 'system', 'content': prompts[mode]},
                        {'role': 'user', 'content': user_text},
                        ],
        )
        return extract_json(response.choices[0].message.content)

def generate_response(client, config, context):
        response= client.chat.completions.create(
                model=config['nebius']['response_model'],
                temperature=config['nebius']['response_temperature'],
                max_tokens=config['nebius']['max_tokens'],
                messages=[
                        {'role': 'system', 'content': response_system_prompt},
                        {'role': 'user', 'content': json.dumps(context)},
                ],
        )
        return strip_reasoning(response.choices[0].message.content)

def run_prediction(resources, client, stats, context_extra):
        """Given a finalized stats dictionary, predict and display the LLM's explaination."""
        x, caveats= build_feature_row(stats, resources)

        prediction= resources['model'].predict(x)[0]
        low, high= prediction_interval(prediction, resources['mae'])

        context= {**context_extra,
                  'predicted_avg': round(float(prediction), 4),
                  'uncertainty_range': [round(float(low), 4), round(float(high), 4)],
                  'defaulted_fields': caveats,
                  'provided_stats': {k: v for k, v in stats.items() if v is not None},
                  }
        reply= generate_response(client, resources['config'],context)
        st.markdown(reply)
def predict_for_player(resources, client, player_id, overrides, context_extra):
        season_row= get_latest_season(player_id, resources['panel'])
        if season_row is None:
                st.error("That player doesn't have a qualifying season available in this dataset.")
                return
        stats= season_row.to_dict()
        stats.update({k: v for k, v in overrides.items() if v is not None})
        run_prediction(resources, client, stats, context_extra)

def resolve_and_predict(resources, client, player_name, overrides, context_extra):
        match = find_player(player_name, resources['name_lookup'])

        if match['status']== 'found':
                predict_for_player(resources, client, match['player_id'], overrides, context_extra)
        elif match['status']== 'suggestion':
            st.session_state.awaiting= 'confirm_suggestion'
            st.session_state.pending= {
                    'suggested_name': match['suggested_name'],
                    'overrides': overrides,
                    'context_extra': context_extra,
            }
        else:
            st.error("I couldn't determine which player you meant. Please check the spelling and try again.")

def handle_hypotheticals(resources, client, user_text):
        extracted= extract_features(client, resources['config'], 'hypothetical', user_text)

        missing= [f for f in required_fields if extracted.get(f) is None]
        if missing:
                st.warning(f'Ineed a bit more information to make a prediction. Please provide: {", ".join(missing)}')
                return

        run_prediction(resources, client, extracted, {'mode': 'hypothetical'})

def main():
        st.title('Baseball (A)IQ')
        st.write("Ask about a real player's projected batting average, describe a hypothetical player, or mix both.")

        resources= load_resources()
        client= get_client(resources['config'])

        if 'awaiting' not in st.session_state:
            st.session_state.awaiting= None
            st.session_state.pending= {}

        if st.session_state.awaiting == 'confirm_suggestion':
            suggested = st.session_state.pending['suggested_name']
            st.write(f'Did you mean **{suggested}**?')
            col1, col2 = st.columns(2)
            if col1.button('Yes'):
                match = find_player(suggested, resources['name_lookup'])
                predict_for_player(
                    resources, client, match['player_id'],
                    st.session_state.pending['overrides'],
                    st.session_state.pending['context_extra'],
                )
                st.session_state.awaiting = None
            if col2.button('No'):
                st.session_state.awaiting = 'confirm_hypothetical'
            return

        if st.session_state.awaiting == 'confirm_hypothetical':
            st.write('Is this a hypothetical player?')
            col1, col2 = st.columns(2)
            if col1.button('Yes, hypothetical'):
                st.info("Please select 'Hypothetical' mode below and describe the player's stats directly.")
                st.session_state.awaiting = None
            if col2.button('No'):
                st.error('Unable to determine the correct player. Please check the name and try again.')
                st.session_state.awaiting = None
            return

        mode_label= st.radio('Mode', ['Real Player', 'Hypothetical', 'Hybrid'])
        mode= {'Real Player': 'real_player', 'Hypothetical': 'hypothetical', 'Hybrid': 'hybrid'}[mode_label]
        user_text= st.text_area('Describe the player or stats:')

        if st.button('Submit') and user_text:
            if mode== 'hypothetical':
                handle_hypotheticals(resources, client, user_text)
            elif mode== 'real_player':
                extracted= extract_features(client, resources['config'], 'real_player', user_text)
                if not extracted.get('player_name'):
                    st.warning("I couldn't find a player name in your question. Please try again.")
                else:
                    resolve_and_predict(resources, client, extracted['player_name'], {}, {'mode': 'real_player'})
            elif mode== 'hybrid':
                extracted= extract_features(client, resources['config'], 'hybrid', user_text)
                if not extracted.get('player_name'):
                    st.warning("I couldn't find a player name in your question. Please try again.")
                else:
                    overrides= extracted.get('overrides', {})
                    resolve_and_predict(resources, client, extracted['player_name'], overrides, {'mode': 'hybrid'})

if __name__== '__main__':
    main()
