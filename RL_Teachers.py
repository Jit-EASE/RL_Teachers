import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
from openai import OpenAI

# ---------------------------------------------------------
# 0. OpenAI client + data loader
# ---------------------------------------------------------

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not api_key:
        st.error("OPENAI_API_KEY not found. Set it as env var or in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

@st.cache_data
def load_panel_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df

# ---------------------------------------------------------
# 1. Task definitions for agrifood systems (6 modules)
# ---------------------------------------------------------

@dataclass
class TaskConfig:
    task_type: str          # e.g. "VEG_CLIMATE", "INCOME_CAP", ...
    difficulty: int         # 1..3
    y_var: str
    x_main: List[str]
    description: str
    policy_question: str

# Expected columns in your CSV (per previous design):
# year, county, nuts3,
# ndvi_mean, ndwi_mean, ndbi_mean,
# s1_soil_moisture, rainfall_total_mm, t2m_mean_c, lst_day_summer_c,
# no2_column_mol_m2_e5, ch4_ppb,
# viirs_nl_rad_nW, disposable_income_eur, deprivation_index_std, farm_count,
# cattle_head, sheep_head,
# pct_arable, pct_pasture, pct_forest, pct_urban, pct_other,
# impervious_pct, cap_support_eur_per_ha, stocking_rate_lu_per_ha,
# ghg_kgco2e_per_ha, fertilizer_n_kg_per_ha, synthetic

TASK_LIBRARY: Dict[str, Dict[str, Any]] = {
    # 1) Vegetation vs climate & soil
    "VEG_CLIMATE": {
        "y_var": "ndvi_mean",
        "x_main": ["rainfall_total_mm", "t2m_mean_c", "s1_soil_moisture"],
        "description": (
            "How do climate and soil moisture dynamics relate to vegetation health "
            "(NDVI) across Irish counties over time?"
        ),
        "policy_question": (
            "What does this teach us about climate resilience and drought-sensitivity "
            "of Irish agricultural landscapes?"
        ),
    },
    # 2) Income vs CAP & rural structure (now includes GHG)
    "INCOME_CAP": {
        "y_var": "disposable_income_eur",
        "x_main": [
            "cap_support_eur_per_ha",
            "farm_count",
            "viirs_nl_rad_nW",
            "deprivation_index_std",
            "ghg_kgco2e_per_ha",
        ],
        "description": (
            "How do CAP supports, farm structure and local economic activity relate "
            "to disposable income at county level?"
        ),
        "policy_question": (
            "What does this imply for targeting CAP payments and rural development "
            "policies to reduce deprivation while controlling emissions?"
        ),
    },
    # 3) Emissions vs intensity (now includes income)
    "EMISSIONS_INTENSITY": {
        "y_var": "ghg_kgco2e_per_ha",
        "x_main": [
            "stocking_rate_lu_per_ha",
            "fertilizer_n_kg_per_ha",
            "pct_arable",
            "pct_pasture",
            "disposable_income_eur",
        ],
        "description": (
            "How does production intensity (stocking rates, fertiliser) relate to "
            "greenhouse gas emissions per hectare?"
        ),
        "policy_question": (
            "How can Ireland design stocking and fertiliser policies that reduce "
            "emissions without collapsing income?"
        ),
    },
    # 4) Water balance & drought/flood risk
    "WATER_BALANCE": {
        "y_var": "ndwi_mean",
        "x_main": [
            "rainfall_total_mm",
            "s1_soil_moisture",
            "t2m_mean_c",
            "lst_day_summer_c",
        ],
        "description": (
            "How do rainfall, soil moisture and temperature extremes relate to the "
            "water index (NDWI) across Irish counties?"
        ),
        "policy_question": (
            "What does this indicate about drought and flood risk management for "
            "Irish agriculture under climate variability?"
        ),
    },
    # 5) Urban pressure & land take
    "URBAN_PRESSURE": {
        "y_var": "impervious_pct",
        "x_main": [
            "pct_urban",
            "ndbi_mean",
            "viirs_nl_rad_nW",
            "disposable_income_eur",
        ],
        "description": (
            "How does urbanisation and economic activity relate to impervious "
            "surface cover across Irish counties?"
        ),
        "policy_question": (
            "How can spatial planning and rural-urban policy reduce land take and "
            "pressure on agricultural land?"
        ),
    },
    # 6) Synthetic resilience / system index (now includes GHG)
    "RESILIENCE_SYNTH": {
        "y_var": "synthetic",
        "x_main": [
            "disposable_income_eur",
            "deprivation_index_std",
            "pct_forest",
            "pct_urban",
            "cap_support_eur_per_ha",
            "ghg_kgco2e_per_ha",
        ],
        "description": (
            "How do income, deprivation and land-use composition relate to a "
            "synthetic resilience or system index for Irish regions?"
        ),
        "policy_question": (
            "How should CAP, social and land-use policies be combined to build "
            "long-run resilience in vulnerable regions without pushing emissions up?"
        ),
    },
}

def make_task_config(task_type: str, difficulty: int) -> TaskConfig:
    t = TASK_LIBRARY[task_type]
    return TaskConfig(
        task_type=task_type,
        difficulty=difficulty,
        y_var=t["y_var"],
        x_main=t["x_main"],
        description=t["description"],
        policy_question=t["policy_question"],
    )

def build_task_dataset(full_df: pd.DataFrame, task: TaskConfig) -> pd.DataFrame:
    cols = ["year", "county", "nuts3", task.y_var] + task.x_main
    df = full_df[cols].dropna().copy()

    # Difficulty = curriculum pressure:
    # 1: small sample; 2: medium; 3: full panel
    if task.difficulty == 1 and len(df) > 80:
        df = df.sample(80, random_state=42)
    elif task.difficulty == 2 and len(df) > 150:
        df = df.sample(150, random_state=42)

    df.reset_index(drop=True, inplace=True)
    return df

# ---------------------------------------------------------
# 2. Econometric workbench: model library (now with DID + ECM)
# ---------------------------------------------------------

MODEL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "POOL_OLS": {
        "label": "Pooled OLS with year dummies",
    },
    "FE_COUNTY": {
        "label": "Panel with county fixed effects + year dummies",
    },
    "FE_NUTS3": {
        "label": "Panel with NUTS3 fixed effects + year dummies",
    },
    "DID": {
        "label": "Difference-in-Differences with county & year FE",
    },
    "ECM": {
        "label": "Error-Correction–type model on Δy with lagged level",
    },
}

def add_did_columns(task: TaskConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic treated/post structure for DID:
    - treated = counties above median intensity (CAP or stocking/fertiliser) in pre-period
    - post    = years >= median year
    This is heuristic but enough to let RL/Agentic AI learn the pattern.
    """
    df = df.copy()
    if {"treated", "post"}.issubset(df.columns):
        return df

    if df["year"].nunique() < 2:
        df["treated"] = 0
        df["post"] = 0
        return df

    # Choose an intensity variable as policy proxy
    candidate_vars = [
        "cap_support_eur_per_ha",
        "stocking_rate_lu_per_ha",
        "fertilizer_n_kg_per_ha",
    ]
    var = next((v for v in candidate_vars if v in df.columns), None)
    if var is None:
        df["treated"] = 0
        df["post"] = 0
        return df

    median_year = df["year"].median()
    pre = df[df["year"] < median_year].copy()
    if pre.empty:
        pre = df.copy()
    baseline = pre.groupby("county")[var].mean()
    thr = baseline.median()
    treated_map = (baseline > thr).astype(int)

    df["treated"] = df["county"].map(treated_map).fillna(0).astype(int)
    df["post"] = (df["year"] >= median_year).astype(int)
    return df

def build_ecm_df(task: TaskConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple ECM-like dataset:
    - grouped by county over time
    - dy = y_t - y_{t-1}
    - dx0 = x0_t - x0_{t-1}, where x0 is the first main driver
    - include y_{t-1} as lagged level (error-correction term)
    """
    df = df.copy().sort_values(["county", "year"])
    main_x = task.x_main[0]
    df["y_lag1"] = df.groupby("county")[task.y_var].shift(1)
    df[f"{main_x}_lag1"] = df.groupby("county")[main_x].shift(1)

    df["dy"] = df[task.y_var] - df["y_lag1"]
    df["dx0"] = df[main_x] - df[f"{main_x}_lag1"]

    df = df.dropna(subset=["dy", "dx0", "y_lag1"])
    return df

def make_formula(model_id: str, task: TaskConfig) -> str:
    x_part = " + ".join(task.x_main)
    if model_id == "POOL_OLS":
        return f"{task.y_var} ~ {x_part} + C(year)"
    elif model_id == "FE_COUNTY":
        return f"{task.y_var} ~ {x_part} + C(county) + C(year)"
    elif model_id == "FE_NUTS3":
        return f"{task.y_var} ~ {x_part} + C(nuts3) + C(year)"
    elif model_id == "DID":
        # DID with treated*post + controls + FE
        return f"{task.y_var} ~ treated*post + {x_part} + C(county) + C(year)"
    else:
        raise ValueError(f"Formula not defined for model_id: {model_id}")

def run_econometric_model(task: TaskConfig, df: pd.DataFrame, model_id: str):
    """
    Returns:
      res: statsmodels result
      r2_test: out-of-sample R² on y (or dy for ECM)
      target_name: "y" or "dy" (for ECM)
    """
    # ECM case
    if model_id == "ECM":
        df_ecm = build_ecm_df(task, df)
        if df_ecm.empty:
            # Fall back to pooled if ECM has no usable data
            formula = make_formula("POOL_OLS", task)
            train_df = df
            target = task.y_var
        else:
            formula = "dy ~ dx0 + y_lag1 + C(county) + C(year)"
            train_df = df_ecm
            target = "dy"
    else:
        # Possibly add DID structure
        if model_id == "DID":
            df = add_did_columns(task, df)
        formula = make_formula(model_id, task)
        train_df = df
        target = task.y_var

    rng = np.random.default_rng(123)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(0.7 * len(idx))
    if split == 0 or split >= len(idx):
        # Fallback: no meaningful split
        train_idx = idx
        test_idx = idx
    else:
        train_idx, test_idx = idx[:split], idx[split:]

    train_data = train_df.iloc[train_idx]
    test_data = train_df.iloc[test_idx]

    # Fit model
    model = smf.ols(formula=formula, data=train_data)
    res = model.fit()

    # --- Robust prediction step to avoid Patsy category mismatches ---
    try:
        y_pred = res.predict(test_data)
    except Exception:
        # Align categorical levels between train and test
        test_aligned = test_data.copy()
        for col in ["county", "nuts3", "year"]:
            if col in test_aligned.columns and col in train_data.columns:
                test_aligned[col] = pd.Categorical(
                    test_aligned[col],
                    categories=train_data[col].unique()
                )
        y_pred = res.predict(test_aligned)

    y_test = test_data[target].values
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2_test = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return res, r2_test, target

# ---------------------------------------------------------
# 3. Domain priors & reward function (with policy constraint)
# ---------------------------------------------------------

SIGN_PRIORS: Dict[str, Dict[str, int]] = {
    "VEG_CLIMATE": {
        "rainfall_total_mm": 1,
        "s1_soil_moisture": 1,
        "t2m_mean_c": 1,
    },
    "INCOME_CAP": {
        "cap_support_eur_per_ha": 1,
        "farm_count": 1,
        "viirs_nl_rad_nW": 1,
        "deprivation_index_std": -1,
        "ghg_kgco2e_per_ha": -1,  # higher GHG should not be 'good' for income
    },
    "EMISSIONS_INTENSITY": {
        "stocking_rate_lu_per_ha": 1,
        "fertilizer_n_kg_per_ha": 1,
        "pct_arable": 1,
        "pct_pasture": 1,
        "disposable_income_eur": 1,  # richer/intensive areas expected to have higher emissions
    },
    "WATER_BALANCE": {
        "rainfall_total_mm": 1,
        "s1_soil_moisture": 1,
        "t2m_mean_c": -1,
        "lst_day_summer_c": -1,
    },
    "URBAN_PRESSURE": {
        "pct_urban": 1,
        "ndbi_mean": 1,
        "viirs_nl_rad_nW": 1,
        "disposable_income_eur": 1,
    },
    "RESILIENCE_SYNTH": {
        "disposable_income_eur": 1,
        "deprivation_index_std": -1,
        "pct_forest": 1,
        "pct_urban": -1,
        "cap_support_eur_per_ha": 1,
        "ghg_kgco2e_per_ha": -1,
    },
}

def compute_sign_score(task: TaskConfig, res) -> float:
    priors = SIGN_PRIORS.get(task.task_type, {})
    if not priors:
        return 0.5
    matches = 0
    total = 0
    for var, sign in priors.items():
        if var in res.params.index:
            beta = res.params[var]
            if beta != 0:
                total += 1
                if beta * sign > 0:
                    matches += 1
    if total == 0:
        return 0.5
    return matches / total

def compute_policy_score(task: TaskConfig,
                         task_df: pd.DataFrame,
                         full_df: pd.DataFrame) -> float:
    """
    Policy-constraint score:
    - If both disposable_income_eur and ghg_kgco2e_per_ha present in task_df,
      compute normalized mean income and GHG, and reward higher income with lower GHG.
    - Otherwise return neutral 0.5.
    Score in [0,1].
    """
    cols = task_df.columns
    if ("disposable_income_eur" not in cols) or ("ghg_kgco2e_per_ha" not in cols):
        return 0.5

    inc_mean = float(task_df["disposable_income_eur"].mean())
    ghg_mean = float(task_df["ghg_kgco2e_per_ha"].mean())

    # Global ranges for normalization
    inc_min = float(full_df["disposable_income_eur"].min())
    inc_max = float(full_df["disposable_income_eur"].max())
    ghg_min = float(full_df["ghg_kgco2e_per_ha"].min())
    ghg_max = float(full_df["ghg_kgco2e_per_ha"].max())

    if inc_max > inc_min:
        inc_norm = (inc_mean - inc_min) / (inc_max - inc_min)
    else:
        inc_norm = 0.5

    if ghg_max > ghg_min:
        ghg_norm = (ghg_mean - ghg_min) / (ghg_max - ghg_min)
    else:
        ghg_norm = 0.5

    # Raw score: income minus GHG (higher better), clipped to [-1,1]
    raw = inc_norm - ghg_norm
    raw = max(-1.0, min(1.0, raw))
    # Map to [0,1]
    score = 0.5 * (raw + 1.0)
    return float(score)

def compute_reward(task: TaskConfig,
                   res,
                   r2_test: float,
                   task_df: pd.DataFrame,
                   full_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Final reward = weighted combination of:
    - Econ/stat score (R² + sign priors)
    - Policy score (income vs GHG constraint)
    """
    r2_norm = max(0.0, min(1.0, r2_test))
    sign_score = compute_sign_score(task, res)
    econ_score = 0.6 * r2_norm + 0.4 * sign_score

    policy_score = compute_policy_score(task, task_df, full_df)

    # Blend: you can tweak weights if you want to emphasise climate constraint more
    reward = 0.7 * econ_score + 0.3 * policy_score

    comps = {
        "r2_test": r2_test,
        "r2_norm": r2_norm,
        "sign_score": sign_score,
        "econ_score": econ_score,
        "policy_score": policy_score,
    }
    return reward, comps

# ---------------------------------------------------------
# 4. RL Teacher state & update rule (curriculum over 6 modules)
# ---------------------------------------------------------

@dataclass
class TeacherState:
    difficulty: int
    last_task_type: str
    episode: int
    avg_reward: float

def init_teacher_state() -> TeacherState:
    return TeacherState(
        difficulty=1,
        last_task_type="VEG_CLIMATE",
        episode=0,
        avg_reward=0.0,
    )

def teacher_policy_update(state: TeacherState, reward: float) -> TeacherState:
    """
    Simple heuristic 'RL' teacher to start with.
    You can later replace this update with a policy learned offline
    from the logged episodes (e.g. PPO / DQN).
    """
    alpha = 0.3
    new_avg = (1 - alpha) * state.avg_reward + alpha * reward

    task_order = [
        "VEG_CLIMATE",
        "INCOME_CAP",
        "EMISSIONS_INTENSITY",
        "WATER_BALANCE",
        "URBAN_PRESSURE",
        "RESILIENCE_SYNTH",
    ]

    if reward > 0.7:
        new_difficulty = min(3, state.difficulty + 1)
        idx = task_order.index(state.last_task_type)
        new_task_type = task_order[(idx + 1) % len(task_order)]
    elif reward < 0.3:
        new_difficulty = max(1, state.difficulty - 1)
        new_task_type = state.last_task_type
    else:
        new_difficulty = state.difficulty
        new_task_type = state.last_task_type

    return TeacherState(
        difficulty=new_difficulty,
        last_task_type=new_task_type,
        episode=state.episode + 1,
        avg_reward=new_avg,
    )

# ---------------------------------------------------------
# 5. Agentic AI Student (gpt-4o-mini)
# ---------------------------------------------------------

def call_student_agent(client: OpenAI, task: TaskConfig, df: pd.DataFrame) -> Dict[str, Any]:
    desc_stats = df[[task.y_var] + task.x_main].describe().to_dict()

    system_prompt = (
        "You are an agentic AI learner specialising in econometrics and strategic "
        "policy analysis for agrifood systems in Ireland. "
        "You are being taught by a Reinforcement Learning teacher. "
        "Given a task description, a policy question, a data summary, and a list of "
        "candidate model IDs, you must choose the most appropriate model ID and explain why. "
        "Available model IDs are: POOL_OLS, FE_COUNTY, FE_NUTS3, DID, ECM. "
        "Use DID only when a policy-like pre/post structure with treated vs control regions "
        "could exist, and ECM when dynamic adjustment over time matters. "
        "Respond ONLY with valid JSON with keys: model_id, reasoning, policy_interpretation."
    )

    user_payload = {
        "task_type": task.task_type,
        "task_description": task.description,
        "policy_question": task.policy_question,
        "y_var": task.y_var,
        "x_main": task.x_main,
        "data_summary": desc_stats,
        "available_models": MODEL_LIBRARY,
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.2,
    )
    raw = completion.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "model_id": "POOL_OLS",
            "reasoning": f"Parsing error; defaulting to POOL_OLS. Raw: {raw}",
            "policy_interpretation": "N/A",
        }

    model_id = parsed.get("model_id", "POOL_OLS")
    if model_id not in MODEL_LIBRARY:
        model_id = "POOL_OLS"

    return {"raw_response": raw, "parsed": parsed, "model_id": model_id}

def call_teacher_commentary(
    client: OpenAI,
    task: TaskConfig,
    model_id: str,
    reward_components: Dict[str, float],
    reward: float,
) -> str:
    system_prompt = (
        "You are an RL-style pedagogical teacher for econometrics and agrifood "
        "policy analysis. Provide short, constructive feedback to an AI student "
        "on their model choice and what they should learn next. "
        "Do NOT apologise or mention being an AI; just speak as a teacher."
    )
    user_payload = {
        "task_type": task.task_type,
        "difficulty": task.difficulty,
        "model_id": model_id,
        "reward": reward,
        "reward_components": reward_components,
        "y_var": task.y_var,
        "x_main": task.x_main,
        "description": task.description,
        "policy_question": task.policy_question,
    }
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content

# ---------------------------------------------------------
# 6. Episode logging for offline RL training
# ---------------------------------------------------------

LOG_PATH = "rl_teacher_episodes.csv"

def log_episode(episode_record: Dict[str, Any]):
    df_log = pd.DataFrame([episode_record])
    file_exists = os.path.exists(LOG_PATH)
    df_log.to_csv(LOG_PATH, mode="a", index=False, header=not file_exists)

# ---------------------------------------------------------
# 7. Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="RL Teacher — Agentic Econometric & Policy AI (Irish Agrifood, Full)",
    layout="wide",
)

st.title("RL Teacher for Agentic Econometric & Policy AI — Irish Agrifood Systems")

st.markdown(
    "This prototype uses **Copernicus + econometric panel for Ireland (2016–2024)** "
    "to simulate how a **Reinforcement Learning Teacher** can train an **Agentic AI "
    "Student** to choose econometric models and reason about agrifood policy trade-offs.\n\n"
    "- **Teacher**: decides which agrifood learning module and difficulty level to present  \n"
    "- **Student **: selects an econometric model (pool / FE / DID / ECM) and explains it  \n"
    "- **Workbench**: estimates the model on your real panel and computes a reward that blends "
    "statistical fit, sign priors, and a **GHG vs income policy constraint**  \n"
    "- **Logger**: writes each episode to `rl_teacher_episodes.csv` so user can train a proper RL "
    "teacher policy offline (PPO/DQN/etc.) and later plug it back in.\n"
)

DATA_PATH = "ie_copernicus_agri_econ_panel_2016_2024.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}. Place your CSV in this path.")
    st.stop()

full_df = load_panel_data(DATA_PATH)
client = get_openai_client()

# Initialise session state
if "teacher_state" not in st.session_state:
    st.session_state.teacher_state = init_teacher_state()
if "last_reward" not in st.session_state:
    st.session_state.last_reward = None
if "last_reward_components" not in st.session_state:
    st.session_state.last_reward_components = {}
if "last_episode_model" not in st.session_state:
    st.session_state.last_episode_model = None

teacher_state: TeacherState = st.session_state.teacher_state

# Sidebar
st.sidebar.header("Teacher Controls")

mode = st.sidebar.selectbox(
    "Teacher Mode",
    ["RL Teacher (auto curriculum)", "Manual override"],
)

if mode == "Manual override":
    manual_task_type = st.sidebar.selectbox(
        "Learning module (task type)",
        list(TASK_LIBRARY.keys()),
        index=list(TASK_LIBRARY.keys()).index(teacher_state.last_task_type),
    )
    manual_difficulty = st.sidebar.slider(
        "Difficulty", min_value=1, max_value=3, value=teacher_state.difficulty
    )
else:
    manual_task_type = None
    manual_difficulty = None

st.sidebar.markdown("### Current Teacher State")
st.sidebar.json(asdict(teacher_state))

st.sidebar.markdown("### Last Reward")
st.sidebar.write(st.session_state.last_reward)
st.sidebar.write(st.session_state.last_reward_components)

if os.path.exists(LOG_PATH):
    with st.sidebar.expander("Recent logged episodes", expanded=False):
        try:
            log_tail = pd.read_csv(LOG_PATH).tail(10)
            st.dataframe(log_tail)
        except Exception:
            st.write("Could not read log.")

# Layout
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("1. Panel Data Overview")
    st.markdown("**Rows:** {}  |  **Years:** {}–{}".format(
        len(full_df),
        full_df["year"].min(),
        full_df["year"].max(),
    ))
    st.markdown("**Counties:** {}  |  **NUTS3 regions:** {}".format(
        full_df["county"].nunique(),
        full_df["nuts3"].nunique(),
    ))
    with st.expander("Preview panel (head)", expanded=False):
        st.dataframe(full_df.head())

with col2:
    st.subheader("2. Learning Modules (Agrifood Tasks)")
    for key, cfg in TASK_LIBRARY.items():
        st.markdown(f"- **{key}** — {cfg['description']}")
    st.markdown("**Econometric Model Library**")
    for mid, info in MODEL_LIBRARY.items():
        st.markdown(f"- **{mid}** — {info['label']}")

st.markdown("---")
st.subheader("3. Current Episode — Task Generation from Real Data")

if mode == "RL Teacher (auto curriculum)":
    task_type = teacher_state.last_task_type
    difficulty = teacher_state.difficulty
else:
    task_type = manual_task_type
    difficulty = manual_difficulty

task = make_task_config(task_type, difficulty)
task_df = build_task_dataset(full_df, task)

st.markdown(f"**Module (Task Type):** `{task.task_type}`  |  **Difficulty:** `{task.difficulty}`")
st.markdown(f"**Question:** {task.description}")
st.markdown(f"**Policy Angle:** {task.policy_question}")
st.markdown(f"**Outcome (y):** `{task.y_var}`  |  **Drivers (x):** `{', '.join(task.x_main)}`")

with st.expander("Task dataset (sample)", expanded=False):
    st.dataframe(task_df.head())

with st.expander("Summary statistics (y and x)", expanded=False):
    st.write(task_df[[task.y_var] + task.x_main].describe())

st.subheader("4. Agentic AI Student & Teacher Reward")

if st.button("Run Student Agent & Evaluate Episode", type="primary"):
    with st.spinner("Calling gpt-4o-mini as Agentic Student..."):
        student_out = call_student_agent(client, task, task_df)
    model_id = student_out["model_id"]
    parsed = student_out["parsed"]
    st.session_state.last_episode_model = model_id

    st.markdown(f"**Student-selected Model ID:** `{model_id}`")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Econometric Reasoning**")
        st.write(parsed.get("reasoning", "N/A"))
    with c2:
        st.markdown("**Policy Interpretation**")
        st.write(parsed.get("policy_interpretation", "N/A"))

    # Run econometric model
    res, r2_test, target_name = run_econometric_model(task, task_df, model_id)

    st.markdown("### 5. Econometric Results on Your Panel")
    st.markdown(f"- **Target for R²:** `{target_name}`")
    st.markdown(f"- **Out-of-sample R² (test):** `{r2_test:.3f}`")
    with st.expander("Regression summary (statsmodels)", expanded=False):
        st.text(res.summary())

    # Compute reward
    reward, comps = compute_reward(task, res, r2_test, task_df, full_df)
    st.session_state.last_reward = reward
    st.session_state.last_reward_components = comps

    st.markdown("### 6. Learning Reward (for RL Teacher)")
    st.markdown(f"- **Reward (0–1):** `{reward:.3f}`")
    st.markdown(f"- Components: `{comps}`")

    # Log episode for offline RL training
    episode_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "episode": teacher_state.episode,
        "mode": mode,
        "task_type": task.task_type,
        "difficulty": task.difficulty,
        "model_id": model_id,
        "reward": reward,
        **{f"comp_{k}": v for k, v in comps.items()},
    }
    log_episode(episode_record)

    # Teacher update
    if mode == "RL Teacher (auto curriculum)":
        new_state = teacher_policy_update(teacher_state, reward)
        st.session_state.teacher_state = new_state
        st.success(
            f"Teacher updated → episode {new_state.episode}, "
            f"next module: {new_state.last_task_type}, "
            f"difficulty: {new_state.difficulty}, "
            f"avg_reward: {new_state.avg_reward:.3f}"
        )
    else:
        st.info("Manual mode: teacher state not auto-updated.")

    # Teacher commentary (LLM)
    with st.expander("Teacher Commentary", expanded=False):
        comment = call_teacher_commentary(client, task, model_id, comps, reward)
        st.write(comment)
else:
    st.info(
        "Click **Run Student Agent & Evaluate Episode** to let the RL Teacher "
        "and Agentic Student interact on all agrifood modules in your real panel."
    )
