import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

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
# 1. Task definitions for agrifood systems (now 6 modules)
# ---------------------------------------------------------

@dataclass
class TaskConfig:
    task_type: str          # e.g. "VEG_CLIMATE", "INCOME_CAP", ...
    difficulty: int         # 1..3
    y_var: str
    x_main: List[str]
    description: str
    policy_question: str

# Each task maps directly to your real columns:
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
    # 2) Income vs CAP & rural structure
    "INCOME_CAP": {
        "y_var": "disposable_income_eur",
        "x_main": [
            "cap_support_eur_per_ha",
            "farm_count",
            "viirs_nl_rad_nW",
            "deprivation_index_std",
        ],
        "description": (
            "How do CAP supports, farm structure and local economic activity relate "
            "to disposable income at county level?"
        ),
        "policy_question": (
            "What does this imply for targeting CAP payments and rural development "
            "policies to reduce deprivation?"
        ),
    },
    # 3) Emissions vs intensity
    "EMISSIONS_INTENSITY": {
        "y_var": "ghg_kgco2e_per_ha",
        "x_main": [
            "stocking_rate_lu_per_ha",
            "fertilizer_n_kg_per_ha",
            "pct_arable",
            "pct_pasture",
        ],
        "description": (
            "How does production intensity (stocking rates, fertiliser) relate to "
            "greenhouse gas emissions per hectare?"
        ),
        "policy_question": (
            "How can Ireland design stocking and fertiliser policies that reduce "
            "emissions without collapsing output?"
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
    # 6) Synthetic resilience / system index
    "RESILIENCE_SYNTH": {
        "y_var": "synthetic",
        "x_main": [
            "disposable_income_eur",
            "deprivation_index_std",
            "pct_forest",
            "pct_urban",
            "cap_support_eur_per_ha",
        ],
        "description": (
            "How do income, deprivation and land-use composition relate to a "
            "synthetic resilience or system index for Irish regions?"
        ),
        "policy_question": (
            "How should CAP, social and land-use policies be combined to build "
            "long-run resilience in vulnerable regions?"
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
# 2. Econometric workbench: model library
# ---------------------------------------------------------

MODEL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "POOL_OLS": {
        "label": "Pooled OLS with year dummies",
        "formula_template": "{y} ~ ",   # x's + C(year) appended
    },
    "FE_COUNTY": {
        "label": "Panel with county fixed effects + year dummies",
        "formula_template": "{y} ~ ",   # x's + C(county) + C(year)
    },
    "FE_NUTS3": {
        "label": "Panel with NUTS3 fixed effects + year dummies",
        "formula_template": "{y} ~ ",   # x's + C(nuts3) + C(year)
    },
    # You can later add DID / ECM / spatial here as separate IDs
}

def make_formula(model_id: str, task: TaskConfig) -> str:
    x_part = " + ".join(task.x_main)
    if model_id == "POOL_OLS":
        return f"{task.y_var} ~ {x_part} + C(year)"
    elif model_id == "FE_COUNTY":
        return f"{task.y_var} ~ {x_part} + C(county) + C(year)"
    elif model_id == "FE_NUTS3":
        return f"{task.y_var} ~ {x_part} + C(nuts3) + C(year)"
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

def run_econometric_model(task: TaskConfig, df: pd.DataFrame, model_id: str):
    formula = make_formula(model_id, task)

    # Simple train–test split for out-of-sample performance
    rng = np.random.default_rng(123)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.7 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_model = smf.ols(formula=formula, data=train_df)
    train_res = train_model.fit()

    y_test = test_df[task.y_var].values
    y_pred = train_res.predict(test_df)
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2_test = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return train_res, r2_test

# ---------------------------------------------------------
# 3. Domain priors & reward function (extended to all modules)
# ---------------------------------------------------------

SIGN_PRIORS: Dict[str, Dict[str, int]] = {
    # VEG_CLIMATE
    "VEG_CLIMATE": {
        "rainfall_total_mm": 1,     # more rainfall → higher NDVI (up to a point)
        "s1_soil_moisture": 1,      # more soil moisture → healthier vegetation
        "t2m_mean_c": 1,            # moderate warming can raise NDVI in Ireland
    },
    # INCOME_CAP
    "INCOME_CAP": {
        "cap_support_eur_per_ha": 1,
        "farm_count": 1,
        "viirs_nl_rad_nW": 1,
        "deprivation_index_std": -1,  # more deprivation → lower income
    },
    # EMISSIONS_INTENSITY
    "EMISSIONS_INTENSITY": {
        "stocking_rate_lu_per_ha": 1,
        "fertilizer_n_kg_per_ha": 1,
        "pct_arable": 1,
        "pct_pasture": 1,
    },
    # WATER_BALANCE
    "WATER_BALANCE": {
        "rainfall_total_mm": 1,       # more rainfall → higher NDWI
        "s1_soil_moisture": 1,
        "t2m_mean_c": -1,             # higher temps → more evapotranspiration (lower NDWI)
        "lst_day_summer_c": -1,
    },
    # URBAN_PRESSURE
    "URBAN_PRESSURE": {
        "pct_urban": 1,
        "ndbi_mean": 1,
        "viirs_nl_rad_nW": 1,
        "disposable_income_eur": 1,   # richer & urban → more impervious surface
    },
    # RESILIENCE_SYNTH
    "RESILIENCE_SYNTH": {
        "disposable_income_eur": 1,
        "deprivation_index_std": -1,
        "pct_forest": 1,
        "pct_urban": -1,              # more urban can reduce agro-ecosystem resilience
        "cap_support_eur_per_ha": 1,
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

def compute_reward(task: TaskConfig, res, r2_test: float) -> Tuple[float, Dict[str, float]]:
    # R²-based component
    r2_norm = max(0.0, min(1.0, r2_test))
    # Econ-sign component (policy-aligned priors)
    sign_score = compute_sign_score(task, res)

    # Combined pedagogical reward
    reward = 0.6 * r2_norm + 0.4 * sign_score
    return reward, {"r2_test": r2_test, "r2_norm": r2_norm, "sign_score": sign_score}

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
    alpha = 0.3
    new_avg = (1 - alpha) * state.avg_reward + alpha * reward

    # Curriculum rotates across ALL modules
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
        "Available model IDs are: POOL_OLS, FE_COUNTY, FE_NUTS3. "
        "Respond ONLY with valid JSON with keys: model_id, reasoning, policy_interpretation."
    )

    user_payload = {
        "task_type": task.task_type,
        "task_description": task.description,
        "policy_question": task.policy_question,
        "y_var": task.y_var,
        "x_main": task.x_main,
        "data_summary": desc_stats,
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
# 6. Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="RL Teacher — Agentic Econometric & Policy AI (Irish Agrifood, Extended)",
    layout="wide",
)

st.title("RL Teacher for Agentic Econometric & Policy AI — Irish Agrifood Systems (Extended)")

st.markdown(
    "This prototype uses your **Copernicus + econometric panel for Ireland (2016–2024)** "
    "to simulate how a **Reinforcement Learning Teacher** can train an **Agentic AI "
    "Student** to choose econometric models and reason about agrifood policy trade-offs "
    "across multiple system dimensions.\n\n"
    "- **Teacher**: decides which agrifood learning module and difficulty level to present  \n"
    "- **Student (gpt-4o-mini)**: selects an econometric model (pooled vs FE) and explains it  \n"
    "- **Workbench**: estimates the model on your real panel and computes a learning reward  \n"
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

# Layout
col1, col2 = st.columns([1.2, 1])

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

st.subheader("4. Agentic AI Student (gpt-4o-mini) & Teacher Reward")

if st.button("Run Student Agent & Evaluate Episode", type="primary"):
    with st.spinner("Calling gpt-4o-mini as Agentic Student..."):
        student_out = call_student_agent(client, task, task_df)
    model_id = student_out["model_id"]
    parsed = student_out["parsed"]

    st.markdown(f"**Student-selected Model ID:** `{model_id}`")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Econometric Reasoning**")
        st.write(parsed.get("reasoning", "N/A"))
    with c2:
        st.markdown("**Policy Interpretation**")
        st.write(parsed.get("policy_interpretation", "N/A"))

    # Run econometric model
    res, r2_test = run_econometric_model(task, task_df, model_id)

    st.markdown("### 5. Econometric Results on Your Panel")
    st.markdown(f"- **Out-of-sample R² (test):** `{r2_test:.3f}`")
    with st.expander("Regression summary (statsmodels)", expanded=False):
        st.text(res.summary())

    # Compute reward
    reward, comps = compute_reward(task, res, r2_test)
    st.session_state.last_reward = reward
    st.session_state.last_reward_components = comps

    st.markdown("### 6. Learning Reward (for RL Teacher)")
    st.markdown(f"- **Reward (0–1):** `{reward:.3f}`")
    st.markdown(f"- Components: `{comps}`")

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
    with st.expander("Teacher Commentary (gpt-4o-mini)", expanded=False):
        comment = call_teacher_commentary(client, task, model_id, comps, reward)
        st.write(comment)
else:
    st.info(
        "Click **Run Student Agent & Evaluate Episode** to let the RL Teacher "
        "and Agentic Student interact on all agrifood modules in your real panel."
    )
