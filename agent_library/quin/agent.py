"""
Quin — SQL / structured-data agent.

Split out of the unified AutoGen pipeline (main.py). Quin owns the
SQL-based analysis path end-to-end:

    Sql_Generator -> Query_Executor -> Sql_tool -> Sql_Execution_Critic
        -> Insight_Generator -> evaluation_agent

Runtime credentials and reference data come from ``.runtime_config``
(env vars — no ambient project-root imports). It does not import anything
from the `eryl` package, and `eryl` does not import anything from here —
cross-flow handoffs are expressed as plain agent-name strings that a
parent orchestrator resolves against its full agent registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import text  # noqa: F401  (kept for parity with original main.py; not used directly here)
from autogen import AssistantAgent
from autogen.oai.client import OpenAIWrapper

from .runtime_config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4_LLM_MODEL_DEPLOYMENT_NAME,
    get_all_table_schemas,
    get_engine,
)


# ---------------------------------------------------------------------------
# AutoGen strips dots from Azure deployment names (gpt-5.4 -> gpt-54).
# Preserve the deployment name exactly as configured in Azure.
# Patching OpenAIWrapper is process-global, so both packages apply the same
# patch defensively (idempotent — the second apply just re-assigns the same
# function).
# ---------------------------------------------------------------------------
def _configure_azure_openai_preserve_dots(self, config, openai_config):
    openai_config["azure_deployment"] = openai_config.get("azure_deployment", config.get("model"))
    openai_config["azure_endpoint"] = openai_config.get("azure_endpoint", openai_config.pop("base_url", None))
    if openai_config.get("azure_ad_token_provider") == "DEFAULT":
        import azure.identity
        openai_config["azure_ad_token_provider"] = azure.identity.get_bearer_token_provider(
            azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )


OpenAIWrapper._configure_azure_openai = _configure_azure_openai_preserve_dots


LLM_CONFIG = {
    "config_list": [
        {
            "model": GPT4_LLM_MODEL_DEPLOYMENT_NAME,
            "azure_deployment": GPT4_LLM_MODEL_DEPLOYMENT_NAME,
            "api_type": "azure",
            "base_url": AZURE_OPENAI_API_BASE,
            "api_key": AZURE_OPENAI_API_KEY,
            "api_version": AZURE_OPENAI_API_VERSION,
        }
    ],
    "temperature": 0.1,
}


POST_GENERATION_EVALUATION_RUBRIC = """
## Post-Generation Evaluation Dimensions (score each 0–10 unless noted)

| Dimension | What to assess |
|-----------|----------------|
| **helpfulness** | How useful the answer is for the user |
| **relevance** | Alignment with the question (and prior turn context when applicable) |
| **level_of_detail** | Sufficient depth without unnecessary filler |
| **groundedness** | Supported by SQL results, retrieved context, or prior verified answers |
| **completeness** | All parts of the question (and multi-turn thread) addressed |
| **faithfulness** | No distortion of source data or context |
| **content_safety** | No harmful, toxic, or unsafe generated content |

Also assess generated **output** (not just the user question):
- **output_safety**: `pii_leak` (bool), `toxic_output` (bool), `score` (0–10)
- **output_governance**: `restricted_data_exposed` (bool), `score` (0–10)
- **output_compliance**: `policy_violation` (bool), `score` (0–10)

## Score calibration (required)
- Do **not** assign all dimensions 10 for normal successful answers.
- Use **differentiated** scores that reflect real quality (e.g. helpfulness 9, level_of_detail 8, groundedness 10).
- Reserve **10** for exceptional performance; **7–9** is typical for good answers.

## Composite score & status (required — compute exactly)
```
composite_score = round(
  (helpfulness + relevance + level_of_detail + groundedness + completeness
   + faithfulness + content_safety
   + output_safety.score + output_governance.score + output_compliance.score) / 10
, 1)
status = "PASS" if composite_score >= 7 else "FAIL"
```

## Confidence, timestamp, and blocked_reason
- Always include `confidence`: `"HIGH"`, `"MEDIUM"`, or `"LOW"` (based on evidence strength and context coverage).
- Always include `evaluation_timestamp` as ISO 8601 UTC at evaluation time (e.g. `"2026-05-28T14:30:00.000Z"`).
- Include `blocked_reason` **only** when `status` is `"FAIL"`. **Omit** `blocked_reason` on PASS.

## Dependent / multi-turn questions
When `analysis_type` is `Both-dependent`, `SQL-Eryl-dependent`, `Eryl-SQL-dependent`, or prior turns exist in the conversation:
- Use **initial_question**, **previous user/updated questions**, **previous query_answer / llm_answer**, and **current answer**.
- For **relevance**, **groundedness**, **completeness**, and business-rule adherence: verify consistency across turns and that the current answer correctly follows prior context.
- Penalize contradictions with earlier turns or answers that ignore established facts from prior steps.
"""


def evaluation_timestamp_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def compute_post_generation_composite(evaluation: dict) -> tuple:
    """Deterministic composite_score and status from standardized evaluation fields."""
    def _score(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    numeric_scores = [
        _score(evaluation.get("helpfulness")),
        _score(evaluation.get("relevance")),
        _score(evaluation.get("level_of_detail")),
        _score(evaluation.get("groundedness")),
        _score(evaluation.get("completeness")),
        _score(evaluation.get("faithfulness")),
        _score(evaluation.get("content_safety")),
        _score((evaluation.get("output_safety") or {}).get("score")),
        _score((evaluation.get("output_governance") or {}).get("score")),
        _score((evaluation.get("output_compliance") or {}).get("score")),
    ]
    composite = round(sum(numeric_scores) / len(numeric_scores), 1)
    status = "PASS" if composite >= 7 else "FAIL"
    return composite, status


def build_agents(llm_config: Optional[dict] = None) -> dict:
    """
    Factory that builds a fresh set of Quin's SQL-flow AssistantAgents and
    registers Quin's one tool function (`db_execute_query`) on them.

    Returns a dict keyed by agent name, matching the names used in the
    original unified `state_transition` / `GroupChat`, so a parent
    orchestrator can drop these straight into its own agent registry:

        {
          "Sql_Generator": ...,
          "Query_Executor": ...,
          "Sql_tool": ...,
          "Sql_Execution_Critic": ...,
          "Insight_Generator": ...,
          "evaluation_agent": ...,
        }

    A factory (rather than module-level singletons) is used because
    AssistantAgent instances are stateful (auto-reply counters, chat
    history) and Quin may need to be instantiated more than once
    (e.g. once per conversation/session).
    """
    cfg = llm_config or LLM_CONFIG

    Sql_Generator = AssistantAgent(
        name="Sql_Generator",
        system_message=f"""
You are the Sql_Generator Agent. Your role is to use the provided metadata to analyze SQL-based questions and generate a valid SQL query to fetch the required information from the database. Ensure the query is accurate and valid according to the table schema. Once formed, return the query.

**Steps**:
1. **Parse the Question**:
- Analyze the user's question and identify the relevant fields based on the metadata.

2. **Meta_Data**:
    {get_all_table_schemas()}

Form the SQL Query:
    1. Based on the identified fields, construct a valid SQL query.
    2. Ensure that the query matches the metadata schema (i.e., use the correct column names and data types).
    3. Use TOP instead of LIMIT.

Example:
For a question like "Which city has the largest population?", the expected SQL query would be:
SELECT TOP 1 city_name, population FROM city_stats ORDER BY population DESC;
Handle Invalid Queries:

If the question references fields not available in the metadata, or if it's impossible to form a valid SQL query, return:

sql_query: None
sql_question: separate the SQL-based question from the initial question.
Return the Output in strictly json format.
    Once a valid SQL query is generated, return the result in the specified format.
    If no valid query can be formed, return None for the sql_query field.

# Output Format strictly json format.
For valid SQL queries:

{{
"initial_question": "Original question from the user",
"analysis_type": "Dont change the type,keep the same selected by routing agent",
"sql_question": "sql question",
"sql_query": "SQL query",
"db_name": "SQL",
"next_agent": "Query_Executor"
}}

For invalid queries (no matching metadata):
{{
"initial_question": "Original question from the user",
"analysis_type": "Dont change the type,keep the same selected by routing agent",
"sql_question": "sql question",
"sql_query": None,
"db_name": "SQL",
"next_agent": "Query_Executor"
}}

Your Objective:
    Ensure that all SQL queries generated are valid according to the provided metadata.
    Return the appropriate query or an error message if no valid SQL query can be formed.
    pass the query to Query_Executor to get the result.
⚠️ **Important**:  
    ✅ **You must always return the output in a strict JSON format as specified below.**  
    ❌ **Never return plain text or explanations in natural language.**  
    ❌ **Never describe the next steps in text; return them in JSON format only.**  
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    Query_Executor = AssistantAgent(
        name="Query_Executor",
        system_message="""
You are the Query_Executor Agent. Your job is to evaluate the SQL query generated by the Sql_Generator agent before executing it, ensuring it is valid and optimized for the required result. After your evaluation, execute the SQL query using the provided function and pass the output to the Insight_Generator agent.
**Objective**: Ensure the SQL query is correct and efficient before passing it to the execution function.

**Steps**:
1. **Evaluate the SQL Query**:
- Review the query for syntax and logic errors.
- Ensure the query aligns with the original question and is capable of producing the necessary results.

2. **Execute the Query**:
- Use the `execute_query` function to run the validated query.

3. **Pass the Output**:
- Send the output of the query to the Insight_Generator agent for further processing.

    """,
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    Sql_tool = AssistantAgent(
        name="Sql_tool",
        system_message="""
You are the Sql_tool Agent. Your job is to take the SQL query given by the Query_Executor agent and use the function to extract the result.

**Steps**:
1. **Receive the SQL Query**:

2. **Execute the execute_query function and pass the output to Insight_Generator agent**
    """,
        max_consecutive_auto_reply=3,
        llm_config=None,
        human_input_mode="NEVER",
    )

    Sql_Execution_Critic = AssistantAgent(
        name="Sql_Execution_Critic",
        system_message=f"""
You are an expert critic for SQL query execution. 
Your role is to evaluate the result or errors coming from SQL query execution (with the Sql_tool agent) and provide detailed feedback to the Sql_Generator agent.
Your Objective:
- An expert SQL critic that evaluates the result or errors coming from SQL query execution (with the Sql_tool agent) and provide detailed feedback to the Sql_Generator agent in case of error.
- If the query is valid and executes correctly, pass it on to the Insight_Generator.

**Inputs**:

1. **Meta_Data**:
    {get_all_table_schemas()}

2. initial user question 

3. result or errors coming from SQL query execution (with the Sql_tool agent).

** Steps to follow**:

1. **Check the SQL Query Execution Results**:
- You will receive the result of an SQL query that was executed via the Sql_tool agent.
- If the query executed successfully, pass it along to the Insight_Generator for insights.
- If an error occurred, identify the error and provide feedback on what went wrong.

2. **Feedback to Sql_Generator**:
- Provide actionable feedback to help Sql_Generator modify the SQL query and address the issues.
- If the query is valid and executed successfully, pass the query to Insight_Generator to retrieve insights based on the result.

4. **Output Format**:

Output Format in strictly json format.

For Valid SQL queries that execute successfully:

{{
"initial_question": "Original question from the user",
"analysis_type": "Don't change the type, keep the same selected by routing agent",
"sql_question": "sql question",
"sql_query": "SQL query",
"db_name": "SQL",
"next_agent": "Insight_Generator",
"feedback": ""Happy with the result""
}}

For queries that encountered errors:
{{
"initial_question": "Original question from the user",
    "analysis_type": "Don't change the type, keep the same selected by routing agent",
    "sql_question": "sql question",
    "sql_query": "SQL query with errors",
    "next_agent": "Sql_Generator",
    "db_name": "SQL",
    "feedback": "Detailed explanation of the issue"
}}


⚠️ **Important**:  
    ✅ **You must always return the output in a strict JSON format as specified below.**  
    ❌ **Never return plain text or explanations in natural language.**  
    ❌ **Never describe the next steps in text; return them in JSON format only.** 
""",
        description="An expert SQL critic that evaluates the result or errors coming from SQL query execution (with the Sql_tool agent) and provide detailed feedback to the Sql_Generator agent in case of error otherwise pass the result to Insight_Generator ",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    Insight_Generator = AssistantAgent(
        name="Insight_Generator",
        system_message="""
1.1 Formulate a Direct Answer:

    Based on the query and the output from the DB tool, provide a precise and professional response.
    Ensure that the answer accurately represents the data retrieved by the query.
    Keep the response clear and structured, without unnecessary elaboration or assumptions beyond the query results.
    Always include numerical values if present in the query results.
    Present numerical summaries or counts to enhance clarity.

Example:
    Instead of: "The shipment statuses for the last 15 entries are as follows: In Transit, Delivered, Delivered, Delayed, Delayed, Delivered, Delayed, Delivered, Delivered, Delivered, Delivered, In Transit, In Transit, Delivered, In Transit."
    Provide: "Out of the last 15 shipments, the statuses are: Delivered (7), In Transit (4), and Delayed (4)."

1.2 Inference:

    Analyze the output logically and provide a brief, data-driven inference.
    Highlight key insights, patterns, or anomalies that can be drawn from the results.
    Maintain a neutral and professional tone, avoiding speculation beyond the given data.

Example:
    "The data indicates that 47% of recent shipments were successfully delivered, while 27% are still in transit and 27% faced delays. This suggests potential logistical challenges affecting timely deliveries."

2.1 Generate Python Code for Visualization:
    Select the Appropriate Chart Type:
    Choose the most suitable chart based on the query result.
    For categorical data: Use bar charts.
    For time-series or sequential data: Use line graphs.
    For correlations between two numerical variables: Use scatter plots.
    If the SQL query results cannot be visualized (e.g., no data or unsuited for visualization), return "None".

2.2 Chart Customization:
    Use Seaborn for better aesthetics and clarity.
    Different categories should have different colors based on insights.
    Display numerical values above each bar for clarity.
    Ensure charts are aligned and formatted for clarity.
    Avoid chart element overlaps (labels, bars, points).
    Use professional chart design principles (clean, readable, and visually appealing).

2.3 **Labeling & Titles**:
    **The X-axis should always represent categories.**
    **The Y-axis should always represent counts or numerical values.**
    Titles should clearly reflect the data being presented.
    Add legends when appropriate to aid in understanding multiple data sets.

Execution:
    The script should be error-free and ready to execute directly.
    Ensure charts effectively visualize the  query results.

3. Task: Validate the Answer for Query Question Based on Key Parameters
After generating the answer, validate it using the following parameters, scoring each from 0 to 10:

    Helpfulness: How useful is the answer in addressing the question? Does it offer practical, actionable advice?
    Relevance: How closely does the answer align with the specific question?
    Level of Detail: Is the answer sufficiently detailed to be informative?
    Groundedness: Is the answer based on factual and reliable information from the provided context?
    Completeness: Does the answer fully address the question?
    Faithfulness: Does the answer accurately reflect the provided information?

**STRICTLY FOLLOW THE OUTPUT JSON FORMAT GIVEN BELOW**:

**Example Output 1**:
{
  "initial_question": "Which city has the largest GDP?",
  "analysis_type": "Don't change the type, keep the same selected by routing agent",
  "db_name": "SQL", 
  "query_question": "sql question",
  "query": "SELECT city_name, GDP FROM city_stats ORDER BY GDP DESC LIMIT 1;",
  "query_answer": "The city with the largest GDP is Tokyo, with a GDP of approximately 1.5 trillion USD.",
  "Inference":"Detailed Inference",
  "scores": {
            "Helpfulness": X,
            "Relevance": Y,
            "Level of Detail": Z,
            "Groundedness": A,
            "Completeness": B,
            "Faithfulness": E
        },
  "feedback":"feedback explaintion",
  "python_code": "Return "None" only if no python code",
  "code_explaination":"Return "None" only if no python code"
}

**Example Output 2**:
{
  "initial_question": "What is the distribution of average temperatures in each city for the last 7 days?",
  "analysis_type": "Don't change the type, keep the same selected by routing agent",
  "db_name": "SQL", 
  "query_question": "What is the average temperature in each city for the last 7 days?",
  "query_query": "SELECT city_name, AVG(temperature) as avg_temperature FROM city_weather WHERE date >= NOW() - INTERVAL 7 DAY GROUP BY city_name;",
  "query_answer": "The average temperatures of the cities for the last 7 days are: Tokyo (15°C), New York (8°C), Los Angeles (20°C), London (10°C), and Paris (12°C).",
  "Inference":"Detailed Inference",
  "query_scores": {
            "Helpfulness": X,
            "Relevance": Y,
            "Level of Detail": Z,
            "Groundedness": A,
            "Completeness": B,
            "Faithfulness": E
        },
  "feedback":"feedback explaintion",
  "python_code": "import matplotlib.pyplot as plt\\nimport pandas as pd\\n\\ndata = {\\n    'City': ['Tokyo', 'New York', 'Los Angeles', 'London', 'Paris'],\\n    'Avg Temperature (°C)': [15, 8, 20, 10, 12]\\n}\\ndf = pd.DataFrame(data)\\n\\nplt.figure(figsize=(10, 6))\\nplt.barh(df['City'], df['Avg Temperature (°C)'], color='skyblue')\\nplt.title('Average Temperature for the Last 7 Days by City')\\nplt.xlabel('Average Temperature (°C)')\\nplt.ylabel('City')\\nplt.tight_layout()\\nplt.show()",
  "code_explaination":"Code Explaination"
  }
NOTE:
**if there is no feedback everything is correct and all the scores are more than or equal to 8 then give as "Happy with the result.".

⚠️ **Important**:  
    ✅ **You must always return the output in a strict JSON format as specified below.**  
    ❌ **Never return plain text or explanations in natural language.**  
    ❌ **Never describe the next steps in text; return them in JSON format only.**  
""",
        max_consecutive_auto_reply=6,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    Evaluation_Agent = AssistantAgent(
        name="evaluation_agent",
        system_message=f"""
You are the **Evaluation Agent** (SQL flow post-generation evaluator). You run **after Insight_Generator** and decide the next pipeline step.

**Pipeline role:**
1. Validate whether the **initial_question** is fully answered by the current **query_answer** (and inference).
2. Score the answer on all post-generation dimensions below.
3. Set `next_agent` based on what is still required (see routing rules below). **Never** send work directly to `Eryl_agent`.

{POST_GENERATION_EVALUATION_RUBRIC}

## Steps

### 1. Gather inputs
From Insight_Generator output: `initial_question`, `analysis_type`, `db_name`, `query_question`, `query`, `query_answer`, `Inference`.
For dependent / multi-turn flows, also use prior conversation messages: previous user question, prior `query_answer`, and any `updated_question`.

### 2. Score all dimensions
Apply the rubric to the **generated answer** (query_answer + inference), not only the SQL row dump.

### 3. Completeness & missing parts
- If **completeness < 8** OR any required part of `initial_question` is unanswered: set `evaluation_status` to `0`, populate `updated_question` with the missing part, and explain in `feedback` / `reasoning`.
- Missing **semantic** parts → set `updated_question` and `"next_agent": "query_transformer"` (routes to Eryl flow after transformation).
- Missing **structured** (SQL) parts → set `updated_question` and `"next_agent": "Sql_Generator"`.
- If fully complete and `status` is `"PASS"`: set `evaluation_status` to `1`, `updated_question` to `"None"`, `"next_agent": "user_proxy"`.

### 4. Next action (`next_agent` routing)
- **Complete** (`status` PASS, `evaluation_status` 1) → `"user_proxy"`
- **More SQL data needed** → `"Sql_Generator"`
- **Semantic / document context needed** → `"query_transformer"` (not `Eryl_agent`)

## Required JSON output (strict — no extra keys at root for scores)

```json
{{
  "initial_question": "<original user question>",
  "analysis_type": "<unchanged from routing>",
  "db_name": "SQL",
  "query_question": "<executed sub-question>",
  "query": "<SQL query>",
  "query_answer": "<answer from Insight_Generator>",
  "helpfulness": 9,
  "relevance": 9,
  "level_of_detail": 8,
  "groundedness": 10,
  "completeness": 8,
  "faithfulness": 9,
  "content_safety": 9,
  "output_safety": {{
    "pii_leak": false,
    "toxic_output": false,
    "score": 9
  }},
  "output_governance": {{
    "restricted_data_exposed": false,
    "score": 9
  }},
  "output_compliance": {{
    "policy_violation": false,
    "score": 8
  }},
  "composite_score": 8.7,
  "status": "PASS",
  "confidence": "HIGH",
  "evaluation_timestamp": "2026-05-28T14:30:00.000Z",
  "feedback": "<summary of evaluation outcome>",
  "reasoning": "<why complete or what is missing>",
  "updated_question": "None",
  "next_agent": "user_proxy",
  "evaluation_status": 1
}}
```

FAIL example — missing SQL (include `blocked_reason`; omit on PASS):
```json
{{
  ...
  "composite_score": 5.8,
  "status": "FAIL",
  "confidence": "LOW",
  "evaluation_timestamp": "2026-05-28T14:30:00.000Z",
  "blocked_reason": "Answer missing required structured data from SQL",
  "updated_question": "<missing SQL sub-question>",
  "next_agent": "Sql_Generator",
  "evaluation_status": 0
}}
```

FAIL example — missing semantic:
```json
{{
  ...
  "blocked_reason": "Answer missing required semantic portion of the question",
  "updated_question": "<refined semantic sub-question>",
  "next_agent": "query_transformer",
  "evaluation_status": 0
}}
```

- `evaluation_status`: `1` when the answer is complete and `status` is `"PASS"`; `0` otherwise.
- Compute `composite_score` and `status` using the rubric formula exactly.
- Use differentiated scores per calibration guidance; avoid all-10 profiles.

⚠️ Return **only** valid JSON. No plain-text explanations outside JSON.
 """,
        max_consecutive_auto_reply=6,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    @Sql_tool.register_for_execution()
    @Query_Executor.register_for_llm(description="You will execute the function and get the result")
    def db_execute_query(query: str = None, db_name: str = None):
        if db_name != "SQL":
            return "Invalid database name provided. Please use 'SQL'."
        try:
            df = pd.read_sql(query, get_engine())
            result_json = df.to_json(orient="records")
            return result_json
        except Exception as e:
            return f"An error occurred while execusting the query:{e}"

    return {
        "Sql_Generator": Sql_Generator,
        "Query_Executor": Query_Executor,
        "Sql_tool": Sql_tool,
        "Sql_Execution_Critic": Sql_Execution_Critic,
        "Insight_Generator": Insight_Generator,
        "evaluation_agent": Evaluation_Agent,
    }


# Name of the agent this flow is entered on (routing_agent hands off here).
ENTRY_AGENT = "Sql_Generator"

# Names that mean "control leaves Quin" when returned by route().
EXIT_AGENTS = {"user_proxy", "query_transformer"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    """
    Quin's internal next-speaker logic — mirrors the Quin-owned branches
    of the original unified `state_transition` function.

    `parsed_content` is the JSON-decoded content of the last speaker's
    message (see contract.json for each agent's output shape).

    Returns the *name* of the next agent. Some returned names
    ("user_proxy", "query_transformer") are outside Quin's own agent
    set — the parent orchestrator resolves those against its full
    registry (query_transformer/Eryl_agent live in the `eryl` package).
    Returns None when this speaker isn't one Quin handles.
    """
    if last_speaker_name == "Sql_Generator":
        return "Query_Executor"

    if last_speaker_name == "Query_Executor":
        return "Sql_tool"

    if last_speaker_name == "Sql_tool":
        return "Sql_Execution_Critic"

    if last_speaker_name == "Sql_Execution_Critic":
        next_agent = parsed_content.get("next_agent", "")
        if next_agent == "Insight_Generator":
            return "Insight_Generator"
        if next_agent in ("Sql_Generator", "SQL_Generator"):
            return "Sql_Generator"
        return None

    if last_speaker_name == "Insight_Generator":
        return "evaluation_agent"

    if last_speaker_name == "evaluation_agent":
        next_agent = parsed_content.get("next_agent", "")
        if next_agent == "user_proxy":
            return "user_proxy"
        if next_agent == "Sql_Generator":
            return "Sql_Generator"
        if next_agent == "query_transformer":
            return "query_transformer"
        if parsed_content.get("evaluation_status") == 1 and parsed_content.get("status") == "PASS":
            return "user_proxy"
        updated_question = parsed_content.get("updated_question", "None")
        if updated_question and str(updated_question).strip().lower() not in ("none", ""):
            return "query_transformer"
        return "user_proxy"

    return None


from .chain_helpers import build_initial_message as _build_chain_initial_message
from .chain_helpers import drive_chain as _drive_chain


class QuinChainRunner:
    """Single entrypoint wrapper around the Quin AutoGen SQL agent chain."""

    def run(
        self,
        initial_question: str,
        analysis_type: str | None = None,
        updated_question: str | None = None,
    ) -> dict:
        """
        Runs the chain starting at ENTRY_AGENT via build_agents()/route(),
        using the existing flow_input_contract / flow_output_contract fields
        already defined in quin/contract.json.
        Returns a dict matching output_schema.fields in contract.json:
        query, query_answer, analysis_type, query_results, inference, python_code.
        """
        agents = build_agents()
        initial_message = _build_chain_initial_message(
            initial_question,
            analysis_type,
            updated_question,
        )
        return _drive_chain(
            agents=agents,
            entry_agent=ENTRY_AGENT,
            exit_agents=EXIT_AGENTS,
            route_fn=route,
            initial_message=initial_message,
        )
