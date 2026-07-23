"""
Eryl — semantic / vector-retrieval agent.

Split out of the unified AutoGen pipeline (main.py). Eryl owns the
semantic analysis path end-to-end:

    query_transformer -> Eryl_agent (selector) -> retriever
        -> llm_answer_maker -> critic_agent

This module is self-contained but not dependency-free: Azure OpenAI / Search
credentials and vector metadata come from ``.runtime_config`` (env vars —
no ambient project-root imports). It does not import anything from the
`quin` package, and `quin` does not import anything from here — cross-flow
handoffs are expressed as plain agent-name strings that a parent
orchestrator resolves against its full agent registry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from autogen import AssistantAgent
from autogen.oai.client import OpenAIWrapper
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from .runtime_config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4_LLM_MODEL_DEPLOYMENT_NAME,
    get_azure_openai_client,
    get_embedding_model,
    get_localcontext_builder,
    get_search_client,
    get_vector_data,
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


# Index / search config (unchanged from the unified main.py).
INDEX_NAME = "dupont_email_demo"

# Lazily constructed so importing the package does not require live Azure clients.
_client = None
search_client = None


def _openai_client():
    global _client
    if _client is None:
        _client = get_azure_openai_client()
    return _client


def _get_search_client():
    global search_client
    if search_client is None:
        search_client = get_search_client(index_name=INDEX_NAME)
    return search_client


def __getattr__(name: str):
    """Lazy module attrs for back-compat exports (Eryl_Meta_data / Vector_Data)."""
    if name in ("Eryl_Meta_data", "Vector_Data"):
        return get_vector_data()
    if name == "Embedding_Model":
        return get_embedding_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_embeddings(text: str, client, embedding_model_deployment_name):
    embeddings = client.embeddings.create(input=[text], model=embedding_model_deployment_name).data[0].embedding
    return embeddings


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


def parsed_last_message(last_message: str) -> dict:
    return json.loads(last_message.replace("```json", "").replace("```", "").strip())


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


def _critic_evaluation_passed(last_message: str) -> bool:
    if "Happy with the answer" in last_message:
        return True
    try:
        cleaned = last_message.replace("Happy with the answer", "").strip()
        parsed = parsed_last_message(cleaned)
        _, status = compute_post_generation_composite(parsed)
        return status == "PASS"
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def build_agents(llm_config: Optional[dict] = None) -> dict:
    """
    Factory that builds a fresh set of Eryl's semantic-flow AssistantAgents
    and registers Eryl's one tool function (`extract_context`) on them.

    Returns a dict keyed by agent name, matching the names used in the
    original unified `state_transition` / `GroupChat`, so a parent
    orchestrator can drop these straight into its own agent registry:

        {
          "query_transformer": ...,
          "Eryl_agent": ...,
          "retriever": ...,
          "llm_answer_maker": ...,
          "critic_agent": ...,
        }

    A factory (rather than module-level singletons) is used because
    AssistantAgent instances are stateful (auto-reply counters, chat
    history) and Eryl may need to be instantiated more than once
    (e.g. once per conversation/session).
    """
    cfg = llm_config or LLM_CONFIG

    query_transformer = AssistantAgent(
        name="query_transformer",
        system_message="""
You are the Query Transformer Agent. Your task is to take the user's original question and transform it into a more specific question by focusing directly on the SQL answer. Use the SQL answer to create a precise question for further vector search, keeping the intent of the original question but refining it based on the SQL answer.
 
**Steps to Follow**:
 
1. **Analyze the SQL Answer**:
   - Carefully examine the SQL answer to identify its key subject or insight, which will serve as the basis for the updated question.
   - Focus on the main entity or fact provided in the SQL answer, aligning the refined question with this focus.
 
2. **Generate the Updated Question Based on SQL Answer**:
   - Formulate the `updated_question` so that it directly inquires about the SQL answer's main subject.
   - Keep the updated question concise and focused on the SQL answer without rephrasing or adding unrelated details from the original question.
   - The goal is to enable a more focused and precise vector search based on the SQL answer's insight.
 
3. **Example**:
   **Initial Question**: "Which country has the highest GDP and what steps have they taken to improve it?"
   **SQL Answer**: "The country with the highest GDP is the United States, with a GDP of approximately 21 trillion USD."
   **Updated Question**: "What steps has the United States taken to improve its GDP?"
 
**Output JSON Format**:
- "initial_question": The user's original question.
- "analysis_type": Maintain the type selected by the routing agent (do not modify).
- "sql_query": The SQL query executed.
- "sql_answer": The SQL answer obtained.
- "updated_question": The refined question focused on the SQL answer’s primary insight.
 
{
  "initial_question": "initial_question",
  "analysis_type": "analysis_type",
  "sql_query": "sql_query",
  "sql_answer": "sql_answer",
  "updated_question": "updated_question"
  "next_agent": "Eryl_agent",
 
}
⚠️ **Important**:  
    ✅ **You must always return the output in a strict JSON format as specified below.**  
    ❌ **Never return plain text or explanations in natural language.**  
    ❌ **Never describe the next steps in text; return them in JSON format only.** 
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    Eryl_agent = AssistantAgent(
        name="Eryl_agent",
        system_message="""
You are the Selector Agent. Your job is to forward the `updated_question` to the Retriever Agent to fetch the required context.
 
**Instructions**:
   
1. **Handle Feedback Queries from Critic Agent**:
   - If you receive a feedback query from the Critic Agent, replace the `updated_question` with the feedback query. Then, forward it to the Retriever Agent to fetch additional context based on the updated query.
 
2. **Output Format**:
   - Ensure the output follows the format below, including all relevant details.
   - Use the exact analysis type selected by the routing agent without changes.
 
**Output Format in strictly json format.**:
{
  "initial_question": "initial_question",
  "analysis_type": "analysis_type",
  "sql_query": "sql_query",
  "sql_answer": "sql_answer",
  "updated_question": "updated_question",
  "next_agent": "retriever"
}
 
Example:
If you receive an initial question "What factors contribute to economic growth?", and `updated_question` is "What steps has the United States taken to improve its GDP?" with no specific selector type provided, output should be:
{
  "initial_question": "What factors contribute to economic growth?",
  "analysis_type": "SQL",
  "sql_query": "SELECT * FROM economic_growth_factors;",
  "sql_answer": "Key factors include GDP growth, innovation, and trade policy.",
  "updated_question": "What steps has the United States taken to improve its GDP?",
  "next_agent": "retriever"
 
}
pass it question or the updated_question to retriever agent to fetch the context..always execute the function to get the chunks.
⚠️ **Important**:  
    ✅ **You must always return the output in a strict JSON format as specified below.**  
    ❌ **Never return plain text or explanations in natural language.**  
    ❌ **Never describe the next steps in text; return them in JSON format only.** 
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    retriever = AssistantAgent(
        name="retriever",
        system_message="""Your role is to execute the extract_context function and pass the context to llm_answer maker.
    """,
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    llm_answer_maker = AssistantAgent(
        name="llm_answer_maker",
        system_message="""You are the Answer Maker Agent. Your role is to generate the detailed answer for a given question using only the provided context from the vector store (Zilliz-retrieved chunks).
    Do not create answers independently without context, and avoid guessing.
    If the context is insufficient, clearly indicate that you don't have enough information to answer the question.
 
 
**Task : Form the Answer Based on Provided Context**
1. When you receive a question and its related context, generate the answer based solely on that retrieved context.
2. Ensure that the answer is detailed, mentioning any numerical values and factual information without error.
3. If you cannot answer the question because the context is insufficient, simply state in answer: "I do not have enough information to answer this question."
4. Do not generate or assume any information outside of the provided context.
 
    """,
        llm_config=cfg,
        max_consecutive_auto_reply=4,
        description="This agent will frame the detailed answer.",
    )

    critic_agent = AssistantAgent(
        name="critic_agent",
        system_message=f"""
You are the **Critic Agent** (Eryl flow post-generation evaluator). You run **after llm_answer_maker** and validate the semantic answer against the question and retrieved context.

**Pipeline role (unchanged):**
1. Evaluate the `llm_answer` against `question` / `Updated_question` and the context used.
2. If `status` is `"FAIL"` or material gaps remain, set `feedback_query` for Eryl_agent to retrieve more context.
3. If `status` is `"PASS"`, set `feedback_query` to `"None"` and append `Happy with the answer` after the JSON.

{POST_GENERATION_EVALUATION_RUBRIC}

## Dependent / multi-turn
When `analysis_type` is dependent or prior turns exist: use **initial_question**, **previous user/updated questions**, prior answers in the thread, and the **current llm_answer**. Check cross-turn consistency for relevance, groundedness, completeness, and business-rule adherence.

## Pass / fail for this agent
- `status` = `"PASS"` when `composite_score` >= 7 (per rubric).
- `status` = `"FAIL"` when `composite_score` < 7 OR any core dimension < 7 OR `output_safety` / `output_governance` / `output_compliance` flags are true.

## Feedback query
- On **FAIL**: provide a focused `feedback_query` (specific missing facts) for Eryl_agent; explain gaps in `feedback_detail`.
- On **PASS**: `feedback_query` = `"None"`, `feedback_detail` = `"None"`, then append exactly: `Happy with the answer`

## Required JSON output

```json
{{
  "question": "<initial_question>",
  "Updated_question": "<current sub-question if any>",
  "llm_answer": "<answer from llm_answer_maker>",
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
  "feedback_query": "None",
  "feedback_detail": "None"
}}
```

FAIL example (include `blocked_reason`; omit on PASS):
```json
{{
  ...
  "status": "FAIL",
  "confidence": "LOW",
  "evaluation_timestamp": "2026-05-28T14:30:00.000Z",
  "blocked_reason": "Answer not grounded in retrieved context",
  "feedback_query": "<focused follow-up>",
  "feedback_detail": "<gap explanation>"
}}
```

Compute `composite_score` and `status` using the rubric formula exactly. Use differentiated scores; avoid all-10 profiles.

⚠️ Return valid JSON first, then `Happy with the answer` only on PASS. No plain-text outside JSON except that exact phrase on success.
    """,
        llm_config=cfg,
        max_consecutive_auto_reply=5,
        description="Post-generation evaluator for the Eryl semantic flow; forms feedback queries when evaluation FAILs.",
        is_termination_msg=lambda msg: "Happy with the answer" in msg["content"] or _critic_evaluation_passed(msg.get("content", "")),
    )

    @retriever.register_for_execution()
    @Eryl_agent.register_for_llm(description="Retrieve relevant context by executing the extract_context function.")
    async def extract_context(question: str = None, vector_weight: float = 0.5, graph_weight: float = 0.5) -> dict:
        selector = "vector"
        if not question:
            print("No question provided")
            return {}

        async def fetch_vector_context():
            vector_query = VectorizedQuery(
                vector=generate_embeddings(question, _openai_client(), get_embedding_model()),
                k_nearest_neighbors=3,
                fields="embedding",
            )
            results = _get_search_client().search(
                search_text=None,
                vector_queries=[vector_query],
                select=["documentId", "content"],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="my-semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=3,
            )
            return [result["content"] for result in results]

        async def fetch_graph_context():
            Localcontext_builder = get_localcontext_builder()
            context = Localcontext_builder.build_context(question, top_k_mapped_entities=2, top_k_relationships=2)
            return context

        if selector == "vector":
            vector_context = await fetch_vector_context()
            return {"vector_context": vector_context}

        elif selector == "graph":
            graph_context = await fetch_graph_context()
            return {"graph_context": str(graph_context)}

        elif selector == "hybrid":
            vector_context, graph_context = await asyncio.gather(fetch_vector_context(), fetch_graph_context())
            combined_context = {
                "combined_context": f"VectorDB context: [{vector_context}, with weight: {vector_weight}], "
                                    f"GraphDB context: [{graph_context}, with weight: {graph_weight}]",
                "question": question,
            }
            return combined_context

        else:
            print("Invalid selector provided")
            return {}

    return {
        "query_transformer": query_transformer,
        "Eryl_agent": Eryl_agent,
        "retriever": retriever,
        "llm_answer_maker": llm_answer_maker,
        "critic_agent": critic_agent,
    }


# Name of the agent this flow is entered on (routing_agent, or Quin's
# evaluation_agent, hands off here).
ENTRY_AGENT = "query_transformer"

# Names that mean "control leaves Eryl" when returned by route().
EXIT_AGENTS = {"user_proxy"}


def route(last_speaker_name: str, last_message: str, parsed_content: dict) -> Optional[str]:
    """
    Eryl's internal next-speaker logic — mirrors the Eryl-owned branches
    of the original unified `state_transition` function.

    `last_message` is the raw (un-parsed) content of the last speaker's
    message — needed because critic_agent's pass/fail check looks for
    the literal "Happy with the answer" phrase, not just JSON fields.
    `parsed_content` is unused for critic_agent but kept for a
    consistent signature across route() in both packages.

    Returns the *name* of the next agent, or None when this speaker
    isn't one Eryl handles.
    """
    if last_speaker_name == "query_transformer":
        return "Eryl_agent"

    if last_speaker_name == "Eryl_agent":
        return "retriever"

    if last_speaker_name == "retriever":
        return "llm_answer_maker"

    if last_speaker_name == "llm_answer_maker":
        return "critic_agent"

    if last_speaker_name == "critic_agent":
        if _critic_evaluation_passed(last_message):
            return "user_proxy"
        return "Eryl_agent"

    return None


from .chain_helpers import build_initial_message as _build_chain_initial_message
from .chain_helpers import drive_chain as _drive_chain


class ErylChainRunner:
    """Single entrypoint wrapper around the Eryl AutoGen semantic RAG agent chain."""

    def run(
        self,
        initial_question: str,
        analysis_type: str | None = None,
        sql_query: str | None = None,
        sql_answer: str | None = None,
        updated_question: str | None = None,
    ) -> dict:
        """
        Runs the chain starting at ENTRY_AGENT ('query_transformer') via
        build_agents()/route(). Returns a dict matching output_schema.fields
        in eryl/contract.json: llm_answer, analysis_type, updated_question.
        """
        agents = build_agents()
        initial_message = _build_chain_initial_message(
            initial_question,
            analysis_type,
            sql_query,
            sql_answer,
            updated_question,
        )
        return _drive_chain(
            agents=agents,
            entry_agent=ENTRY_AGENT,
            exit_agents=EXIT_AGENTS,
            route_fn=route,
            initial_message=initial_message,
        )
