"""
Eryl Agent — standalone semantic / document search agent.

Flow:
    user_proxy -> Eryl_agent (forwards question) -> retriever (vector search)
    -> llm_answer_maker (answers from retrieved context) -> critic_agent
    (scores the answer; on FAIL sends a refined query back to Eryl_agent,
    on PASS returns control to user_proxy)

Everything unrelated to Eryl's own retrieve -> answer -> critique loop
(routing agent, Responsible AI gate, SQL flow, cross-flow evaluation,
SQL-driven "query_transformer" step, logging to a DB, etc.) has been
removed. This file is meant to run end to end on its own.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.oai.client import OpenAIWrapper

load_dotenv()

# ---------------------------------------------------------------------------
# AutoGen strips dots from Azure deployment names (gpt-5.4 -> gpt-54).
# Preserve the deployment name exactly as configured in Azure.
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

# ---------------------------------------------------------------------------
# Configuration — fill these in via environment variables / .env
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME")

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "dupont_email_demo")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "my-semantic-config")

llm_config = {
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

embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_BASE,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)


def generate_embeddings(text: str) -> list:
    return embedding_client.embeddings.create(
        input=[text], model=EMBEDDING_MODEL_DEPLOYMENT_NAME
    ).data[0].embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_json(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _extract_user_question(message: str) -> str:
    text = message.strip()
    if text.startswith('"question":'):
        return text.split('"question":', 1)[1].strip().strip('"').strip()
    try:
        payload = json.loads(_clean_json(text))
        if isinstance(payload, dict):
            return payload.get("question") or text
    except json.JSONDecodeError:
        pass
    return text


def _critic_passed(last_message: str) -> bool:
    return "Happy with the answer" in last_message


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You pass the user's question to Eryl_agent and receive the final answer.",
    code_execution_config=False,
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: msg["content"],
)

Eryl_agent = AssistantAgent(
    name="Eryl_agent",
    system_message="""
You are the Eryl selector agent. Your job is to forward the current question
(or, if you are being re-invoked after a failed critique, the critic's
`feedback_query`) to the retriever agent so it can fetch relevant context.

Output strictly in JSON:
{
  "question": "the question to search for",
  "next_agent": "retriever"
}

Never return plain text outside JSON.
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

retriever = AssistantAgent(
    name="retriever",
    system_message="Your role is to execute the extract_context function and pass the returned context to llm_answer_maker.",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

llm_answer_maker = AssistantAgent(
    name="llm_answer_maker",
    system_message="""
You are the Answer Maker Agent. Generate a detailed answer to the question
using ONLY the provided retrieved context. Do not invent information.

- Include any relevant numbers or facts present in the context.
- If the context is insufficient, respond exactly with:
  "I do not have enough information to answer this question."
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=4,
    description="Generates the final answer from retrieved context.",
)

critic_agent = AssistantAgent(
    name="critic_agent",
    system_message="""
You are the Critic Agent. Evaluate `llm_answer` against the question and the
context that was used to produce it.

Score these 0-10: helpfulness, relevance, groundedness, completeness, faithfulness.
composite_score = average of the five scores.
status = "PASS" if composite_score >= 7 else "FAIL".

Output strictly in JSON:
{
  "question": "<question>",
  "llm_answer": "<answer being evaluated>",
  "helpfulness": 9,
  "relevance": 9,
  "groundedness": 9,
  "completeness": 8,
  "faithfulness": 9,
  "composite_score": 8.8,
  "status": "PASS",
  "feedback_query": "None"
}

If status is "FAIL", set `feedback_query` to a focused follow-up question that
would retrieve the missing information, and explain the gap in `feedback_detail`.
If status is "PASS", set `feedback_query` to "None" and, after the JSON block,
append exactly the line: Happy with the answer
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=5,
    description="Scores the semantic answer and requests another retrieval round on failure.",
    is_termination_msg=lambda msg: _critic_passed(msg.get("content", "")),
)


@retriever.register_for_execution()
@Eryl_agent.register_for_llm(description="Retrieve relevant context via vector search.")
async def extract_context(question: str = None) -> dict:
    if not question:
        return {}

    vector_query = VectorizedQuery(
        vector=generate_embeddings(question),
        k_nearest_neighbors=3,
        fields="embedding",
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["documentId", "content"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=AZURE_SEARCH_SEMANTIC_CONFIG,
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=3,
    )
    return {"vector_context": [result["content"] for result in results]}


# ---------------------------------------------------------------------------
# Flow control
# ---------------------------------------------------------------------------
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1]["content"]

    if last_speaker is user_proxy:
        return Eryl_agent

    elif last_speaker is Eryl_agent:
        return retriever

    elif last_speaker is retriever:
        return llm_answer_maker

    elif last_speaker is llm_answer_maker:
        return critic_agent

    elif last_speaker is critic_agent:
        if _critic_passed(last_message):
            return user_proxy
        return Eryl_agent


groupchat = GroupChat(
    agents=[user_proxy, Eryl_agent, retriever, llm_answer_maker, critic_agent],
    messages=[],
    max_round=30,
    speaker_selection_method=state_transition,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def get_answer(question: str) -> dict:
    """Run the Eryl pipeline end to end and return the final answer."""
    chat_history = user_proxy.initiate_chat(
        manager,
        message=json.dumps({"question": question}),
        summary_method="reflection_with_llm",
    )

    llm_answer = ""
    for item in chat_history.chat_history:
        if item.get("name") == "llm_answer_maker":
            llm_answer = item["content"]

    return {"question": question, "llm_answer": llm_answer}


if __name__ == "__main__":
    # Standalone smoke run. For reuse/codegen, prefer ErylChainRunner.run
    # (agent_library.eryl.chain_helpers) which wraps get_answer with the
    # orchestrator-friendly input/output contract.
    q = "What specific global factors cloud the 2025 foreign exchange outlook? "
    result = get_answer(q)
    print(json.dumps(result, indent=2))
