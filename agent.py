from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from schemas import generate_compact_agent_schema
from databaseExe import execute_sql
from prompt import SQL_PROMPT, AMBIGUITY_PROMPT, SCHEMA_LINKER, SCOPE_GUARD_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

MAX_SQL_ATTEMPTS = 3


class GraphState(BaseModel):
    user: str

    # Scope Guard
    is_valid_db_query: bool = True
    scope_reason: str = ""

    # Schema Linker
    relevant_schema: str = ""

    # Ambiguity Agent
    is_ambiguous: bool = False
    reason: str = ""

    # Clarification
    clarification: str = ""

    # SQL Agent
    query: str = ""
    sql_attempts: int = 0
    last_db_error: str = ""

    # Execute Agent
    result: str = ""


class ScopeResult(BaseModel):
    is_valid_db_query: bool = Field(
        ...,
        description="TRUE if the request could be answered using the customers/products database. FALSE otherwise."
    )
    reason: str = Field(
        ...,
        description="Only if FALSE: a short, polite one-sentence explanation for the user."
    )


class AmbiguityResult(BaseModel):
    is_ambiguous: bool = Field(
        ...,
        description="If query is unclear, mark True. Otherwise mark False."
    )
    reason: str = Field(
        ...,
        description=(
            "Only generate this if query is unclear. "
            "Explain exactly what information is missing."
        )
    )


class SQLResult(BaseModel):
    query: str = Field(
        ...,
        description="Only executable PostgreSQL SQL. Do not include markdown fences or explanations."
    )


def safe_llm_call(invoke_fn, *args, **kwargs):
    """
    Wraps an LLM call and converts a quota/rate-limit error into a clean,
    catchable exception instead of the raw Google API error text.
    """
    try:
        return invoke_fn(*args, **kwargs)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            raise RuntimeError(
                "Gemini free-tier daily quota exceeded for this model. "
                "Wait for the daily reset, use a different API key, or "
                "enable billing on your Google AI Studio project."
            ) from e
        raise


structured_scope_llm = llm.with_structured_output(ScopeResult)
structured_ambiguity_llm = llm.with_structured_output(AmbiguityResult)
structured_sql_llm = llm.with_structured_output(SQLResult)


def ScopeGuard(state: GraphState):
    messages = [
        SystemMessage(content=SCOPE_GUARD_PROMPT),
        HumanMessage(content=state.user),
    ]

    answer = safe_llm_call(structured_scope_llm.invoke, messages)

    return {
        "is_valid_db_query": answer.is_valid_db_query,
        "scope_reason": answer.reason if not answer.is_valid_db_query else "",
    }


def check_scope(state: GraphState):
    return "schemaLinker" if state.is_valid_db_query else "rejected"


def RejectedAgent(state: GraphState):
    return {"result": state.scope_reason or "This request isn't something I can answer from the database."}


def SchemaLinker(state: GraphState):
#Directly calling the fucntion saving the api cost
    return {"relevant_schema": generate_compact_agent_schema()}


def AmbiguityAgent(state: GraphState):
    messages = [
        SystemMessage(
            content=f"{AMBIGUITY_PROMPT}\n\nRelevant schema:\n{state.relevant_schema}"
        ),
        HumanMessage(
            content=(
                f"Original user request:\n{state.user}\n\n"
                f"Clarification provided by user:\n{state.clarification}"
            )
        ),
    ]

    answer = safe_llm_call(structured_ambiguity_llm.invoke, messages)

    return {"is_ambiguous": answer.is_ambiguous, "reason": answer.reason}


def check_ambiguity(state: GraphState):
    if state.is_ambiguous:
        return "clarification"
    return "sqlQuery"


def ClarificationAgent(state: GraphState):
    user_reply = interrupt({"question": state.reason})
    return {"clarification": user_reply}


def SQLAgent(state: GraphState):
    error_context = ""
    if state.last_db_error:
        error_context = (
            f"\n\nIMPORTANT: your previous query failed with this exact "
            f"Postgres error:\n{state.last_db_error}\n"
            f"The error usually tells you the correct column/table name. "
            f"Fix the query accordingly."
        )
        prior_query_context = f"\n\nYour previous (failed) query was:\n{state.query}"
    else:
        prior_query_context = ""

    messages = [
        SystemMessage(
            content=(
                f"{SQL_PROMPT}\n\n"
                f"Relevant schema:\n{state.relevant_schema}\n\n"
                "IMPORTANT:\n"
                "Return only executable PostgreSQL SQL.\n"
                "Do not return markdown code fences.\n"
                "Do not return explanations.\n"
                "Do not return ```sql.\n"
                "The SQL must be read-only."
                f"{error_context}"
            )
        ),
        HumanMessage(
            content=(
                f"Original user request:\n{state.user}\n\n"
                f"Clarification:\n{state.clarification}"
                f"{prior_query_context}"
            )
        ),
    ]

    answer = safe_llm_call(structured_sql_llm.invoke, messages)

    query = answer.query.strip()
    query = re.sub(r"^```sql\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^```\s*", "", query)
    query = re.sub(r"\s*```$", "", query)

    return {"query": query.strip()}


def validate_sql(query: str):
    cleaned_query = query.strip().rstrip(";").strip()

    if not cleaned_query:
        raise ValueError("Generated SQL query is empty.")

    if not re.match(r"^(SELECT|WITH)\b", cleaned_query, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")

    forbidden_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "TRUNCATE", "CREATE", "GRANT", "REVOKE",
    ]
    for keyword in forbidden_keywords:
        if re.search(rf"\b{keyword}\b", cleaned_query, re.IGNORECASE):
            raise ValueError(f"Unsafe SQL detected: {keyword}")

    return cleaned_query


def ExecuteAgent(state: GraphState):
    try:
        safe_query = validate_sql(state.query)
        sql_result = execute_sql(safe_query)
    except Exception as e:
        return {
            "result": f"SQL execution failed: {str(e)}",
            "last_db_error": str(e),
            "sql_attempts": state.sql_attempts + 1,
        }

    messages = [
        SystemMessage(
            content=(
                "You are a database result formatter.\n"
                "The data below came directly from the database.\n"
                "Format it clearly for the user.\n"
                "Do not invent, calculate, assume, or add information "
                "that is not present in the database result.\n"
                "If the data is empty, say plainly that no results were found "
                "— do not guess why.\n"
                "Use only the provided data."
                "Always show money in ₹ not in $ "
            )
        ),
        HumanMessage(content=f"DATA FROM DATABASE:\n{sql_result}"),
    ]

    answer = safe_llm_call(llm.invoke, messages)

    return {"result": answer.content, "last_db_error": ""}


def check_execution(state: GraphState):
    if not state.last_db_error:
        return "done"
    if state.sql_attempts >= MAX_SQL_ATTEMPTS:
        return "done"
    return "retry"


graph = StateGraph(GraphState)

graph.add_node("scopeGuard", ScopeGuard)
graph.add_node("rejected", RejectedAgent)
graph.add_node("schemaLinker", SchemaLinker)
graph.add_node("ambiguityAgent", AmbiguityAgent)
graph.add_node("clarification", ClarificationAgent)
graph.add_node("sqlQuery", SQLAgent)
graph.add_node("executeAgent", ExecuteAgent)

graph.add_edge(START, "scopeGuard")

graph.add_conditional_edges(
    "scopeGuard",
    check_scope,
    {"schemaLinker": "schemaLinker", "rejected": "rejected"},
)

graph.add_edge("rejected", END)
graph.add_edge("schemaLinker", "ambiguityAgent")

graph.add_conditional_edges(
    "ambiguityAgent",
    check_ambiguity,
    {"clarification": "clarification", "sqlQuery": "sqlQuery"},
)

graph.add_edge("clarification", "ambiguityAgent")
graph.add_edge("sqlQuery", "executeAgent")

graph.add_conditional_edges(
    "executeAgent",
    check_execution,
    {"retry": "sqlQuery", "done": END},
)

workflow = graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":

    config = {"configurable": {"thread_id": "test-session-1"}}

    result = workflow.invoke(
        {"user": "Show me the best customers."},
        config
    )

    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        user_reply = input(f"\nClarification needed: {question}\nYour answer: ")
        result = workflow.invoke(Command(resume=user_reply), config)

    print("\nGenerated SQL:")
    print(result["query"])

    print("\nResult:")
    print(result["result"])