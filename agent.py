from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from schemas import generate_compact_agent_schema
from databaseExe import execute_sql
from prompt import SQL_PROMPT, AMBIGUITY_PROMPT
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


structured_ambiguity_llm = llm.with_structured_output(AmbiguityResult)
structured_sql_llm = llm.with_structured_output(SQLResult)


def SchemaLinker(state: GraphState):
    myschemaa = generate_compact_agent_schema()
    # No LLM call here — your schema is small enough (6 tables) that passing
    # it straight through is simpler and free, and it removes one of the
    # guaranteed API calls that were eating into your daily quota.
    return {"relevant_schema": myschemaa}


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
    # If we're here after a failed execution, include the exact DB error
    # so the model can self-correct instead of repeating the same mistake.
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


def format_rows_for_display(rows) -> str:
    """
    Plain-Python formatter — no LLM call. Turns query results into a
    readable markdown list/table. This was previously an LLM call, which
    burned quota for zero real benefit: formatting known-shape DB rows
    doesn't need a language model.
    """
    if not rows:
        return "No results found."

    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

    return str(rows)


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

    # Formatting is separate from execution now — if this fails, the query
    # itself still succeeded, so we don't want to blame the DB or trigger a retry.
    formatted = format_rows_for_display(sql_result)
    return {"result": formatted, "last_db_error": ""}


def check_execution(state: GraphState):
    # No error -> we're done. Error but out of retries -> also done
    # (the "result" field already holds the failure message for the user).
    # Error and retries remain -> loop back to SQLAgent with the error context.
    if not state.last_db_error:
        return "done"
    if state.sql_attempts >= MAX_SQL_ATTEMPTS:
        return "done"
    return "retry"


graph = StateGraph(GraphState)

graph.add_node("schemaLinker", SchemaLinker)
graph.add_node("ambiguityAgent", AmbiguityAgent)
graph.add_node("clarification", ClarificationAgent)
graph.add_node("sqlQuery", SQLAgent)
graph.add_node("executeAgent", ExecuteAgent)

graph.add_edge(START, "schemaLinker")
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