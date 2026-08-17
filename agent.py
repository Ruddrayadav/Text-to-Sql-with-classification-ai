from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from schemas import generate_compact_agent_schema
from databaseExe import execute_sql
from prompt import SQL_PROMPT, AMBIGUITY_PROMPT, SCHEMA_LINKER
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)


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


# CHANGE: SQL output is now structured instead of allowing arbitrary LLM text.
class SQLResult(BaseModel):
    query: str = Field(
        ...,
        description="Only executable PostgreSQL SQL. Do not include markdown fences or explanations."
    )


structured_ambiguity_llm = llm.with_structured_output(AmbiguityResult)

# CHANGE: SQL generation is now structured to prevent ```sql ... ``` from reaching PostgreSQL.
structured_sql_llm = llm.with_structured_output(SQLResult)


def SchemaLinker(state: GraphState):
    schemas_data = generate_compact_agent_schema()

    messages = [
        SystemMessage(
            content=f"{SCHEMA_LINKER}\n\nSchema:\n{schemas_data}"
        ),
        HumanMessage(content=state.user),
    ]

    answer = llm.invoke(messages)

    return {
        "relevant_schema": answer.content
    }


def AmbiguityAgent(state: GraphState):
    messages = [
        SystemMessage(
            content=(
                f"{AMBIGUITY_PROMPT}\n\n"
                f"Relevant schema:\n{state.relevant_schema}"
            )
        ),
        HumanMessage(
            content=(
                f"Original user request:\n{state.user}\n\n"
                f"Clarification provided by user:\n{state.clarification}"
            )
        ),
    ]

    answer = structured_ambiguity_llm.invoke(messages)

    return {
        "is_ambiguous": answer.is_ambiguous,
        "reason": answer.reason
    }


def check_ambiguity(state: GraphState):
    if state.is_ambiguous:
        return "clarification"

    return "sqlQuery"


def ClarificationAgent(state: GraphState):
    user_reply = interrupt(
        {
            "question": state.reason
        }
    )

    # CHANGE: Do not modify state.user. Store the clarification separately.
    return {
        "clarification": user_reply
    }


def SQLAgent(state: GraphState):
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
            )
        ),
        HumanMessage(
            content=(
                f"Original user request:\n{state.user}\n\n"
                f"Clarification:\n{state.clarification}"
            )
        ),
    ]

    answer = structured_sql_llm.invoke(messages)

    query = answer.query.strip()

    query = re.sub(r"^```sql\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^```\s*", "", query)
    query = re.sub(r"\s*```$", "", query)

    return {
        "query": query.strip()
    }


# CHANGE: Added SQL safety validation before sending anything to PostgreSQL.
def validate_sql(query: str):
    cleaned_query = query.strip().rstrip(";").strip()

    if not cleaned_query:
        raise ValueError("Generated SQL query is empty.")

    # CHANGE: Only allow read-only SQL.
    if not re.match(r"^(SELECT|WITH)\b", cleaned_query, re.IGNORECASE):
        raise ValueError(
            "Only SELECT or WITH queries are allowed."
        )

    # CHANGE: Block destructive/write operations even if they appear later in the query.
    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    ]

    for keyword in forbidden_keywords:
        if re.search(
            rf"\b{keyword}\b",
            cleaned_query,
            re.IGNORECASE
        ):
            raise ValueError(
                f"Unsafe SQL detected: {keyword}"
            )

    return cleaned_query


def ExecuteAgent(state: GraphState):
    try:

        safe_query = validate_sql(state.query)

        sql_result = execute_sql(safe_query)

        messages = [
            SystemMessage(
                content=(
                    "You are a database result formatter.\n"
                    "The data below came directly from the database.\n"
                    "Format it clearly for the user.\n"
                    "Do not invent, calculate, assume, or add information "
                    "that is not present in the database result.\n"
                    "Use only the provided data."
                )
            ),
            HumanMessage(
                content=f"DATA FROM DATABASE:\n{sql_result}"
            ),
        ]

        answers = llm.invoke(messages)

        return {
            "result": answers.content
        }

    except Exception as e:
        # CHANGE: Store execution errors instead of silently crashing the graph.
        return {
            "result": f"SQL execution failed: {str(e)}"
        }


graph = StateGraph(GraphState)

graph.add_node("schemaLinker", SchemaLinker)
graph.add_node("ambiguityAgent", AmbiguityAgent)
graph.add_node("clarification", ClarificationAgent)
graph.add_node("sqlQuery", SQLAgent)
graph.add_node("executeAgent", ExecuteAgent)


graph.add_edge(
    START,
    "schemaLinker"
)

graph.add_edge(
    "schemaLinker",
    "ambiguityAgent"
)


graph.add_conditional_edges(
    "ambiguityAgent",
    check_ambiguity,
    {
        "clarification": "clarification",
        "sqlQuery": "sqlQuery",
    },
)


graph.add_edge(
    "clarification",
    "ambiguityAgent"
)

graph.add_edge(
    "sqlQuery",
    "executeAgent"
)

graph.add_edge(
    "executeAgent",
    END
)


workflow = graph.compile(
    checkpointer=MemorySaver()
)


if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "test-session-1"
        }
    }

    result = workflow.invoke(
        {
            "user": "Show me the best customers.c"
        },
        config
    )

    while "__interrupt__" in result:

        question = result["__interrupt__"][0].value["question"]

        user_reply = input(
            f"\nClarification needed: {question}\n"
            f"Your answer: "
        )

        result = workflow.invoke(
            Command(resume=user_reply),
            config
        )

    print("\nGenerated SQL:")
    print(result["query"])

    print("\nResult:")
    print(result["result"])