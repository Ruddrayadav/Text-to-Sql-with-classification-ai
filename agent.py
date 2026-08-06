import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from main import get_relevant_chunks


class ConflictingFact(BaseModel):
    claim: str = Field(description="The conflicting claim found in a source chunk.")
    source: str = Field(description="The file name or document source for the claim.")


class ConflictEvaluation(BaseModel):
    conflict: bool = Field(description="True when the retrieved chunks disagree.")
    conflicting_facts: list[ConflictingFact] = Field(
        default_factory=list,
        description="Concrete facts that conflict with each other.",
    )
    recommended_action: Literal[
        "answer_user",
        "ask_user_which_version_applies",
        "request_more_context",
    ] = Field(description="The safest next action.")
    final_answer: str | None = Field(
        default=None,
        description="Final answer when there is no conflict; null when clarification is needed.",
    )
    source: str | None = Field(
        default=None,
        description="Source filename used for the final answer when conflict is false.",
    )


class AgentState(TypedDict, total=False):
    query: str
    relevant_chunks: list[dict[str, Any]]
    evaluation: dict[str, Any]


class CompanyPolicyAgent:
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash-lite",
        llm: Any | None = None,
        temperature: float = 0,
        top_k: int = 8,
    ) -> None:
        self.top_k = top_k
        self.llm = llm or ChatGoogleGenerativeAI(model=model, temperature=temperature)
        try:
            self.structured_llm = self.llm.with_structured_output(ConflictEvaluation)
        except NotImplementedError:
            self.structured_llm = None
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a careful company-policy QA agent. Use only the retrieved chunks
            to answer — never use outside knowledge.

            Step 1 — Extract atomic facts: from the retrieved chunks, pull out
            individual facts relevant to the user's question. Each fact must cover
            exactly ONE attribute (e.g. "annual leave days = 18"), not a whole
            paragraph or multiple attributes bundled together.

            Step 2 — Compare: group facts that refer to the SAME attribute. Two
            facts only conflict if they give DIFFERENT VALUES for the SAME specific
            attribute (e.g. two different numbers for "annual leave days"). Facts on
            related but different attributes (e.g. "remote work days" vs "annual
            leave days") are NOT conflicts, even if they appear in the same
            paragraph or topic.

            Step 3 — Handle drafts separately: if a source is marked as "Draft" or
            "Proposal" and not yet approved, do not treat it as conflicting with an
            approved policy. Instead note it separately as a pending change.

            Step 4 — Decide:
            - If two or more APPROVED sources give different values for the same
              attribute → conflict=true, list each conflicting fact with its exact
              value and source, set recommended_action="ask_user_which_version_applies",
              final_answer=null.
            - If there is no conflict → answer using the most recent approved
              source, set recommended_action="answer_user", include the answer in
              final_answer, and put the exact source filename in source.
            - If the chunks don't contain enough information to answer →
              recommended_action="request_more_context", final_answer=null,
              source=null.

            Always cite the exact source filename for every fact you use.
            Return only valid JSON with these keys:
            conflict, conflicting_facts, recommended_action, final_answer, source.
                    """
                ),
                (
                    "human",
                    "User query:\n{query}\n\nRetrieved chunks:\n{chunks}",
                ),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("evaluate", self._evaluate)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_edge("evaluate", END)
        return graph.compile()

    def _retrieve(self, state: AgentState) -> AgentState:
        query = state["query"]
        return {
            "relevant_chunks": get_relevant_chunks(query, top_k=self.top_k),
        }

    def _format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant chunks were found."

        formatted_chunks = []
        for index, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                f"Chunk {index}\n"
                f"Source: {chunk.get('source', 'unknown')}\n"
                f"Text: {chunk.get('chunk', '')}"
            )
        return "\n\n".join(formatted_chunks)

    def _evaluate(self, state: AgentState) -> AgentState:
        chunks = state.get("relevant_chunks", [])
        messages = self.prompt.invoke(
            {
                "query": state["query"],
                "chunks": self._format_chunks(chunks),
            }
        )
        if self.structured_llm is not None:
            evaluation = self.structured_llm.invoke(messages)
            return {"evaluation": evaluation.model_dump()}

        response = self.llm.invoke(
            self._build_json_prompt(state["query"], self._format_chunks(chunks))
        )
        evaluation = self._parse_json_response(response)
        return {"evaluation": evaluation.model_dump()}

    def _build_json_prompt(self, query: str, chunks: str) -> str:
        return f"""
You are a company-policy conflict detection agent. Use only the retrieved chunks.

User query:
{query}

Retrieved chunks:
{chunks}

Rules:
- Compare only facts that answer the same specific attribute in the user query.
- If approved sources disagree, set conflict=true and list the conflicting facts.
- If a source is a draft/proposal, do not count it as an approved-source conflict.
- If there is no conflict and enough information exists, answer from the best source.
- If there is not enough information, use recommended_action="request_more_context".

Return only valid JSON in exactly this shape:
{{
  "conflict": true,
  "conflicting_facts": [
    {{"claim": "specific claim", "source": "source_filename.txt"}}
  ],
  "recommended_action": "answer_user",
  "final_answer": "answer string or null",
  "source": "source_filename.txt or null"
}}
""".strip()

    def _parse_json_response(self, response: Any) -> ConflictEvaluation:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        text = str(content).strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        return ConflictEvaluation.model_validate(json.loads(text))

    def run(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query cannot be empty.")

        result = self.graph.invoke({"query": query})
        return result["evaluation"]


if __name__ == "__main__":
    user_query = input("Enter the query: ")
    agent = CompanyPolicyAgent()
    print(agent.run(user_query))
