# Actionableitems, decison, questions    implemented in this


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm import get_llm

def build_chain(system_prompt: str):
    llm = get_llm(temperature=0.2)

    return (RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")]) | llm | StrOutputParser())


def _extract(transcript: str, instructions: str, empty_message: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError("Cannot extract meeting details from an empty transcript.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=300).split_text(transcript)
    chain = build_chain(instructions)
    outputs = [chain.invoke(chunk) for chunk in chunks]
    if len(outputs) == 1:
        return outputs[0]
    combine = build_chain(
        "Merge the following partial meeting-analysis results. Remove duplicates, "
        "preserve owners and deadlines, and return only the requested numbered list. "
        f"If there are no items, say exactly: {empty_message}"
    )
    return combine.invoke("\n\n".join(outputs))


def extract_action_items(transcript: str) -> str:
    return _extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all the action items. For each provide: \n"
        "- Task description\n"
        "- Owner (person responsible)\n"
        "-Deadline (if mentioned, else 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        "No action items found.",
    )


def extract_key_decisions(transcript: str) -> str:
    return _extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all the key decisions."
        "Format as a numbered list. If none found say 'No key decisions found.'",
        "No key decisions found.",
    )


def extract_questions(transcript: str) -> str:
    return _extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all the questions asked during the meeting."
        "Format as a numbered list. If none found say 'No questions found.'",
        "No questions found.",
    )
