from typing import List
import streamlit as st
import time
import requests
import openai
from abstract_ranker.openai_utils import get_key


def refine_question(question: str, history) -> str:
    "Use a LLM to refine the question given the history of the conversation."
    prompt = f"""
    Below is a question the user is asking. Generate a query that can be used to
    search for relevant documents to answer the user's question. Use prior conversation
    history to make the question unambiguous.

    User Question: {question}

    Prior user and LLM conversation:
    {"\n".join(history)}

    Provide a good document search query to be used in a vector database lookup.
    """

    # Call OpenAI GPT-4o API
    try:
        openai_client = openai.OpenAI(api_key=get_key())
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in particle physics and a helpful assistant "
                    "who answers questions accurately and concisely.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Extract and return the answer from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"

    return question


def fetch_rag_documents(question: str):
    "Query the RAG backend for the documents"
    url = "http://localhost:9621/query"
    headers = {"Content-Type": "application/json"}
    payload = {"query": question, "mode": "hybrid", "only_need_context": True}

    # TODO: tune things like top_k, etc.

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print(response.json())
        return response.json()["response"]
    else:
        response.raise_for_status()


def write_answer(question: str, history: List[str], documents: str):
    """
    Use OpenAI GPT-4o to answer the question based on the provided documents.
    """
    # Prepare the prompt for GPT-4o
    prompt = f"""
    Answer the following question based on recent conversation history with the user and documents
    that should be relevant to the question.

    Question: {question}

    Conversation History:
    {"\n".join(history)}

    Reference Documents:
    {documents}

    Provide an accurate answer.
    """

    # Call OpenAI GPT-4o API
    try:
        openai_client = openai.OpenAI(api_key=get_key())
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in particle physics and a helpful assistant "
                    "who answers questions accurately and concisely.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Extract and return the answer from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"


# Set up the Streamlit page
st.set_page_config(page_title="Conference Chat", layout="wide")

# Header with the conference name
st.title("European Strategy Submission Archive")

# Chat record section
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
# st.subheader("Chat Record")

chat_container = st.container()
with chat_container:
    for message in st.session_state["chat_history"]:
        st.write(message)

# Create a placeholder for status messages
status_placeholder = st.empty()

# Input box for user questions
user_input = st.text_area(
    "Ask a question:",
    key="user_input",
    height=100,
    value="",
)

# Handle Shift+Enter to submit
if st.button("Submit", key="submit_button"):
    if user_input.strip():
        # Add user input to chat history
        st.session_state["chat_history"].append(f"You: {user_input}")
        chat_container.write(f"You: {user_input}")

        query_text = user_input

        # Show spinner and update status messages dynamically
        with st.spinner("Processing your question...", show_time=True):

            # Consider only the last 10 responses or so.
            history = st.session_state["chat_history"][:10]

            # Refine question if we have history to use.
            if len(st.session_state["chat_history"]) > 1:
                status_placeholder.text("Refining question...")
                query_text = refine_question(query_text, history)
                chat_container.write(f"Refined question: {query_text}")
                print(f"refined question: {query_text}")

            time.sleep(1)  # Simulate phase 1

            # Fetch RAG documents
            status_placeholder.text("Fetching relevant document chunks...")
            documents = fetch_rag_documents(query_text)

            # Use LLM to write answer.
            status_placeholder.text("Writing answer...")
            llm_response = write_answer(user_input, history, documents)
            st.session_state["chat_history"].append(f"LLM: {llm_response}")

        # Clear the status message
        status_placeholder.empty()

        st.rerun()
