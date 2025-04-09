import streamlit as st
import time
import requests


def refine_question(question: str, history) -> str:
    "Use a LLM to refine the question given the history of the conversation."
    return question


def fetch_rag_documents(question: str):
    "Query the RAG backend for the documents"
    url = "http://localhost:9621/query"
    headers = {"Content-Type": "application/json"}
    payload = {"query": question, "mode": "hybrid", "only_need_context": True}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print(response.json())
        return response.json()
    else:
        response.raise_for_status()


def write_answer(question: str, documents):
    return "42"


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

            # Refine question if we have history to use.
            if len(st.session_state["chat_history"]) > 0:
                status_placeholder.text("Refining question...")
                query_text = refine_question(
                    query_text, st.session_state["chat_history"]
                )
                chat_container.write(f"Refined question: {query_text}")

            time.sleep(1)  # Simulate phase 1

            # Fetch RAG documents
            status_placeholder.text("Fetching relevant document chunks...")
            documents = fetch_rag_documents(query_text)

            # Use LLM to write answer.
            status_placeholder.text("Writing answer...")
            llm_response = write_answer(query_text, documents)
            st.session_state["chat_history"].append(f"LLM: {llm_response}")

        # Clear the status message
        status_placeholder.empty()

        st.rerun()
