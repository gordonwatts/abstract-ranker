import streamlit as st
import time

# Set up the Streamlit page
st.set_page_config(page_title="Conference Chat", layout="wide")

# Header with the conference name
st.title("European Strategy Submission Archive")

# Chat record section
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
if len(st.session_state["chat_history"]) != 0:
    st.subheader("Chat Record")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state["chat_history"]:
            st.write(message)

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

        # Show spinner and status message
        with st.spinner("Processing your question...", show_time=True):
            st.session_state["chat_history"].append("LLM: [Phase 1] Initializing...")
            time.sleep(1)  # Simulate phase 1

            st.session_state["chat_history"].append(
                "LLM: [Phase 2] Generating response..."
            )
            time.sleep(2)  # Simulate phase 2

            # Simulate LLM response (replace this with actual LLM call)
            llm_response = f"LLM Response to: {user_input}"
            st.session_state["chat_history"].append(f"LLM: {llm_response}")
            st.rerun()
