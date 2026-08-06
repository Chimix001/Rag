import streamlit as st
from retrieval import ask_question

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Banking Customer Support",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Banking Customer Support Agent")
st.write("Ask any question about our banking services.")

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User Input
# -----------------------------
if prompt := st.chat_input("Ask a banking question..."):

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):

            answer = ask_question(prompt)

            if answer:
                st.markdown(answer)
            else:
                answer = (
                    "Sorry, I couldn't find any information "
                    "related to your question."
                )
                st.warning(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )