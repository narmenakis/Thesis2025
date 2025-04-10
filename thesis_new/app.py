import os
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Constants ---
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Απάντησε στην ερώτηση μόνο με βάση το παρακάτω κείμενο:

{context}

-------

Ερώτηση: {question}
"""

# --- Load Krikri globally ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ilsp/Llama-Krikri-8B-Instruct"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    return tokenizer, model

# --- Vector DB ---
@st.cache_resource
def load_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


# --- Krikri Generator ---
def generate_with_krikri(tokenizer, model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=(input_ids != tokenizer.eos_token_id).long()
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    question_position = decoded_output.find(user_prompt.strip())

    if question_position != -1:
        response_text = decoded_output[question_position + len(user_prompt.strip()):].strip()
        return response_text.lstrip("assistant").strip()
    else:
        return decoded_output.strip()


# --- Core Logic ---
def answer_query(query_text, tokenizer, model, db):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        return "❌ Δεν βρέθηκαν σχετικές πληροφορίες.", "", []

    context_text = "\n\n---\n\n".join([
        f"URL: {doc.metadata.get('url', 'URL λείπει')}\n"
        f"Τίτλος: {doc.metadata.get('title', 'Τίτλος λείπει')}\n"
        f"Ιστοσελίδα: {doc.metadata.get('website', 'Ιστοσελίδα λείπει')}\n"
        f"Ημερομηνία: {doc.metadata.get('datetime', 'Ημερομηνία λείπει')}\n"
        f"Στήλη: {doc.metadata.get('section', 'Στήλη λείπει')}\n\n"
        f"{doc.page_content}"
        for doc, _ in results
    ])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    user_prompt = prompt_template.format(context=context_text, question=query_text)

    system_prompt = (
        "Είσαι ένα εξαιρετικά ανεπτυγμένο μοντέλο Τεχνητής Νοημοσύνης για τα ελληνικά "
        "και απαντάς σε ερωτήσεις δημοσιογραφικού ενδιαφέροντος με αναφορά σε πηγές."
    )

    answer = generate_with_krikri(tokenizer, model, system_prompt, user_prompt)

    # Extract source IDs
    source_ids = [f"{doc.metadata.get('id', 'No ID')}" for doc, _ in results]

    return answer, context_text, source_ids


# --- Main UI ---
def main():
    st.set_page_config(page_title="Krikri Chatbot", page_icon="🤖")
    st.title("💬 Krikri Chatbot για Ελληνικά Κείμενα")

    tokenizer, model = load_model_and_tokenizer()
    db = load_vector_db()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat messages
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    # New user input
    user_question = st.chat_input("Ρώτα κάτι σχετικό με τα δεδομένα...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("💭 Σκέφτομαι..."):
            answer, context, source_ids = answer_query(user_question, tokenizer, model, db)

        with st.chat_message("assistant"):
            st.write(answer)

        # Display the source IDs
        if source_ids:
            st.write("🔗 Πηγές: ", ", ".join(source_ids))

        # Save to chat history
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))


# --- Entry Point ---
if __name__ == "__main__":
    main()
