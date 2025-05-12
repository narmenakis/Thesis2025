import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List
import streamlit as st
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime

# Disable CUDA memory caching to save GPU memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Fix PyTorch, Streamlit compatibility issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Constants
CHROMA_PATH = "chroma"
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model_name = "ilsp/Llama-Krikri-8B-Instruct"
embedding_model_name = "intfloat/multilingual-e5-large-instruct"

class E5InstructEmbeddings:
    def __init__(self):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.model.eval()
        
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents by adding 'passage: ' prefix"""
        instruction_texts = ["passage: " + text for text in texts]
        return self._embed(instruction_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query by adding proper instruction prefix"""
        instruction_text = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}'
        return self._embed([instruction_text])[0]
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts, 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_vector_db():
    embedding_function = E5InstructEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def generate_with_krikri(tokenizer, model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=(input_ids != tokenizer.eos_token_id).long()
    )

    generated_tokens = output[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not decoded_output:
        return "⚠️ Δεν κατάφερα να δημιουργήσω απάντηση. Δοκίμασε να επαναδιατυπώσεις την ερώτηση."

    return decoded_output

def answer_query(query_text, tokenizer, model, db, chat_history, author_filter, website_filter, section_filter, start_date, end_date, max_docs):
    filter_dict = {}

    if section_filter:
        filter_dict["section"] = {"$in": section_filter}
    if author_filter:
        filter_dict["author"] = {"$in": author_filter}
    if website_filter:
        filter_dict["website"] = {"$in": website_filter}
    if start_date or end_date:
        date_range = {}
        if start_date:
            date_range["$gte"] = start_date.isoformat()
        if end_date:
            date_range["$lte"] = end_date.isoformat()
        filter_dict["datetime"] = date_range

    results = db.similarity_search_with_relevance_scores(query_text, k=50, filter=filter_dict if filter_dict else None)

    if not results:
        return "❌ Δεν βρέθηκαν σχετικές πληροφορίες με τα επιλεγμένα φίλτρα.", "", []

    filtered_results = sorted(
        [(doc, score) for doc, score in results if score >= 0.50],
        key=lambda x: x[1],
        reverse=True
    )[:max_docs]

    if len(filtered_results) == 0:
        return "❌ Δεν βρέθηκαν σχετικές πληροφορίες.", "", []

    context_text = "\n\n---\n\n".join([
        f"URL: {doc.metadata.get('url', 'URL λείπει')}\n"
        f"Τίτλος: {doc.metadata.get('title', 'Τίτλος λείπει')}\n"
        f"Ιστοσελίδα: {doc.metadata.get('website', 'Ιστοσελίδα λείπει')}\n"
        f"Ημερομηνία: {doc.metadata.get('datetime', 'Ημερομηνία λείπει')}\n"
        f"Στήλη: {doc.metadata.get('section', 'Στήλη λείπει')}\n\n"
        f"{doc.page_content}"
        for doc, _ in filtered_results
    ])

    conversation_history = ""
    for role, msg, _ in chat_history:
        role_prefix = "Χρήστης:" if role == "user" else "Βοηθός:"
        conversation_history += f"{role_prefix} {msg}\n"

    user_prompt = f"""
Συνομιλία μέχρι τώρα:
{conversation_history}

Βασισμένος ΜΟΝΟ στο παρακάτω κείμενο και στην Συνομιλία μέχρι τώρα, απάντησε στην ερώτηση:

{context_text}

-------

Ερώτηση: {query_text}
"""

    system_prompt = (
        "Είσαι ένα ανεπτυγμένο μοντέλο Τεχνητής Νοημοσύνης ειδικά σχεδιασμένο για την Ελληνική γλώσσα. "
        "Απαντάς σε ερωτήσεις δημοσιογραφικού ενδιαφέροντος με βάση διαθέσιμα άρθρα. "
        "Πρέπει να απαντάς με αντικειμενικότητα, ουδετερότητα και χωρίς εικασίες. "
        "ΜΗΝ προσθέτεις προσωπικές υποθέσεις ή πληροφορίες που δεν αναφέρονται ρητά στα κείμενα. "
        "Δώσε έμφαση στα άρθρα τα οποία σχετίζονται με τη Συνομιλία μέχρι τώρα. "
    )

    answer = generate_with_krikri(tokenizer, model, system_prompt, user_prompt)
    
    sources = [
        (doc.metadata.get('url', 'URL λείπει'), round(score, 2))
        for doc, score in filtered_results
    ]

    return answer, context_text, sources

def main():
    st.set_page_config(page_title="Krikri Chatbot", page_icon="🤖")
    st.title("💬 Krikri Chatbot για Ελληνικά Κείμενα")

    tokenizer, model = load_model_and_tokenizer()
    db = load_vector_db()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar filters
    st.sidebar.header("📊 Φίλτρα")
    all_metadata = db.get()['metadatas']
    all_authors = sorted(set(meta.get("author") for meta in all_metadata if meta.get("author")))
    all_websites = sorted(set(meta.get("website") for meta in all_metadata if meta.get("website")))
    all_sections = sorted(set(meta.get("section") for meta in all_metadata if meta.get("section")))

    author_filter = st.sidebar.multiselect("Συγγραφέας", options=all_authors)
    website_filter = st.sidebar.multiselect("Ιστοσελίδα", options=all_websites)
    section_filter = st.sidebar.multiselect("Στήλη", options=all_sections)
    start_date = st.sidebar.date_input("Από Ημερομηνία", value=None)
    end_date = st.sidebar.date_input("Έως Ημερομηνία", value=None)

    max_docs = st.sidebar.slider("Μέγιστος αριθμός εγγράφων", min_value=1, max_value=20, value=10)

    for role, msg, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)
        if role == "assistant" and sources:
            with st.expander("🔗 Δες τις Πηγές"):
                for source in sources:
                    st.write(f"- {source}")

    user_question = st.chat_input("Ρώτα κάτι σχετικό με τα δεδομένα...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append(("user", user_question, []))
        if len(st.session_state.chat_history) > 30:
            st.session_state.chat_history = st.session_state.chat_history[-30:]

        with st.spinner("💭 Σκέφτομαι..."):
            answer, context, urls = answer_query(
                user_question, tokenizer, model, db, st.session_state.chat_history,
                author_filter, website_filter, section_filter,
                start_date if start_date else None,
                end_date if end_date else None,
                max_docs
            )

        with st.chat_message("assistant"):
            st.write(answer)
            if urls:
                with st.expander("🔗 Δες τις Πηγές"):
                    for url, score in urls:
                        st.write(f"- **[{score}]** {url}")

        st.session_state.chat_history.append(("assistant", answer, urls))
        if len(st.session_state.chat_history) > 30:
            st.session_state.chat_history = st.session_state.chat_history[-30:]

if __name__ == "__main__":
    main()