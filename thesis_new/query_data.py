import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Απάντησε στην ερώτηση μόνο με βάση το παρακάτω κείμενο:

{context}

-------

Ερώτηση: {question}
"""


# Load Krikri model once globally
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ilsp/Llama-Krikri-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
model.to(device)

def generate_with_krikri(system_prompt, user_prompt):
    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generate the prompt using tokenizer
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate the output from the model
    output = model.generate(
        input_ids, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7, 
        
        pad_token_id=tokenizer.eos_token_id,  # treat padding tokens like EOS tokens
        attention_mask=(input_ids != tokenizer.eos_token_id).long()  # which tokens to pay attention to
    )
    
    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Trim everything before and including the question part
    question_text = user_prompt.strip()  # Get the question from the user prompt
    
    # Find the position of the question in the output
    question_position = decoded_output.find(question_text)

    if question_position != -1:
        # The response starts after the question text
        response_text = decoded_output[question_position + len(question_text):].strip()
        
        # Optional: Remove any leading part like 'assistant:' or other unwanted texts
        response_text = response_text.lstrip('assistant').strip()
    else:
        # If the question isn't found, return the entire decoded output (or handle as an error)
        response_text = decoded_output.strip()

    return response_text
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Set up vector DB
    # embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"❌ Unable to find matching results.")
        return

    # Format context
    context_text = "\n\n---\n\n".join([
        f"URL: {doc.metadata.get('url', 'URL λείπει')}\n"
        f"Τίτλος: {doc.metadata.get('title', 'Τίτλος λείπει')}\n"
        f"Ιστοσελίδα: {doc.metadata.get('website', 'Ιστοσελίδα λείπει')}\n"
        f"Ημερομηνία: {doc.metadata.get('datetime', 'Ημερομηνία λείπει')}\n"
        f"Στήλη: {doc.metadata.get('section', 'Στήλη λείπει')}\n\n"
        f"{doc.page_content}"
        for doc, _ in results
    ])

    # Prepare final prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    user_prompt = prompt_template.format(context=context_text, question=query_text)
    
    system_prompt = "Είσαι ένα εξαιρετικά ανεπτυγμένο μοντέλο Τεχνητής Νοημοσύνης για τα ελληνικά και απαντάς σε ερωτήσεις δημοσιογραφικού ενδιαφέροντος με αναφορά σε πηγές. " 

    # Generate response using Krikri
    response_text = generate_with_krikri(system_prompt, user_prompt)

    # Sources
    sources = [f"{doc.metadata.get('id', 'No ID')}" for doc, _ in results]

    formatted_response = f"\nΑπάντηση: \n{response_text}\n\n📚 Context:\n{context_text}\n\n🔗 Πηγές: {', '.join(sources)}"
    print(formatted_response)
    
if __name__ == "__main__":
    main()
