import argparse
# from dataclasses import dataclass
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB. !!!
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

#   # Debugging: Print metadata for the first result
#     for i, (doc, score) in enumerate(results):
#         print(f"Result {i + 1} metadata: {doc.metadata}")

    context_text = "\n\n---\n\n".join([
    f"URL: {doc.metadata.get('url', 'URL λείπει')}\n"
    f"Τίτλος: {doc.metadata.get('title', 'Τίτλος λείπει')}\n"
    # f"ID: {doc.metadata.get('id', 'No ID')}\n"
    f"Ιστοσελίδα: {doc.metadata.get('website', 'Ιστοσελίδα λείπει')}\n"
    f"Ημερομηνία: {doc.metadata.get('datetime', 'Ημερομηνία λείπει')}\n"
    f"Στήλη: {doc.metadata.get('section', 'Στήλη λείπει')}\n\n"
    f"{doc.page_content}"
    for doc, _score in results
    ])

                # "url": row["url"],
                # "title": row["title"],  
                # "author": row["author"],  
                # "website": row["website"],  
                # "datetime": row["datetime"],  
                # "section": row["section"],

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOpenAI()
    # model = AutoModelForCausalLM.from_pretrained("ilsp/Llama-Krikri-8B-Instruct")

    response = model.invoke([{"role": "user", "content": prompt}])
    response_text = response.content  # Extract only the model's answer

    # # we can put w/e we want in the metadata i.e. link, title etc
    # sources = [doc.metadata.get("title", "No Source") for doc, _score in results]

    # Modify sources to include the format link:row_in_csv:chunk_id
    # Use the "id" field directly from the metadata
    sources = [f"{doc.metadata.get('id', 'No ID')}" for doc, _score in results]


    formatted_response = f"\n{response_text}\n\nContext:\n{context_text}\n\nSources: {', '.join(sources)}"    
    print(formatted_response)

if __name__ == "__main__":
    main()
