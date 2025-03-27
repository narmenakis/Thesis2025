# from langchain.document_loaders import DirectoryLoader
import argparse
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

# # needed this line of code in order to work for the first time, then comment out
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 

openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"
CSV_FILE_PATH = "/Users/narmen/CEID/Διπλωματική/thesis/source/output.csv"


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    generate_data_store()
    
   
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    add_to_chroma(chunks)
     
    # # Print metadata of the first chunk to check if the IDs are assigned properly
    # print("🔍 Example of first chunk metadata after processing:")
    # print(chunks[50].metadata)  # Printing metadata of the first chunk

def load_documents():
    # Load the CSV file using CSVLoader
    df = pd.read_csv(CSV_FILE_PATH)

    documents = [
        Document(
            page_content=row["text"],  # The main article content
            metadata={
                "author": row["author"],
                "title": row["title"],
                "link": row["link"],
                "row_index": index
            }
        )
        # extract text, and attach metadata to each document
        for index, row in df.iterrows()
    ]

    print(f"Loaded {len(documents)} documents.")  # Debugging: Ensure documents are loaded

    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # # Print the 11th chunk to see the content and metadata
    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

def add_to_chroma(chunks: list[Document]):
    # 🔹 Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 🔹 Ensure 'id' is inside metadata
    for chunk in chunks_with_ids:
        chunk.metadata["id"] = chunk.metadata["id"]  # Explicitly assign it

    # 🔹 Load or create ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    # 🔹 Retrieve existing documents
    existing_items = db.get(include=["metadatas"])  
    existing_ids = {item.get("id") for item in existing_items["metadatas"] if "id" in item}

    # 🔹 Add only new documents
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new chunks")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        # 🛠 Fix: Ensure metadata includes 'id' when adding
        for chunk in new_chunks:
            chunk.metadata["id"] = chunk.metadata["id"]  

        db.add_documents(new_chunks, ids=new_chunk_ids)  # Ensure IDs are used
        # db.persist()  
        print("✅ New chunks added successfully!")

        # Debug: Print stored metadata
        stored_documents = db.get(include=["metadatas"])
        # print("🔍 Stored Documents Metadata in ChromaDB:")
        # for i, meta in enumerate(stored_documents["metadatas"][:10]):  
        #     print(f"Document {i+1}: {meta}")

    else:
        print("✅ No new chunks to add")


def calculate_chunk_ids(chunks):
    """
    Assigns unique chunk IDs based on document metadata (CSV format).
    Example ID format: "link:row_in_csv:chunk_index"
    """

    last_row_id = None
    current_chunk_index = 0

    ## Chunk indexing !!!
    for chunk in chunks:
        # Extract metadata
        link = chunk.metadata.get("link", "Unknown Link")
        row_in_csv = chunk.metadata.get("row_index")

        # Create a unique row ID
        current_row_id = f"{link}:{row_in_csv}"

        # Increment chunk index if it's the same row as the last one
        if current_row_id == last_row_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign the chunk ID
        chunk_id = f"{current_row_id}:{current_chunk_index}"
        last_row_id = current_row_id

        # Store in metadata
        chunk.metadata["id"] = chunk_id

        # # Debugging: Print the chunk id and metadata to check if id is properly assigned
        # print(f"Assigned ID: {chunk.metadata['id']} to chunk with metadata: {chunk.metadata}")

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()