import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import os
import shutil
# from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings

# embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
# embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


CHROMA_PATH = "chroma"
CSV_FILE_PATH = "output.csv"


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
            page_content=row["full-text"],  # The main article content
            metadata={
                "url": row["url"],
                "title": row["title"],  
                "author": row["author"],  
                "website": row["website"],  
                "datetime": row["datetime"],  
                "section": row["section"],  
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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks

def add_to_chroma(chunks: list[Document]):
    # 🔹 Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 🔹 Ensure 'id' is inside metadata
    for chunk in chunks_with_ids:
        chunk.metadata["id"] = chunk.metadata["id"]  # Explicitly assign it

    # 🔹 Load or create ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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
        url = chunk.metadata.get("url", "Unknown URL")
        row_in_csv = chunk.metadata.get("row_index")

        # Create a unique row ID
        current_row_id = f"{url}:{row_in_csv}"

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