from pathlib import Path
from typing import Iterable
from uuid import uuid5, NAMESPACE_URL

import chromadb
import pypdf
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COMPANY_DOCS_FOLDER = BASE_DIR / "company_docs"
DEFAULT_CHROMA_DB_PATH = BASE_DIR / "chroma_db"
SUPPORTED_FILE_TYPES = {".txt", ".pdf"}
embeddings = None
embedding_model_name = None


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    global embeddings, embedding_model_name

    if embeddings is None or embedding_model_name != model_name:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embedding_model_name = model_name

    return embeddings


def create_document_embeddings(
    chunks: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[list[float]]:
    embedding_model = get_embeddings(model_name)
    return embedding_model.embed_documents(chunks)


def create_query_embedding(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float]:
    embedding_model = get_embeddings(model_name)
    return embedding_model.embed_query(query)


def _read_txt_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(file_path: Path) -> str:
    reader = pypdf.PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


def _load_document(file_path: str | Path) -> dict[str, str]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _read_txt_file(path)
    elif suffix == ".pdf":
        text = _read_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .txt or .pdf files.")

    return {
        "source": str(path),
        "text": text.strip(),
    }


def _discover_files(folder_path: str | Path, recursive: bool = True) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder path, got: {folder}")

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES
    ]

    return sorted(files)


def _resolve_input_files(
    folder_path: str | Path | None = None,
    file_paths: Iterable[str | Path] | None = None,
    recursive: bool = True,
) -> list[Path]:
    files = []

    if folder_path is not None:
        files.extend(_discover_files(folder_path, recursive))

    if file_paths is not None:
        for file_path in file_paths:
            path = Path(file_path)
            if path.is_dir():
                files.extend(_discover_files(path, recursive))
            elif path.suffix.lower() in SUPPORTED_FILE_TYPES:
                files.append(path)

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise ValueError("No .txt or .pdf files found to index.")

    return unique_files


def _chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - chunk_overlap

    return [chunk for chunk in chunks if chunk.strip()]


def get_relevant_chunks(
    query: str,
    folder_path: str | Path | None = DEFAULT_COMPANY_DOCS_FOLDER,
    *,
    file_paths: Iterable[str | Path] | None = None,
    recursive: bool = True,
    top_k: int = 5,
    collection_name: str = "company_documents",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str | Path = DEFAULT_CHROMA_DB_PATH,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict[str, object]]:
    """
    Return relevant text chunks from every .txt/.pdf file in a folder.

    Args:
        query: User question/search text.
        folder_path: Folder containing company .txt/.pdf files.
            Defaults to this project's company_docs folder.
        file_paths: Optional extra .txt/.pdf files or folders to include.
        recursive: If True, searches subfolders too.
        top_k: Number of relevant chunks to return.
        collection_name: ChromaDB collection name.
        embedding_model: Hugging Face sentence-transformer model name.
        persist_directory: ChromaDB storage folder.
        chunk_size: Chunk size measured in words.
        chunk_overlap: Overlap between chunks measured in words.
    """
    if not query.strip():
        raise ValueError("query cannot be empty.")

    resolved_files = _resolve_input_files(folder_path, file_paths, recursive)
    documents = [_load_document(file_path) for file_path in resolved_files]

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    old_data = collection.get()
    if old_data["ids"]:
        collection.delete(ids=old_data["ids"])

    ids = []
    chunks = []
    metadatas = []

    for document in documents:
        for chunk_number, chunk in enumerate(
            _chunk_text(document["text"], chunk_size, chunk_overlap),
            start=1,
        ):
            chunk_id = uuid5(NAMESPACE_URL, f"{document['source']}:{chunk_number}")
            ids.append(str(chunk_id))
            chunks.append(chunk)
            metadatas.append(
                {
                    "source": document["source"],
                    "chunk_number": chunk_number,
                }
            )

    if not chunks:
        return []

    document_embeddings = create_document_embeddings(chunks, embedding_model)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=document_embeddings,
        metadatas=metadatas,
    )

    query_embedding = create_query_embedding(query, embedding_model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(chunks)),
    )

    matched_chunks = []
    for index, chunk in enumerate(results["documents"][0]):
        matched_chunks.append(
            {
                "source": Path(results["metadatas"][0][index]["source"]).name,
                "chunk": chunk,
            }
        )

    return matched_chunks


if __name__ == "__main__":
    user_query = input("Enter the query: ")

    for result in get_relevant_chunks(user_query):
        print(f"\nSource: {result['source']}")
        print(result["chunk"])

    


