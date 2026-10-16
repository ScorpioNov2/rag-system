"""
scripts/index_documents.py — Index tài liệu vào vector stores.

Chạy độc lập, không cần agent system.
Hỗ trợ index vào Chroma, TurboVec, hoặc cả hai.

Định dạng hỗ trợ:
    .txt .md .pdf .docx .csv .json .html .epub

Cách dùng:
    python scripts/index_documents.py --file report.pdf
    python scripts/index_documents.py --dir agent_data/
    python scripts/index_documents.py --file data.pdf --store turbovec
    python scripts/index_documents.py --dir docs/ --chunk-size 500 --chunk-overlap 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from turbovec.langchain import TurboQuantVectorStore
from typing_extensions import List

from tools.embeddings import _embeddings

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMA_DIR   = "./db/chroma_db"
TURBOVEC_DIR = "./db/turbovec_db"
COLLECTION   = "arxiv_papers"

# ── Loader registry ────────────────────────────────────────────────────────────
# Mỗi extension map tới 1 loader factory.
# Thêm định dạng mới: chỉ cần thêm 1 dòng vào dict này.

def _get_loader(path: Path):
    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(path), encoding="utf-8")

    elif ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader(str(path))

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(path))

    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(str(path), encoding="utf-8")

    elif ext == ".json":
        from langchain_community.document_loaders import JSONLoader
        return JSONLoader(str(path), jq_schema=".", text_content=False)

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(str(path))

    elif ext == ".epub":
        from langchain_community.document_loaders import UnstructuredEPubLoader
        return UnstructuredEPubLoader(str(path))

    else:
        return None


SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".htm", ".epub"}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_file(path: Path) -> List[Document]:
    loader = _get_loader(path)
    if loader is None:
        print(f"⚠️  Unsupported format: {path.suffix} — skipping {path.name}")
        return []
    print(f"📄 Loading {path.name}  [{path.suffix}]")
    try:
        return loader.load()
    except Exception as e:
        print(f"❌ Failed to load {path.name}: {e}")
        return []


def load_dir(path: Path) -> List[Document]:
    print(f"📁 Scanning {path}...")
    files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"⚠️  No supported files found in {path}")
        print(f"   Supported: {', '.join(sorted(SUPPORTED))}")
        return []

    print(f"   Found {len(files)} file(s): {', '.join(f.name for f in files)}\n")

    docs = []
    for f in files:
        docs.extend(load_file(f))
    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n✂️  {len(docs)} doc(s) → {len(chunks)} chunks  (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── Indexers ───────────────────────────────────────────────────────────────────

def index_chroma(chunks: List[Document]):
    print("🗄️  Indexing into Chroma...")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embeddings,
        collection_name=COLLECTION,
    )
    db.add_documents(chunks)
    print(f"✅ Chroma: {len(chunks)} chunks → {CHROMA_DIR}")


def index_turbovec(chunks: List[Document]):
    print("⚡ Indexing into TurboVec...")
    db_path = Path(TURBOVEC_DIR)

    if (db_path / "index.tvim").exists():
        db = TurboQuantVectorStore.load(
            TURBOVEC_DIR, embedding=_embeddings
        )
        db.add_documents(chunks)
    else:
        db = TurboQuantVectorStore.from_documents(chunks, embedding=_embeddings, bit_width=4)

    db_path.mkdir(exist_ok=True)
    db.dump(TURBOVEC_DIR)
    print(f"✅ TurboVec: {len(chunks)} chunks → {TURBOVEC_DIR}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index documents into vector stores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported formats: {', '.join(sorted(SUPPORTED))}",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Single file to index")
    src.add_argument("--dir",  type=Path, help="Directory — indexes all supported files recursively")

    parser.add_argument("--store", choices=["chroma", "turbovec", "both"],
                        default="both", help="Target vector store (default: both)")
    parser.add_argument("--chunk-size",    type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    # Load
    if args.file:
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        docs = load_file(args.file)
    else:
        if not args.dir.exists():
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        docs = load_dir(args.dir)

    if not docs:
        print("❌ No documents loaded.")
        sys.exit(1)

    # Split
    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)

    # Index
    if args.store in ("chroma", "both"):
        index_chroma(chunks)
    if args.store in ("turbovec", "both"):
        index_turbovec(chunks)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
