"""
scripts/rag_cli.py — Interactive CLI để test RAG độc lập.

Không cần chạy agent, không cần LLM — chỉ embed + retrieve + hiển thị kết quả.

Cách dùng:
    python scripts/rag_cli.py                      # test cả 2 stores
    python scripts/rag_cli.py --store chroma       # chỉ Chroma
    python scripts/rag_cli.py --store turbovec     # chỉ TurboVec
    python scripts/rag_cli.py --k 3                # lấy top-3 thay vì top-5
    python scripts/rag_cli.py --compare            # chạy cả 2 và so sánh kết quả
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from turbovec.langchain import TurboQuantVectorStore
from tools.embeddings import _embeddings

CHROMA_DIR   = "./db/chroma_db"
TURBOVEC_DIR = "./db/turbovec_db"
COLLECTION   = "arxiv_papers"


# ── Store loaders ──────────────────────────────────────────────────────────────

def load_chroma(k: int):
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embeddings,
        collection_name=COLLECTION,
    )
    return db.as_retriever(search_kwargs={"k": k})


def load_turbovec(k: int):
    db_path = Path(TURBOVEC_DIR)
    if not (db_path / "index.tvim").exists():
        print("⚠️  TurboVec index not found. Run index_documents.py first.")
        return None
    db = TurboQuantVectorStore.load(
        TURBOVEC_DIR, embedding=_embeddings
    )
    return db.as_retriever(search_kwargs={"k": k})


# ── Display ────────────────────────────────────────────────────────────────────

def print_results(docs, store_name: str, elapsed: float):
    print(f"\n{'═'*60}")
    print(f"  {store_name}  ({elapsed*1000:.0f}ms)  —  {len(docs)} results")
    print(f"{'═'*60}")
    if not docs:
        print("  (no results)")
        return
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = meta.get("title", meta.get("source", "—"))
        print(f"\n  [{i}] {title}")
        print(f"  {doc.page_content[:400].strip()}")
        if len(doc.page_content) > 400:
            print("  ...")


def query_store(retriever, query: str, store_name: str):
    t0 = time.time()
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"❌ {store_name} error: {e}")
        return
    print_results(docs, store_name, time.time() - t0)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test RAG retrieval interactively.")
    parser.add_argument("--store", choices=["chroma", "turbovec", "both"],
                        default="both", help="Which store to query (default: both)")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--compare", action="store_true",
                        help="Show both stores side by side for comparison")
    args = parser.parse_args()

    # Load retrievers
    retrievers = {}
    if args.store in ("chroma", "both"):
        retrievers["Chroma"] = load_chroma(args.k)
    if args.store in ("turbovec", "both") or args.compare:
        r = load_turbovec(args.k)
        if r:
            retrievers["TurboVec"] = r

    if not retrievers:
        print("❌ No stores available.")
        sys.exit(1)

    print(f"\n🔍 RAG CLI — stores: {', '.join(retrievers)} | top-k: {args.k}")
    print("   Type your query and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue

        if args.compare and len(retrievers) == 2:
            # Side-by-side comparison
            names = list(retrievers.keys())
            results = {}
            for name in names:
                t0 = time.time()
                try:
                    results[name] = (retrievers[name].invoke(query), time.time() - t0)
                except Exception as e:
                    results[name] = ([], 0)
                    print(f"❌ {name}: {e}")

            for name in names:
                docs, elapsed = results[name]
                print_results(docs, name, elapsed)

            # Highlight overlap
            if all(results[n][0] for n in names):
                texts_a = {d.page_content[:100] for d in results[names[0]][0]}
                texts_b = {d.page_content[:100] for d in results[names[1]][0]}
                overlap = len(texts_a & texts_b)
                print(f"\n  📊 Overlap: {overlap}/{args.k} results in common\n")
        else:
            for name, retriever in retrievers.items():
                query_store(retriever, query, name)


if __name__ == "__main__":
    main()
