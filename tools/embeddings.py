import sys
import os
from langchain_huggingface import HuggingFaceEmbeddings


def _detect_device() -> str:
    if sys.platform.startswith('win') and "DriverData" in os.environ:
        return "cuda"
    elif sys.platform.startswith('linux') and ("CUDA_VERSION" in os.environ or "CUDA_HOME" in os.environ):
        return "cuda"
    elif sys.platform.startswith('darwin'):
        return "cuda"
    return "cpu"


def create_embeddings(model: str = "BAAI/bge-m3") -> HuggingFaceEmbeddings:
    """
    Load HuggingFace embedding model về local.

    Device được tự detect theo thứ tự ưu tiên: CUDA → MPS → CPU.
    Model được cache tại ~/.cache/huggingface/ — dùng chung cho mọi project.

    Args:
        model: HuggingFace model name. Mặc định bge-m3 (đa ngôn ngữ, ~1.5GB RAM).
    """
    model = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", model)

    device = _detect_device()
    print(f"🖥️  Embedding device: {device.upper()}")

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={
            "device": device,
            "token": "hf_token ..."
        },
        encode_kwargs={"normalize_embeddings": True},
    )


# Singleton — load 1 lần, dùng chung cho Chroma và TurboVec
_embeddings = create_embeddings()
