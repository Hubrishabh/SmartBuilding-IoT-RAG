import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from backend.retriever import build_index
if __name__ == '__main__':
    build_index(clear=True)