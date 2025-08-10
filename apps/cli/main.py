# main.py
import os
import logging
import warnings
import asyncio
from src.financial_rag.parsers.parser import MultimodalParser
from src.financial_rag.retrieval.engine import RetrievalEngine
from src.financial_rag.utils import helpers
from src.financial_rag.config import config
warnings.filterwarnings("ignore")
logging.getLogger("camelot").setLevel(logging.ERROR)
logging.getLogger("tabula").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def main():
    file_path = config.INPUT_FILE_PATH
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    helpers.print_status("Initializing Parser")
    parser = MultimodalParser()
    docs = parser.parse(file_path)

    helpers.print_status("Initializing Retrieval Engine")
    engine = RetrievalEngine()
    engine.build_pipeline(docs)

    helpers.print_status("Executing Queries")
    queries = [
    "What was the primary reason for the increase in iPhone net sales during the third quarter of 2022 compared to the same quarter in 2021?",

    "Which two new Mac models powered by the M2 chip were introduced at the end of the third quarter of 2022?",
    
    "What operating system updates did Apple announce in the third quarter of 2022 that were expected to be available in fall 2022?",

    ]

    for q in queries:
        result = engine.query(q)
        print("\nAnswer:")
        print(result.get('result', 'No answer was generated.'))
        helpers.print_separator()

if __name__ == "__main__":
    main()
