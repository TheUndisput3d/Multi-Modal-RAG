import os
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.partition.pdf import partition_pdf
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import tabula
import camelot
import re
from bs4 import BeautifulSoup
from ..config import config

class MultimodalParser:
    """
    Enhanced parser that handles text, tables, and figures with improved table processing.
    """

    def __init__(self):
        """Initializes the parser and loads the VLM."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing VLM ({config.VLM_MODEL_ID}) on {self.device}...")

        self.processor = BlipProcessor.from_pretrained(config.VLM_MODEL_ID)
        self.model = BlipForConditionalGeneration.from_pretrained(config.VLM_MODEL_ID).to(self.device)
        
        # Initialize text splitter for better chunk management
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        print("VLM initialized successfully.")

    def _describe_image(self, image_path: str) -> str:
        """Uses the loaded VLM to generate a detailed description of an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=64)
            description = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return f"Figure Description: {description}"
        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            return "Figure Description: Error during generation."

    def _extract_images(self, pdf_path: str) -> list[str]:
        """Extracts all images from a PDF and saves them to the output directory."""
        output_dir = config.IMAGE_OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(image_path)

        doc.close()
        return image_paths

    def _extract_tables_camelot(self, pdf_path: str, source_name: str) -> list[Document]:
        """Extract tables using Camelot for better table detection."""
        docs = []
        try:
            # Extract tables using Camelot
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                # Convert to readable format
                df = table.df
                
                # Clean up the dataframe
                df = df.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
                
                if not df.empty:
                    # Create a structured representation
                    table_content = self._format_table_content(df, i)
                    
                    docs.append(Document(
                        page_content=table_content,
                        metadata={
                            "type": "table",
                            "source": source_name,
                            "table_id": i,
                            "page": table.page
                        }
                    ))
            
            print(f"Extracted {len(docs)} tables using Camelot")
            
        except Exception as e:
            print(f"Error extracting tables with Camelot: {e}")
            
        return docs

    def _extract_tables_tabula(self, pdf_path: str, source_name: str) -> list[Document]:
        """Extract tables using Tabula as fallback."""
        docs = []
        try:
            # Extract tables using Tabula
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(dfs):
                if not df.empty and len(df.columns) > 1:
                    table_content = self._format_table_content(df, i)
                    
                    docs.append(Document(
                        page_content=table_content,
                        metadata={
                            "type": "table",
                            "source": source_name,
                            "table_id": f"tabula_{i}",
                            "extraction_method": "tabula"
                        }
                    ))
            
            print(f"Extracted {len(docs)} tables using Tabula")
            
        except Exception as e:
            print(f"Error extracting tables with Tabula: {e}")
            
        return docs

    def _format_table_content(self, df: pd.DataFrame, table_id: int) -> str:
        """Format table content in a more readable and searchable way."""
        content_parts = [f"=== TABLE {table_id} ===\n"]
        
        # Add headers
        headers = [str(col).strip() for col in df.columns if str(col).strip()]
        if headers:
            content_parts.append(f"Headers: {' | '.join(headers)}\n")
        
        # Add data rows with clear formatting
        for idx, row in df.iterrows():
            row_data = []
            for col in df.columns:
                value = str(row[col]).strip()
                if value and value != 'nan' and value != 'NaN':
                    row_data.append(f"{col}: {value}")
            
            if row_data:
                content_parts.append(f"Row {idx}: {' | '.join(row_data)}")
        
        # Also include the raw table as markdown for context
        content_parts.append("\n--- Raw Table (Markdown) ---")
        content_parts.append(df.to_string(index=False))
        
        return "\n".join(content_parts)

    def _process_html_tables(self, html_content: str) -> str:
        """Process HTML table content to make it more readable."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all tables
            tables = soup.find_all('table')
            
            processed_content = []
            for i, table in enumerate(tables):
                processed_content.append(f"\n=== HTML TABLE {i} ===")
                
                # Process headers
                headers = []
                header_rows = table.find_all('tr')
                if header_rows:
                    header_cells = header_rows[0].find_all(['th', 'td'])
                    headers = [cell.get_text(strip=True) for cell in header_cells]
                    processed_content.append(f"Headers: {' | '.join(headers)}")
                
                # Process data rows
                for row_idx, row in enumerate(table.find_all('tr')[1:]):  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    cell_data = [cell.get_text(strip=True) for cell in cells]
                    if any(cell_data):  # Only add non-empty rows
                        row_text = ' | '.join(cell_data)
                        processed_content.append(f"Row {row_idx}: {row_text}")
                
                processed_content.append("=" * 30)
            
            return '\n'.join(processed_content)
            
        except Exception as e:
            print(f"Error processing HTML table: {e}")
            return html_content

    def parse(self, file_path: str, original_filename: str = None) -> list[Document]:
        """Main parsing method with enhanced table handling."""
        print(f"Starting enhanced multimodal parsing for {file_path}...")
        
        # Use original filename if provided, otherwise use basename of file_path
        source_name = original_filename if original_filename else os.path.basename(file_path)
        
        docs = []
        
        # 1. Extract tables using multiple methods
        print("Extracting tables with Camelot...")
        table_docs_camelot = self._extract_tables_camelot(file_path, source_name)
        docs.extend(table_docs_camelot)
        
        print("Extracting tables with Tabula...")
        table_docs_tabula = self._extract_tables_tabula(file_path, source_name)
        docs.extend(table_docs_tabula)
        
        # 2. Extract text and additional tables with unstructured
        print("Extracting text and additional content with Unstructured.io...")
        elements = partition_pdf(
            filename=file_path,
            strategy=config.TABLE_EXTRACTION_STRATEGY,
            infer_table_structure=True,
            model_name="yolox"
        )
        
        for element in elements:
            element_type = str(type(element))
            
            if "Table" in element_type:
                # Process HTML tables more thoroughly
                if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                    processed_table = self._process_html_tables(element.metadata.text_as_html)
                    docs.append(Document(
                        page_content=processed_table,
                        metadata={
                            "type": "table_html",
                            "source": source_name,
                            "extraction_method": "unstructured"
                        }
                    ))
                else:
                    docs.append(Document(
                        page_content=str(element.text),
                        metadata={
                            "type": "table_text",
                            "source": source_name,
                            "extraction_method": "unstructured"
                        }
                    ))
            else:
                # Regular text content - split into chunks
                text_content = str(element.text)
                if len(text_content.strip()) > 50:  # Only process meaningful content
                    chunks = self.text_splitter.split_text(text_content)
                    for chunk in chunks:
                        docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "type": "text",
                                "source": source_name,
                                "extraction_method": "unstructured"
                            }
                        ))

        # 3. Extract and describe images
        print("Extracting figures with PyMuPDF...")
        image_paths = self._extract_images(file_path)
        if image_paths:
            print(f"Describing {len(image_paths)} extracted figures with VLM...")
            for image_path in image_paths:
                description = self._describe_image(image_path)
                docs.append(Document(
                    page_content=description,
                    metadata={
                        "type": "figure",
                        "source": image_path,  # Keep original image path for figures
                        "original_document": source_name,  # Add reference to original document
                        "extraction_method": "vlm"
                    }
                ))
        else:
            print("No figures found in the document.")

        print(f"Enhanced multimodal parsing complete. Extracted {len(docs)} documents.")
        return docs