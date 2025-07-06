# import fitz  # PyMuPDF
# import easyocr
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# from typing import List, Dict
# import json

# class PDFExtractor:
#     def __init__(self):
#         # Initialize EasyOCR once (lazy-load to save memory)
#         self.reader = None  
        
#     def _init_easyocr(self):
#         if self.reader is None:
#             self.reader = easyocr.Reader(['en'])  # Add more languages if needed

#     def extract_text_with_pymupdf(self, pdf_path: str) -> List[Dict]:
#         """Extract selectable text from PDF using PyMuPDF (fast)."""
#         text_chunks = []
#         doc = fitz.open(pdf_path)
#         for page_num, page in enumerate(doc):
#             text = page.get_text("text")
#             if text.strip():
#                 text_chunks.append({
#                     "content": text,
#                     "page": page_num + 1,
#                     "type": "text",
#                     "source": pdf_path
#                 })
#         return text_chunks

#     def extract_images_with_easyocr(self, pdf_path: str) -> List[Dict]:
#         """Extract text from images/scanned PDFs using EasyOCR."""
#         self._init_easyocr()
#         images = convert_from_path(pdf_path)
#         ocr_chunks = []
#         for page_num, img in enumerate(images):
#             img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             results = self.reader.readtext(img_cv, paragraph=True)
#             for (bbox, text, confidence) in results:
#                 ocr_chunks.append({
#                     "content": text,
#                     "page": page_num + 1,
#                     "type": "ocr_text",
#                     "source": pdf_path,
#                     "confidence": float(confidence)
#                 })
#         return ocr_chunks

#     def detect_charts(self, image_path: str) -> Dict:
#         """Detect charts using OpenCV (simplified example)."""
#         img = cv2.imread(image_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         chart_type = "unknown"
#         if len(contours) > 5:  # Heuristic for chart detection
#             chart_type = "bar_chart" if any(cnt.shape[0] > 4 for cnt in contours) else "line_plot"
        
#         return {"chart_type": chart_type, "contours_found": len(contours)}

#     def process_pdf(self, pdf_path: str) -> List[Dict]:
#         """Orchestrate text + image extraction with fallback logic."""
#         # Step 1: Try fast text extraction first
#         text_chunks = self.extract_text_with_pymupdf(pdf_path)
        
#         # Step 2: Fallback to OCR if no text found
#         if not text_chunks:
#             text_chunks = self.extract_images_with_easyocr(pdf_path)
        
#         # Step 3: Optional chart detection (for images)
#         for chunk in text_chunks:
#             if chunk["type"] == "ocr_text":
#                 chart_info = self.detect_charts(f"temp_page_{chunk['page']}.png")
#                 chunk.update(chart_info)
        
#         return text_chunks

# # Example Usage
# if __name__ == "__main__":
#     extractor = PDFExtractor()
#     data = extractor.process_pdf("data/document1.pdf")
    
#     # Save to JSON
#     with open("output/extracted_data.json", "w") as f:
#         json.dump(data, f, indent=2)
    
#     print(f"Extracted {len(data)} chunks. Saved to 'output/extracted_data.json'.")


import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List, Dict
import json
import os

class PDFExtractor:
    def __init__(self):
        # Initialize EasyOCR once (lazy-load to save memory)
        self.reader = None  
        
    def _init_easyocr(self):
        if self.reader is None:
            self.reader = easyocr.Reader(['en'])  # Add more languages if needed

    def extract_text_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract selectable text from PDF using PyMuPDF (fast)."""
        text_chunks = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                # Extract regular text
                text = page.get_text("text")
                if text.strip():
                    text_chunks.append({
                        "content": text,
                        "page": page_num + 1,
                        "type": "text",
                        "source": pdf_path
                    })
                
                # Extract tables (simplified approach)
                tables = page.find_tables()
                if tables:
                    for table_num, table in enumerate(tables):
                        text_chunks.append({
                            "content": str(table.extract()),  # Convert table to string
                            "page": page_num + 1,
                            "type": "table",
                            "source": pdf_path
                        })
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
        return text_chunks

    def extract_images_with_easyocr(self, pdf_path: str) -> List[Dict]:
        """Extract text from images/scanned PDFs using EasyOCR with chart detection."""
        self._init_easyocr()
        ocr_chunks = []
        try:
            images = convert_from_path(pdf_path)
            for page_num, img in enumerate(images):
                # Convert PIL image to OpenCV format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Perform OCR
                results = self.reader.readtext(img_cv, paragraph=True)
                
                # Detect chart type
                chart_info = self.detect_charts(img_cv)
                
                for (bbox, text, confidence) in results:
                    ocr_chunks.append({
                        "content": text,
                        "page": page_num + 1,
                        "type": "ocr_text",
                        "source": pdf_path,
                        "confidence": float(confidence),
                        **chart_info
                    })
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {str(e)}")
        return ocr_chunks

    def detect_charts(self, img_cv: np.ndarray) -> Dict:
        """Improved chart detection using OpenCV."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (for axes)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # Detect rectangles (for bars)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)) == 4]
        
        chart_type = "unknown"
        if lines is not None and len(lines) >= 2:  # At least 2 axes lines
            chart_type = "bar_chart" if len(rectangles) >= 3 else "line_plot"
        
        return {"chart_type": chart_type}

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Orchestrate text + image extraction with fallback logic."""
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return []
        
        # Step 1: Try fast text extraction first
        text_chunks = self.extract_text_with_pymupdf(pdf_path)
        
        # Step 2: Fallback to OCR if no text found
        if not text_chunks:
            text_chunks = self.extract_images_with_easyocr(pdf_path)
        
        return text_chunks

# Example Usage
if __name__ == "__main__":
    extractor = PDFExtractor()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Process PDF and save results
    data = extractor.process_pdf("data/document1.pdf")
    with open("output/extracted_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Extracted {len(data)} chunks. Saved to 'output/extracted_data.json'.")