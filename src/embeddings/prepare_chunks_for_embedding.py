import os
import json
import pickle
import base64

def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode to base64 string"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return ""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

def prepare_chunks_for_embedding(extracted_json_path: str, output_pickle_path: str):
    """
    Convert extracted JSON data to pickle file with chunks list for embedding script.
    Adds base64 encoded image data to image chunks and maps 'source' to 'source_file'.

    Args:
        extracted_json_path: Path to extracted_data.json from extraction step.
        output_pickle_path: Path to save the pickle file for embedding input.
    """
    if not os.path.exists(extracted_json_path):
        print(f"Extracted JSON file not found: {extracted_json_path}")
        return

    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for chunk in data:
        # Map 'source' to 'source_file' for compatibility
        if 'source' in chunk:
            chunk['source_file'] = chunk['source']

        # For image chunks, add base64 encoded image data if image_path exists
        if chunk.get('type') in ['embedded_image', 'page_image', 'docx_image']:
            image_path = chunk.get('image_path', '')
            if image_path and os.path.exists(image_path):
                chunk['image_data'] = encode_image_to_base64(image_path)
            else:
                chunk['image_data'] = ""

    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Prepared chunks pickle saved to: {output_pickle_path}")

if __name__ == "__main__":
    extracted_json_path = "output/extracted_data.json"
    output_pickle_path = "extracted_data/chunks/all_chunks.pkl"
    prepare_chunks_for_embedding(extracted_json_path, output_pickle_path)
