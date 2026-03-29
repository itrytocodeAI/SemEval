import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers location: {transformers.__file__}")
    from transformers import AutoTokenizer, AutoModel
    print("Successfully imported AutoTokenizer and AutoModel")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
