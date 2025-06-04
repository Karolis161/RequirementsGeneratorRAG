# RAG-Based Requirement Generation and Evaluation

This project evaluates how Retrieval-Augmented Generation (RAG) affects the quality of automatically generated software requirements using GPT-based language models.

To run the system:

1. Clone the repository:
```bash
git clone https://github.com/Karolis161/RequirementsGeneratorRAG.git
cd your-repo-name
```
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key by creating a .env file:
```bash
OPENAI_API_KEY=your-openai-api-key
```

4. Prepare documents by placing them in the following folders (create them if needed):
```bash
documents/pdf/
documents/docx/
documents/pptx/
```

5. Adjust parameters by editing config.json to match your setup.

6. To extract and store embeddings from documents run:
```bash
python embedder.py
```
7. To generate requirements for a query run:
```bash
python retrieval.py
```

8. To compare RAG vs No-RAG results visually run:
```bash
python visualizer.py
```