# Retrieval-Augmented Generation (RAG) — Company Regulations Assistant

**Author:** Sayed Alireza Shafiey Moghaddas

A simple, end-to-end example of Retrieval-Augmented Generation: connecting an LLM to a
private knowledge base so it can answer questions grounded in your own documents instead
of relying on the model's general training data.

## The idea

Imagine a company with hundreds of internal regulations — HR rules, benefits, leave
policies, dress code, and so on. Employees often need to look these up quickly, but
searching long PDFs or wiki pages is slow. This project builds a small RAG system that
connects a language model to the company's internal regulation database, retrieves the
most relevant policy sections for a question, and generates a clear, conversational
answer — like an intelligent HR assistant.

## How it works

1. **Data** — 100 synthetic company regulations (`data/company_regulations.jsonl`), each
   with an id, topic, and content, covering things like working hours, leave, security,
   conduct, and procurement.
2. **Embedding** — each regulation's text is embedded with OpenAI's
   `text-embedding-3-large` model.
3. **Vector store** — embeddings are indexed in a persistent [Chroma](https://www.trychroma.com/)
   collection.
4. **Retrieval + generation** — a LangChain `RetrievalQA` chain retrieves the top-3 most
   relevant regulations for a user's question and feeds them, along with a custom HR
   assistant prompt, to `gpt-4o-mini` to produce the final answer. If nothing relevant is
   found, the assistant says so instead of guessing.
5. **Tracing** — [LangSmith](https://www.langchain.com/langsmith) tracing is wired in
   (optional) so retrieval and generation steps can be inspected.

## Files

| File | Description |
|------|-------------|
| `rag_company_regulations.ipynb` | The full pipeline: install deps, load data, embed, index, build the RAG chain, and run example queries. |
| `data/company_regulations.jsonl` | The 100 synthetic company regulations used as the knowledge base (one JSON object per line: `id`, `topic`, `content`). |

## Requirements

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) (for embeddings and chat completions)
- Optionally, a [LangSmith API key](https://smith.langchain.com/) for tracing

```bash
pip install -r requirements.txt
```

## Running

The notebook was written for **Google Colab** (it mounts Google Drive to persist the
Chroma vector store and downloads the dataset from this repo's `data/` folder). Open it
directly via the "Open in Colab" badge at the top of the notebook, or run it locally:

```bash
jupyter notebook rag_company_regulations.ipynb
```

When run locally, replace the `google.colab.drive.mount(...)` calls with a local
directory for `BASE_DIR`. You'll be prompted for your OpenAI (and optionally LangSmith)
API key at runtime — keys are entered interactively via `getpass` and are never written
to disk or committed to this repo.

## Example

```
Question: What is the company's policy on remote work or working from home?

Answer: The company allows remote work only with manager approval. Eligibility for
remote work is based on job function, performance, and business needs. Additionally,
remote schedules must be agreed upon in writing.
```

```
Question: How many days of annual leave are employees entitled to each year?

Answer: I'm not sure about this policy. Please check with the HR department.
```

The second example shows the system correctly declining to answer when the specific
detail (exact number of leave days) isn't present in the retrieved regulations, rather
than making one up.

## Techniques used

RAG · OpenAI embeddings (`text-embedding-3-large`) · ChromaDB vector store ·
LangChain (`RetrievalQA`, prompt templates) · `gpt-4o-mini` · LangSmith tracing

## License

MIT — see [LICENSE](LICENSE).
