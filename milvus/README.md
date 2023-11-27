# Question Answering Chatbot with Milvus, Towhee, and Gradio

This Python script enables a question-answering system functionality using Milvus for storage and Towhee for processing. It creates a collection in Milvus, loads embeddings, and performs question-based searches, serving it via web interface by using Gradio.

## Prerequisites

- Python 3.x
- Required Python packages (`pymilvus`, `towhee`, `pandas`, `gradio`)
- Milvus database

## Setup

1. **Install Dependencies:**

   ```bash
   pip install pymilvus towhee pandas gradio
   ```
2. **Environment Variables:**

Set the following environment variables with your Milvus Proxy settings:

```bash
export MILVUS_HOST: <HOST>
export MILVUS_PORT: <PORT>
```

3. **Dataset:**

Prepare the [_question_answer.csv_](https://github.com/towhee-io/examples/releases/download/data/question_answer.csv) file containing the columns: _id_, _question_, and _answer_.

```bash
curl -L https://github.com/towhee-io/examples/releases/download/data/question_answer.csv -O
```

## Execution
Run the script using:

```bash
python reverse_image_search.py
```

## Usage
The script performs the following actions:

1. Create Milvus Collection:

 * Reads the _question_answer.csv_ file.
 * Creates a collection in Milvus for question-answer pairs with embeddings.

2. Embeddings Load:

 * Embeds questions and inserts them into the Milvus collection.

3. Question Query:

* Accepts a question and retrieves the closest matching answer from Milvus.

4. Showcase Interface:

* Launches a chatbot interface to interactively query the system.

