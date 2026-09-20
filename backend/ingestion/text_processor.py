from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages):
    """
    Convert page-level PDF data into smaller text chunks
    while preserving source and page metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    documents = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk_index, chunk in enumerate(chunks):

            documents.append({
                "text": chunk,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": f"{page['page']}_{chunk_index}"
                }
            })

    return documents