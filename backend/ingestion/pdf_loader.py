import pymupdf
from pathlib import Path


def load_pdf(pdf_path: str):
    """
    Extract text from a PDF while preserving page information.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append({
                "page": page_number,
                "text": text,
                "source": pdf_path.name
            })

    return pages


if __name__ == "__main__":

    pdf_path = input("Enter PDF path: ").strip()

    pages = load_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("PDF EXTRACTION RESULT")
    print("=" * 60)

    print(f"Total pages with text: {len(pages)}")

    for page in pages[:3]:

        print("\n" + "-" * 60)
        print(f"Page: {page['page']}")
        print("-" * 60)

        print(page["text"][:1000])