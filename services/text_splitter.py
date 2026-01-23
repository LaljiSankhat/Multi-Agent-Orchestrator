from services.content import contents
from langchain_text_splitters import RecursiveCharacterTextSplitter




def split_text_into_chunks(contents):
    # 1. Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Groq's limit is roughly tokens, but this splitter counts CHARACTERS.
        # 1 token is ~4 characters. So 4000 tokens ≈ 16000 characters.
        chunk_size=10000, 
        chunk_overlap=1000, # Overlap ensures context isn't cut in half
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Order of priority for splitting
    )

    # 2. Create the chunks
    # 'result' is your 25k token string
    chunks = text_splitter.split_text(" ".join(contents))
    return chunks




# chunks = split_text_into_chunks(contents)
# print(chunks)
# print("\n \n")

# print(f"Total chunks created: {len(chunks)}")