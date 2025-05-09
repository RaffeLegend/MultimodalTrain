from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
import tiktoken

# Basic Tool Functions
import os
os.environ["GOOGLE_CSE_ID"] = "" # Your Google CSE ID
os.environ["GOOGLE_API_KEY"] = "" # Your Google API Key

def get_search(query:str="", k:int=1): # get the top-k resources with google
    search = GoogleSearchAPIWrapper(k=k)
    def search_results(query):
        return search.results(query, k)
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=search_results,
    )
    ref_text = tool.run(query)
    if 'Result' not in ref_text[0].keys():
        return ref_text
    else:
        return None

def get_page_content(link:str):
    loader = AsyncHtmlLoader([link])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if len(docs_transformed) > 0:
        return docs_transformed[0].page_content
    else:
        return None

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_by_sentence(text, chunk_size=2048):
    """Chunk the $text into sentences with less than 2k tokens."""
    sentences = text.split('. ')
    chunked_text = []
    curr_chunk = []
    # 逐句添加文本片段，确保每个段落都小于2k个token
    for sentence in sentences:
        if num_tokens_from_string(". ".join(curr_chunk)) + num_tokens_from_string(sentence) + 2 <= chunk_size:
            curr_chunk.append(sentence)
        else:
            chunked_text.append(". ".join(curr_chunk))
            curr_chunk = [sentence]
    # 添加最后一个片段
    if curr_chunk:
        chunked_text.append(". ".join(curr_chunk))
    return chunked_text[0]

def chunk_text_front(text, chunk_size = 2048):
    '''
    get the first `trunk_size` token of text
    '''
    chunked_text = ""
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return text
    else:
        ratio = float(chunk_size) / tokens
        char_num = int(len(text) * ratio)
        return text[:char_num]

def chunk_texts(text, chunk_size = 2048):
    '''
    trunk the text into n parts, return a list of text
    [text, text, text]
    '''
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return [text]
    else:
        texts = []
        n = int(tokens/chunk_size) + 1
        # 计算每个部分的长度
        part_length = len(text) // n
        # 如果不能整除，则最后一个部分会包含额外的字符
        extra = len(text) % n
        parts = []
        start = 0

        for i in range(n):
            # 对于前extra个部分，每个部分多分配一个字符
            end = start + part_length + (1 if i < extra else 0)
            parts.append(text[start:end])
            start = end
        return parts
     