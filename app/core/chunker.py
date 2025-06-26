from typing import List
import re

class Chunker:
    def __init__(self, file_type:str, max_tokens:int, overlap:int=50):
        """
        :param file_type: 'text' or 'code'
        :param max_tokens: target token length per chunk (tier-based)
        :param overlap: number of tokens to overlap between chunks
        """
        self.file_type = file_type
        self.max_tokens = max_tokens
        self.overlap = overlap
        
    def chunk(self,content:str) -> List[str]:
        if self.file_type == "code" :
            return self.chunk_code(content)
        else:
            return self.chunk_text(content)
        
    def chunk_text(self,text:str) -> List[str]:
        """
        Sentence-based chunking for plain text / PDF-extracted content.
        Groups sentences into token-controlled windows with overlap.
        """
        sentences = re.split(r'(?<=[.!?])\s+',text.strip())
        sentence_tokens = [(s,self.estimate_tokens(s)) for s in sentences if s]
        print(f"Split into {len(sentence_tokens)} sentences")
        chunks = []
        current_chunk = []
        current_tokens = []
        i=0
        while i< len(sentence_tokens):
            sentence,tokens = sentence_tokens[i]
            
            if current_tokens + tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    
                    #backtrack to create overlap winndow
                    overlap_tokens = 0
                    j = len(current_chunk) - 1
                    while j>=0 and overlap_tokens<self.overlap : 
                        overlap_tokens += self.estimate_tokens(current_chunk[j])
                        j -= 1
                    current_chunk = current_chunk[j + 1:]
                    current_tokens = sum(self.estimate_tokens(s) for s in current_chunk)
                else:
                    chunks.append(sentence)
                    i+=1
                    continue
            current_chunk.append(sentence)
            current_tokens += tokens
            i +=1
        
        if current_tokens:
            chunks.append(" ".join(current_chunk))
            
        return chunks
        
    def chunk_code(self, code: str) -> List[str]:
        """
        Split code into chunks based on function and class definitions,
        while respecting max_tokens and overlap.
        """
        # Split at class or def/function declarations
        blocks = re.split(r'(?=^\s*(def|class)\s)', code, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = []
        current_tokens = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            block_tokens = self.estimate_tokens(block)

            if current_tokens + block_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                    # overlap logic
                    overlap_tokens = 0
                    j = len(current_chunk) - 1
                    while j >= 0 and overlap_tokens < self.overlap:
                        overlap_tokens += self.estimate_tokens(current_chunk[j])
                        j -= 1
                    current_chunk = current_chunk[j + 1:]
                    current_tokens = sum(self.estimate_tokens(b) for b in current_chunk)
                else:
                    chunks.append(block)
                    continue

            current_chunk.append(block)
            current_tokens += block_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    
    def estimate_tokens(self,text:str) -> int:
        return max(1,len(text)//4)