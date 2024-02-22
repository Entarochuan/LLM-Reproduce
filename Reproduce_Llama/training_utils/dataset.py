"""

    dataset : 初版dataset实现，暂时不考虑streaming dataset
"""

import json
import torch
from torch.utils.data import IterableDataset
from typing import Tuple, List, Iterable

class LMdataset(IterableDataset):
    def __init__(self,
                 file_path: str,
                 max_token_length: int = 1024,
                 tokenizer: object = None):
        super().__init__()
        self.file_path = file_path
        self.max_token_length = max_token_length
        self.tokenizer = tokenizer
        
    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                content = data.get('content', '')
                tokens = self.tokenizer.encode(content)

                # 使用滑动窗口大小为1来生成input和target对
                for i in range(0, len(tokens) - 1, self.max_token_length):  # 不包括最后一个token
                    input_tokens = tokens[i:i + self.max_token_length]
                    target_tokens = tokens[i+1 : i + self.max_token_length+1]  # 下一个token作为目标

                    if len(input_tokens) > self.max_token_length:
                        input_tokens = input_tokens[:self.max_token_length]
                    if len(target_tokens) > self.max_token_length:
                        target_tokens = target_tokens[:self.max_token_length]

                    padded_input_tokens = self.pad(input_tokens)
                    padded_target_tokens = self.pad(target_tokens)
                    
                    yield padded_input_tokens, padded_target_tokens # dataset部分不实现shift，在loss内实现。
                    
    def pad(self, tokens) : 
        
        padding_length = self.max_token_length - len(tokens)
        padded_input_tokens = tokens + [self.tokenizer.pad_id()] * padding_length
        padded_input_tokens = torch.tensor(padded_input_tokens)
        
        return padded_input_tokens

        
if __name__ == "__main__": 
    
    from sentencepiece import SentencePieceProcessor
    model_path = '/fs-computility/llm/shared/mayichuan/base_models/internLM2/7B_boost/tokenizer.model'
    # 加载SentencePiece模型
    sp = SentencePieceProcessor()
    sp.load(model_path)
    
    dataset = LMdataset(file_path='/fs-computility/llm/shared/mayichuan/pjlab_projects/datasets/pretrain/arxiv/arxiv-subset/part-000139-79e2882f.jsonl', tokenizer=sp)
    
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    for i, batch in enumerate(data_loader):
        # 假设batch是一个包含填充后的tokens和张量的元组
        inputs = batch
        # 将PyTorch张量转换为Python列表，并确保数据类型为int
        tokens_list = inputs[0][0].tolist()
        print(len(tokens_list))
        # 将整数列表传递给DecodePieces进行解码
        decoded_text = sp.DecodePieces([int(token) for token in tokens_list if token != sp.pad_id()])
        print(decoded_text)
        if i == 100:
            break
