"""

    train.py
    训练代码。
"""

from models import Transformer, multi_head_attention, model_utils
from training_utils.loss import LMLoss
from training_utils.dataset import LMdataset

from datetime import datetime
import os, json
import torch
import math
from torch import nn
from torch import optim
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam

model_args = Transformer.ModelArgs(
    hidden_size = 1024,
    n_layers = 16,
    n_heads = 16,
    vocab_size = 32000,  # 使用mistral tokenizer
    multiple_of = 128,  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps = 1e-5,
    use_RMS = True,
    use_RoPE = True,
    max_batch_size = 32,
    max_seq_len = 2048,
)

model = Transformer.Transformer(model_args)
model.to('cuda')

# print(f"After moving to CUDA: {next(iter(model.tok_embeddings.parameters())).device}")
# print(model.device)

print('-----model initialized-----')
tokenizer_path = '/fs-computility/llm/shared/mayichuan/base_models/mistral/tokenizer.model'
sp = SentencePieceProcessor()
sp.load(tokenizer_path)
criterion = LMLoss(pad_idx=sp.pad_id())

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),  # adam_beta1, adam_beta2
    eps=1e-8,  # 对应于adam_eps
)
scheduler = StepLR(optimizer, step_size=100, gamma=0.95)  # 每step_size个epoch后将学习率乘以gamma

dataset = LMdataset(file_path='/fs-computility/llm/shared/mayichuan/LLMs/Reproduce_Llama/datasets/cn-weixin/train-000199-19f1f779.jsonl', tokenizer=sp)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 创建SummaryWriter实例以记录训练过程
# writer = SummaryWriter('runs/experiment')
# 训练循环
num_epochs = 1
max_steps = 10000 
# scaler = GradScaler()
for epoch in range(num_epochs):
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # 前向传播
        # print(inputs)
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        # with autocast():
        #     # 前向传播
        #     model_outputs = model(inputs)
        #     loss = criterion(model_outputs, targets)

        # 梯度缩放与反向传播
        # scaler.scale(loss).backward()
        
        # # 更新权重前检查梯度是否溢出，如果没溢出则应用更新
        # if scaler.get_scale() > 0:
        #     scaler.step(optimizer)
        #     scaler.update()

        model_outputs = model(inputs)  # 输出是对应每个时间步的token概率分布
        loss = criterion(model_outputs, targets)
        # loss = criterion(model_outputs[:, -1, :], targets)  # 只计算最后一个时间步的损失
        
        # 反向传播及参数更新
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  

        if (batch_idx+1) % 10 == 0 : 
            batch_tokens = inputs.size(0) * inputs.size(1)
            batch_loss = loss.item() * batch_tokens
            avg_loss = batch_loss / batch_tokens
            batch_ppl = math.exp(avg_loss)
            print(f'Step : {batch_idx+1}, Loss : {loss.item()}, batch ppl = {batch_ppl}')
        
        if (batch_idx+1) % 100 == 0 : 
            # 应用 softmax 转换为概率分布
            # with autocast():
            logits = model(inputs)
            probs = nn.functional.softmax(logits, dim=-1)
            # 获取最高概率的 token ID
            predicted_ids = torch.argmax(probs, dim=-1)
            # 假设您已经有了 SentencePiece tokenizer
            predicted_tokens = sp.decode(predicted_ids[0][:50].tolist())
            print(predicted_tokens)
        
        if (batch_idx+1) == max_steps : 
            break
                        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 保存模型
# 获取当前日期和时间
current_datetime = datetime.now()
# 生成month-day hour:minute:second字符串
save_dir = f'ckpts/{current_datetime.strftime("%m-%d")}'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'model.pth')
config_path = os.path.join(save_dir, 'params.json')
model_args_dict = model_args.__dict__
json_str = json.dumps(model_args_dict)
with open(config_path, 'w') as f:
    f.write(json_str)
    
torch.save(model.state_dict(), model_path)
if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
    torch.save(model.module.state_dict(), model_path)
    
# 关闭SummaryWriter
# writer.close()


