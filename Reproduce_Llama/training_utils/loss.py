from torch import nn
import torch
import torch.nn.functional as F

class LMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, pad_idx, label_smoothing=0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing, ignore_index=pad_idx)

    def forward(self, logits, labels):
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)#确保标签tensor和logits的tensor在相同的设备上
        
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        
        loss = self.loss_fn(
            shift_logits, shift_labels
        )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        return loss
    

if __name__ == "__main__" : 
    
    # print('yeah')
    loss = LMLoss()
    
    # 假设我们有一个随机生成的logits张量和相应的标签张量
    logits = torch.randn(1, 128, dtype=torch.float32)  # 代表模型输出的未归一化预测值
    # labels = torch.randint(0, 128, (1, 128), dtype=torch.float32)  # 代表每个位置上的真实标签
    labels = logits
    labels = labels.reshape(1, 128)

    print(logits.shape, labels.shape)
    
    calculated_loss = loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    print(calculated_loss)