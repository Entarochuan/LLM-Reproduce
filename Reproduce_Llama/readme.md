## Reproduce_Llama

### V1.0 
预期开发目标 : 
    复现llama架构,实现训练、推理架构。1B左右规模，基于本地调试机(2卡A800)可以运行。
    暂时不考虑并行、优化等实现。
    暂时未实现kv-cache推理加速。

启动命令 : 

torchrun --nproc_per_node=2 --standalone train.py