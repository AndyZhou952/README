VeOmni 是面向大规模预训练/SFT 的框架，采用 Trainer-free 的线性脚本设计，基于 PyTorch 原生分布式，支持多模态模型和多种并行策略，训练逻辑透明可控。
MM-GRPO 是面向扩散模型强化学习训练的库，基于 Ray 分布式框架，封装了 Flow-GRPO 等 RL 算法，支持异步训练策略，专注于 RLHF/RL 训练场景。
两者互补：VeOmni 负责预训练/SFT，MM-GRPO 负责后续的 RL 对齐，可组合使用形成完整的训练流程。

<img width="1479" height="1361" alt="image" src="https://github.com/user-attachments/assets/706885e8-594b-4db4-b57a-9f664f2836d3" />
<img width="1359" height="1734" alt="image" src="https://github.com/user-attachments/assets/99bf5e98-2a9b-42d5-8c12-ea38b293067c" />
<img width="1308" height="1458" alt="image" src="https://github.com/user-attachments/assets/c4227242-4f3f-484e-9e72-9e3d4eba3255" />
<img width="1404" height="1688" alt="image" src="https://github.com/user-attachments/assets/847329a5-3a0f-40cf-a1ea-1dbef9589bd9" />
<img width="1302" height="1347" alt="image" src="https://github.com/user-attachments/assets/938309fc-2d4a-46a0-8e6d-c23ff1024b6f" />
<img width="1446" height="1736" alt="image" src="https://github.com/user-attachments/assets/a411d5cc-bc69-4cee-a23a-b771451d0f60" />
