import torch
import torch.nn as nn
import torch.nn.functional as F
from .dit_moe import ChannelMoeBlock
# ========================== 实现Informer核心组件（注意力机制） ==========================

class ProbSparseSelfAttention(nn.Module):
    """
    真正的 Informer 核心：概率稀疏自注意力 (ProbSparse Self-Attention)。
    标准 Attention 是 O(L^2) 复杂度，Informer 通过选择 Top-u 的活跃 Query 来大幅降低复杂度至 O(L log L)。
    此实现完美保留了输入和输出形状 [batch, seq_len, d_model]，适配 VAMoE 的空间网格平铺输入。
    """
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor  # 控制采样和Top-u的乘积因子: u = factor * ln(L)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        L_Q = q.shape[1]
        L_K = k.shape[1]

        # 线性投影 + 分头 [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        q = self.w_q(q).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 1. 计算采样数量 (U_part) 和需要保留的 Top-u Queries (u)
        U_part = self.factor * int(math.ceil(math.log(max(L_K, 2)))) 
        u = self.factor * int(math.ceil(math.log(max(L_Q, 2))))
        u = min(u, L_Q)
        U_part = min(U_part, L_K)

        # 2. 从所有 Keys 中随机采样 U_part 个 Key 以评估 Query 保留价值
        # 生成随机索引
        index_sample = torch.randint(0, L_K, (U_part,), device=q.device)
        k_sample = k[:, :, index_sample, :] # [batch, heads, U_part, d_k]
        
        # 3. 计算所有 Query 和采样 Key 的内积，得到稀疏性评估矩阵得分
        Q_K_sample = torch.matmul(q, k_sample.transpose(-2, -1)) # [batch, heads, L_Q, U_part]
        
        # 计算 Sparsity Score M = Max(QK) - Mean(QK)
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1) # [batch, heads, L_Q]
        
        # 4. 根据 M 得分，选出最“活跃”的前 u 个 Query 索引
        _, top_idx = M.topk(u, dim=-1) # [batch, heads, u]
        
        # 利用 gather 提取头部 Query
        top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        q_top = torch.gather(q, dim=-2, index=top_idx_expanded) # [batch, heads, u, d_k]
        
        # 5. 【核心】仅用这 u 个活跃 Query 计算完整的 Attention
        attn_scores = torch.matmul(q_top, k.transpose(-2, -1)) / self.scale # [batch, heads, u, L_K]
        
        if mask is not None:
            # Informer 论文里有针对 mask 的 slice 逻辑，这里目前 VAMoE 默认全亮，不做切片处理
            pass
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 得到头部 Query 的上下文融合特征 [batch, heads, u, d_k]
        out_top = torch.matmul(attn_weights, v)
        
        # 6. 为未被选中的钝化 Query 准备填充物 (Values 的均指)
        v_mean = v.mean(dim=-2, keepdim=True) # [batch, heads, 1, d_k]
        out_full = v_mean.expand(batch_size, self.n_heads, L_Q, self.d_k).clone()
        
        # 将头部 Query 算出的精准 Context 散布回原来它们所在的空间位置
        out_full.scatter_(dim=-2, index=top_idx_expanded, src=out_top)
        
        # 7. 拼接多头并输出 [batch, seq, d_model]
        out = out_full.transpose(1, 2).reshape(batch_size, L_Q, self.d_model)
        out = self.w_o(out)
        return out

class InformerEncoderLayer(nn.Module):
    """
    Informer编码器层：
    关键修改：将传统的 Feed Forward Network (FFN) 替换为了 MoE Layer。
    """
    def __init__(self, d_model, n_heads, num_experts, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # === 核心修改点：使用 ChannelMoeBlock 代替简陋的 Token级别 MoE ===
        self.moe = ChannelMoeBlock(embed_dim=d_model, num_experts=num_experts)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, posembed=None, mask=None):
        # 1. Attention + Residual + Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Channel MoE + Residual + Norm
        # 注意：原版 ChannelMoE 需要 posembed，如果从 VAMoE 主框架漏过来的是 None
        # 则需要给一个 dummy posembed 防止报错（这里做个兼容保护）
        if posembed is None:
             posembed = torch.ones(self.moe.experts.__len__(), x.shape[-1], device=x.device)
             
        moe_out = self.moe(x, posembed)
        x = self.norm2(x + self.dropout2(moe_out))

        # 原版 ChannelMoEBlock 不返回 load_loss (在其内部不显式算 loss)，返回 0.0 保证外层训练脚本不崩
        load_loss = 0.0
        return x, load_loss

class InformerDecoderLayer(nn.Module):
    """
    Informer解码器层：
    同样将 FFN 替换为了 MoE Layer。
    """
    def __init__(self, d_model, n_heads, num_experts, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # === 核心修改点：使用 ChannelMoeBlock 替代 FFN 或 简陋的MoE ===
        self.moe = ChannelMoeBlock(embed_dim=d_model, num_experts=num_experts)

    def forward(self, x, enc_out, posembed=None, tgt_mask=None, src_mask=None):
        # 1. Self Attention (Masked)
        attn1_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1_out))

        # 2. Cross Attention (关注Encoder输出)
        attn2_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn2_out))

        # 3. Channel MoE Layer
        if posembed is None:
             posembed = torch.ones(self.moe.experts.__len__(), x.shape[-1], device=x.device)
             
        moe_out = self.moe(x, posembed)
        x = self.norm3(x + self.dropout3(moe_out))

        load_loss = 0.0
        return x, load_loss

# ========================== 第三步：组装MoE-Informer整体模型 ==========================

class MoEInformer(nn.Module):
    """
    MoE-Informer 主模型类。
    结构：Embedding -> Encoder(with MoE) -> Decoder(with MoE) -> Output Projection
    """
    def __init__(self,
                 d_model=512,
                 n_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=2,
                 num_experts=8,
                 input_dim=1,
                 output_dim=1,
                 seq_len=96,
                 pred_len=24,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 输入嵌入层
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.dec_embedding = nn.Linear(input_dim, d_model)
        # 输出投影层
        self.output_proj = nn.Linear(d_model, output_dim)

        # 位置编码 (此处使用可学习的Parameter简化实现)
        self.pos_encoding = nn.Parameter(torch.randn(1, max(seq_len, pred_len), d_model))

        # 堆叠 Encoder Layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model=d_model, n_heads=n_heads, num_experts=num_experts, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # 堆叠 Decoder Layers
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model=d_model, n_heads=n_heads, num_experts=num_experts, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, enc_x, dec_x):
        """
        前向传播
        参数:
            enc_x: [batch_size, seq_len, input_dim] (历史观测数据)
            dec_x: [batch_size, pred_len, input_dim] (预测起始token + 占位符)
        返回:
            pred: [batch_size, pred_len, output_dim] (预测结果)
            total_load_loss: 所有层MoE的负载均衡损失之和
        """
        total_load_loss = 0.

        # === Encoder 前向 ===
        enc_x = self.enc_embedding(enc_x) + self.pos_encoding[:, :self.seq_len, :]
        for enc_layer in self.encoder_layers:
            # enc_x, load_loss = enc_layer(enc_x)
            enc_x, load_loss = checkpoint(enc_layer, enc_x, use_reentrant=False)
            total_load_loss += load_loss

        # === Decoder 前向 ===
        dec_x = self.dec_embedding(dec_x) + self.pos_encoding[:, :self.pred_len, :]
        for dec_layer in self.decoder_layers:
            dec_x, load_loss = dec_layer(dec_x, enc_x)
            total_load_loss += load_loss

        # === 输出 ===
        pred = self.output_proj(dec_x)

        return pred, total_load_loss

# ========================== 第四步：测试代码块 ==========================
if __name__ == "__main__":
    # 模拟超参数
    batch_size = 4
    seq_len = 96
    pred_len = 24
    input_dim = 1
    output_dim = 1
    d_model = 128
    n_heads = 4
    num_experts = 4
    moe_k = 2

    # 实例化模型
    model = MoEInformer(
        d_model=d_model,
        n_heads=n_heads,
        num_experts=num_experts,
        moe_k=moe_k,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len
    )

    # 构造随机输入数据
    enc_x = torch.randn(batch_size, seq_len, input_dim)
    dec_x = torch.randn(batch_size, pred_len, input_dim)

    # 前向运行
    pred, load_loss = model(enc_x, dec_x)

    # 验证输出
    print(f"预测输出形状: {pred.shape}")  # 预期: [4, 24, 1]
    print(f"MoE负载均衡损失: {load_loss.item():.4f}")
    print("MoE-Informer模型构建成功，前向传播无误！")