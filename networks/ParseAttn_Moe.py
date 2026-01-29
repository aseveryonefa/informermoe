import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

# ========================== 第一步：实现MoE核心组件（复用VAMoE逻辑） ==========================

class Mlp(nn.Module):
    """
    MoE中的专家网络（Expert）。
    本质上是一个基础的前馈神经网络（Feed-Forward Network），包含两层线性层和激活函数。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SparseDispatcher:
    """
    稀疏调度器：负责将输入数据“分发”给对应的专家，并将专家的输出“组合”回来。
    """
    def __init__(self, num_experts, gates):
        self.num_experts = num_experts
        self.gates = gates  # shape: [batch_size * seq_len, num_experts]
        
        # 找出哪些门控值是非零的（即被激活的专家）
        self.nonzero_gates = gates[gates > 0]
        # 获取非零门控对应的 样本索引 和 专家索引
        self.batch_index, self.expert_index = torch.nonzero(gates, as_tuple=True)

        # 统计每个专家需要处理多少个样本
        self.part_sizes = torch.zeros(num_experts, dtype=torch.long, device=gates.device)
        self.part_sizes.scatter_add_(0, self.expert_index, torch.ones_like(self.expert_index))

        self._batch_index = self.batch_index
        self._expert_index = self.expert_index

    def dispatch(self, inp):
        """
        将输入 inp 按照 expert_index 重新排序并切分，生成一个列表，
        列表第 i 项就是第 i 个专家需要处理的数据。
        """
        inp_expanded = inp[self.batch_index]
        expert_inputs = []
        start_idx = 0
        for i in range(self.num_experts):
            end_idx = start_idx + self.part_sizes[i]
            expert_inputs.append(inp_expanded[start_idx:end_idx])
            start_idx = end_idx
        return expert_inputs

    def combine(self, expert_outputs, multiply_by_gates=True):
        """
        将专家输出列表 [Expert_0_Out, Expert_1_Out, ...] 组合回原始的 [batch_size, out_features] 形状。
        """
        batch_size = self.gates.shape[0]
        out_features = expert_outputs[0].shape[-1] if len(expert_outputs) > 0 else 0
        device = self.gates.device

        # 初始化输出容器
        output = torch.zeros((batch_size, out_features), device=device)
        # 拼接所有专家的输出
        expert_outputs_cat = torch.cat(expert_outputs, dim=0)

        # 如果需要，乘上门控权重（加权求和）
        if multiply_by_gates:
            expert_outputs_cat = expert_outputs_cat * self.nonzero_gates.unsqueeze(1)
        
        # 将结果加回对应的原位置
        output.index_add_(0, self._batch_index, expert_outputs_cat)
        return output

class MoE(nn.Module):
    """
    融合到Informer中的MoE层。
    包含：
    1. 多个专家网络 (Experts)
    2. 门控网络 (Gating Network)：决定样本去往哪个专家
    3. 负载均衡损失 (Load Balancing Loss)
    """
    def __init__(self, d_model, num_experts, k=2, hidden_dim=None, drop=0., noisy_gating=True):
        super().__init__()
        self.d_model = d_model  # 输入维度
        self.num_experts = num_experts
        self.k = k  # Top-K: 每个样本激活 k 个专家
        self.noisy_gating = noisy_gating # 是否使用噪声门控（增加训练稳定性）
        self.hidden_dim = hidden_dim or d_model * 4  # 专家内部FFN的隐藏层维度

        # 1. 初始化专家列表
        self.experts = nn.ModuleList([
            Mlp(in_features=d_model, hidden_features=self.hidden_dim, out_features=d_model, drop=drop)
            for _ in range(num_experts)
        ])

        # 2. 初始化门控参数
        self.w_gate = nn.Parameter(torch.randn(d_model, num_experts) / math.sqrt(d_model))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        
        if self.noisy_gating:
            self.w_noise = nn.Parameter(torch.randn(d_model, num_experts) / math.sqrt(d_model))
        self.normal = Normal(0, 1) 

    def cv_squared(self, x):
        """计算变异系数的平方，用于负载均衡损失。衡量专家负载的方差。"""
        ###将x的形式转为为.float符合pytorch的mean方法规则！！！！！！！！
        x = x.float()
        if x.numel() <= 1:
            return torch.tensor(0., device=x.device)
        mean = x.mean()
        var = x.var()
        return var / (mean ** 2 + 1e-6) 

    def _gates_to_load(self, gates):
        """计算每个专家的负载（被选中的次数）。"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        计算当前值属于Top-K的概率。
        这是Noisy Gating机制的一部分，使门控选择可微分。
        """
        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1)  # m = k+1
        top_values_flat = noisy_top_values.flatten()

        # 阈值计算逻辑（判断是否大于第k个值）
        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(0, 1)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / (noise_stddev + 1e-6))
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / (noise_stddev + 1e-6))

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x):
        """
        输入: [batch_size, seq_len, d_model]
        输出: [batch_size, seq_len, d_model], load_balancing_loss
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 展平 Batch 和 Seq_len 维度 -> [batch_size*seq_len, d_model]
        # MoE是Token级别的，每个时间步的数据都独立选择专家
        x_flat = x.reshape(-1, d_model)
        batch_flat_size = x_flat.shape[0]

        # 2. 计算门控值 (Gating Logits)
        clean_logits = x_flat @ self.w_gate
        if self.noisy_gating and self.training:
            # 训练时添加噪声
            noise_logits = x_flat @ self.w_noise
            noise_stddev = self.softplus(noise_logits)
            noise = self.normal.sample(clean_logits.shape).to(clean_logits.device) * noise_stddev
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
            noise_stddev = None

        # 3. Top-K 路由选择
        # 选取前 k+1 个是为了辅助计算概率，实际只用前 k 个
        top_k_1_logits, top_k_1_indices = torch.topk(noisy_logits, min(self.k + 1, self.num_experts), dim=1)
        
        top_k_logits = top_k_1_logits[:, :self.k]
        top_k_indices = top_k_1_indices[:, :self.k]
        top_k_weights = self.softmax(top_k_logits)

        # 4. 生成稀疏门控矩阵 [batch_flat_size, num_experts]
        gates = torch.zeros(batch_flat_size, self.num_experts, device=x.device)
        gates.scatter_(1, top_k_indices, top_k_weights)

        # 5. 计算负载均衡损失 (Load Balancing Loss)
        # 目的是防止模型只使用某几个专家，导致部分专家过劳，部分专家空闲
        if self.noisy_gating and self.training:
            prob_in_top_k = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_k_1_logits)
            load = prob_in_top_k.sum(0)
        else:
            load = self._gates_to_load(gates)
        
        ideal_load = torch.tensor(batch_flat_size * self.k / self.num_experts, device=load.device)
        load_balancing_loss = self.cv_squared(load) + self.cv_squared(ideal_load)

        # 6. 调度与计算
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat) # 分发数据
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)] # 专家计算
        x_flat_out = dispatcher.combine(expert_outputs, multiply_by_gates=True) # 组合结果

        # 7. 恢复原始形状
        x_out = x_flat_out.reshape(batch_size, seq_len, d_model)

        return x_out, load_balancing_loss

# ========================== 第二步：实现Informer核心组件（注意力机制） ==========================

class ProbSparseSelfAttention(nn.Module):
    """
    Informer的核心：概率稀疏自注意力（简化版实现）。
    标准Attention是 O(L^2)，Informer通过选择 Top-u 的Query来降低复杂度。
    注：此处代码为通用Self-Attention实现，保留了Informer的接口。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # 线性投影 + 分头 [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        q = self.w_q(q).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 + 拼接多头
        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.d_model)
        out = self.w_o(out)
        return out

class InformerEncoderLayer(nn.Module):
    """
    Informer编码器层：
    关键修改：将传统的 Feed Forward Network (FFN) 替换为了 MoE Layer。
    """
    def __init__(self, d_model, n_heads, num_experts, k=2, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # === 核心修改点：使用MoE替代FFN ===
        self.moe = MoE(d_model, num_experts, k, hidden_dim, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Attention + Residual + Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. MoE + Residual + Norm
        moe_out, load_loss = self.moe(x)
        x = self.norm2(x + self.dropout2(moe_out))

        return x, load_loss

class InformerDecoderLayer(nn.Module):
    """
    Informer解码器层：
    同样将 FFN 替换为了 MoE Layer。
    """
    def __init__(self, d_model, n_heads, num_experts, k=2, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # === 核心修改点：使用MoE替代FFN ===
        self.moe = MoE(d_model, num_experts, k, hidden_dim, dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # 1. Self Attention (Masked)
        attn1_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1_out))

        # 2. Cross Attention (关注Encoder输出)
        attn2_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn2_out))

        # 3. MoE Layer
        moe_out, load_loss = self.moe(x)
        x = self.norm3(x + self.dropout3(moe_out))

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
                 moe_k=2,
                 hidden_dim=2048,
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
            InformerEncoderLayer(d_model, n_heads, num_experts, moe_k, hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 堆叠 Decoder Layers
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, num_experts, moe_k, hidden_dim, dropout)
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
            enc_x, load_loss = enc_layer(enc_x)
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