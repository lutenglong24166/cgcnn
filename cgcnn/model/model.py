from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Inputs:
        atom_feature:   [N, F]
        edge_feature:   [E, B]
        edge_index:     [E, 2]

    Output:
        out_feature:    [N, F]
    """

    def __init__(self, atom_feature_len: int, edge_feature_len: int):
        super().__init__()
        self.atom_feature_len = atom_feature_len
        self.edge_feature_len = edge_feature_len

        # (2F + B) -> (2F)
        self.fc_full = nn.Linear(
            2 * atom_feature_len + edge_feature_len, 2 * atom_feature_len
        )
        self.bn_msg = nn.BatchNorm1d(2 * atom_feature_len)
        self.bn_agg = nn.BatchNorm1d(atom_feature_len)

        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

    def _parse_edge_index(self, edge_index: torch.Tensor):
        # src: E
        # dst: E
        src, dst = edge_index[:, 0], edge_index[:, 1]
        return src, dst

    # 1) Message step
    # z_ij = [v_i || v_j || u_ij]
    # m_ij = sigma(W_f z_ij + b_f) ⊙ Softplus(W_s z_ij + b_s)
    def build_message(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        src, dst = self._parse_edge_index(edge_index)

        # center atom feature
        # (E, F)
        v_i = atom_feature[src]

        # neighbor atom feature
        # (E, F)
        v_j = atom_feature[dst]

        # (E, 2F + B)
        z_ij = torch.cat([v_i, v_j, edge_feature], dim=1)

        # (E, 2F)
        gated = self.fc_full(z_ij)
        gated = self.bn_msg(gated)

        # split feature to two branches
        # filter gated f_ij, (E, F)
        # core signal s_ij, (E, F)
        f_ij, s_ij = gated.chunk(2, dim=1)

        # filter gate branch
        f_ij = self.sigmoid(f_ij)

        # core signal branch
        s_ij = self.softplus1(s_ij)

        # use element-wise product to build message
        # messgae m_ij, (E, F)
        m_ij = f_ij * s_ij

        return m_ij, src

    # 2) Aggregate step
    # m_i = sum_{j in N(i)} m_ij
    def aggregate(self, message: torch.Tensor, src: torch.Tensor, num_nodes: int):
        m_i = torch.zeros(
            num_nodes, message.size(1), dtype=message.dtype, device=message.device
        )

        m_i.index_add_(0, src, message)
        # (N, F)
        m_i = self.bn_agg(m_i)
        return m_i

    # 3) Update step
    # v_i^{t+1} = Softplus(v_i^{t} + BN(m_i))
    def update(self, atom_feature: torch.Tensor, agg: torch.Tensor):
        return self.softplus2(atom_feature + agg)

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        message, src = self.build_message(atom_feature, edge_feature, edge_index)
        agg = self.aggregate(message, src, atom_feature.size(0))
        out = self.update(atom_feature, agg)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Inputs:
        atom_feature:     [N] (0-based atomic-number indices)
        edge_feature:     [E, B]
        edge_index:       [E, 2]
        crystal_atom_idx: list[LongTensor], len = batch_size
                          each tensor contains atom indices for one crystal
    Output:
        regression:       [batch_size, 1]
        classification:   [batch_size, 2] (log-prob)
    """

    def __init__(
        self,
        edge_feature_len: int,
        max_num_elements: int = 100,
        atom_feature_len: int = 64,
        n_conv: int = 3,
        h_feature_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification

        # atom index embedding: [0, max_num_elements) -> F
        self.embedding = nn.Embedding(max_num_elements, atom_feature_len)

        # convolution layer stack
        self.convs = nn.ModuleList(
            [ConvLayer(atom_feature_len, edge_feature_len) for _ in range(n_conv)]
        )

        # crystal-level MLP head
        self.conv_to_fc = nn.Linear(atom_feature_len, h_feature_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_feature_len, h_feature_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_feature_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_feature_len, 1)

    def readout(self, atom_feature: torch.Tensor, crystal_atom_idx: list[torch.Tensor]):
        # mean-pooling atom features into crystal features
        pooled = []
        for idx_map in crystal_atom_idx:
            idx_map = idx_map.to(atom_feature.device)
            pooled.append(
                atom_feature.index_select(0, idx_map).mean(dim=0, keepdim=True)
            )

        # (batch_size, F)
        pooling = torch.cat(pooled, dim=0)
        return pooling

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
        crystal_atom_idx: list[torch.Tensor],
    ):
        # 1) node embedding
        atom_feature = self.embedding(atom_feature.long())

        # 2) graph convolution
        for conv in self.convs:
            atom_feature = conv(atom_feature, edge_feature, edge_index)

        # 3) readout pooling
        crystal_feature = self.readout(atom_feature, crystal_atom_idx)

        # 4) prediction head
        crystal_feature = self.conv_to_fc_softplus(self.conv_to_fc(crystal_feature))

        if self.classification:
            crystal_feature = self.dropout(crystal_feature)

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, sp in zip(self.fcs, self.softpluses):
                crystal_feature = sp(fc(crystal_feature))

        # regression: (batch_size, 1)
        out = self.fc_out(crystal_feature)

        if self.classification:
            out = self.logsoftmax(out)

        return out
