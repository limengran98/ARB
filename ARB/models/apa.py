from collections import defaultdict
from itertools import permutations
import random
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops


def get_normalized_adjacency(edge_index, n_nodes, mode=None):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    if mode == "left":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    elif mode == "right":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = edge_weight * deg_inv_sqrt[col]
    elif mode == "article_rank":
        d = deg.mean()
        deg_inv_sqrt = (deg+d).pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    else:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def get_propagation_matrix(edge_index: Adj, n_nodes: int, mode: str = "adj") -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


def get_edge_index_from_y(y: torch.Tensor, know_mask: torch.Tensor = None) -> Adj:
    nodes = defaultdict(list)
    label_idx_iter = enumerate(y.numpy()) if know_mask is None else zip(know_mask.numpy(),y[know_mask].numpy())
    for idx, label in label_idx_iter:
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T


def get_edge_index_from_y_ratio(y: torch.Tensor, ratio: float = 1.0) -> torch.Tensor:
    n = y.size(0)
    mask = []
    nodes = defaultdict(list)
    for idx, label in random.sample(list(enumerate(y.numpy())), int(ratio*n)):
        mask.append(idx)
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T, torch.tensor(mask, dtype=torch.long)


def to_dirichlet_loss(attrs, laplacian):
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()


class APA:
    def __init__(self, edge_index: Adj, x: torch.Tensor, know_mask: torch.Tensor, is_binary: bool):
        self.edge_index = edge_index
        self.x = x
        self.n_nodes = x.size(0)
        self.know_mask = know_mask
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask] - self.mean) / self.std
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]
        self._adj = None

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    @staticmethod
    def _symmetric_normalize(A: torch.Tensor) -> torch.Tensor:
        A = A.coalesce()
        row, col = A.indices()
        val = A.values()
        deg = torch.zeros(A.size(0), device=A.device, dtype=A.dtype).index_add_(0, row, val)
        deg = torch.clamp(deg, min=1e-12)
        d_inv_sqrt = torch.pow(deg, -0.5)
        new_val = d_inv_sqrt[row] * val * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(A.indices(), new_val, size=A.shape, device=A.device, dtype=A.dtype).coalesce()

    @staticmethod
    def _ensure_symmetric(A: torch.Tensor) -> torch.Tensor:
        AT = torch.sparse_coo_tensor(A.indices().flip(0), A.values(), A.shape, device=A.device, dtype=A.dtype).coalesce()
        return (A + AT).coalesce().coalesce() * 0.5

    @staticmethod
    def _to_sparse(indices, values, size, device, dtype):
        return torch.sparse_coo_tensor(indices, values, size=size, device=device, dtype=dtype).coalesce()

    # ---------- Virtual edges: builders ----------
    def _build_virtual_adj_full(self):
        # Full mode: do not explicitly build a dense fully-connected sparse graph,
        # use mean channel instead (see umtp_ve)
        return None

    def _knn_topk(self, X: torch.Tensor, k: int):
        # Top-k cosine similarity; suitable for small/medium graphs.
        # For large graphs, consider FAISS / approximate NN
        Xn = F.normalize(X, p=2, dim=1)
        S = Xn @ Xn.T  # [N, N]
        N = Xn.size(0)
        S.fill_diagonal_(float('-inf'))
        topv, topi = torch.topk(S, k=k, dim=1)
        rows = torch.arange(N, device=X.device).unsqueeze(1).expand_as(topi).reshape(-1)
        cols = topi.reshape(-1)
        vals = torch.clamp(topv.reshape(-1), min=0)  # keep non-negative
        return rows, cols, vals, N

    def _build_virtual_adj_knn(self, X: torch.Tensor, k: int):
        rows, cols, vals, N = self._knn_topk(X, k)
        A = self._to_sparse(torch.stack([rows, cols], dim=0), vals, (N, N), X.device, X.dtype)
        A = self._ensure_symmetric(A)
        return self._symmetric_normalize(A)

    def _build_virtual_adj_thresh(self, X: torch.Tensor, thresh: float):
        Xn = F.normalize(X, p=2, dim=1)
        S = Xn @ Xn.T
        N = Xn.size(0)
        S.fill_diagonal_(0.0)
        mask = (S > thresh)
        idx = mask.nonzero(as_tuple=False).T  # [2, E]
        if idx.numel() == 0:
            return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long, device=X.device),
                                           torch.zeros((0,), dtype=X.dtype, device=X.device),
                                           size=(N, N), device=X.device, dtype=X.dtype).coalesce()
        vals = torch.clamp(S[idx[0], idx[1]], min=0)
        A = self._to_sparse(idx, vals, (N, N), X.device, X.dtype)
        A = self._ensure_symmetric(A)
        return self._symmetric_normalize(A)

    def _build_virtual_adj_rbf(self, X: torch.Tensor, k: int, sigma: float = None):
        # RBF-weighted kNN: w_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))
        rows, cols, cosv, N = self._knn_topk(X, k)  # cosv ~ similarity
        dist2 = 2 * (1 - torch.clamp(cosv, min=-1, max=1))  # ||a-b||^2 = 2(1 - cos)
        if sigma is None:
            finite = dist2[torch.isfinite(dist2) & (dist2 > 0)]
            sigma = torch.sqrt(torch.median(finite) + 1e-12).item() if finite.numel() > 0 else 1.0
        w = torch.exp(-dist2 / (2 * (sigma ** 2) + 1e-12))
        A = self._to_sparse(torch.stack([rows, cols], dim=0), w, (N, N), X.device, X.dtype)
        A = self._ensure_symmetric(A)
        return self._symmetric_normalize(A)

    def _build_virtual_adj_random(self, n: int, m_edges: int, seed: int = 0, dtype=None):
        g = torch.Generator(device=self.out.device)
        g.manual_seed(seed)
        i = torch.randint(0, n, (m_edges,), generator=g, device=self.out.device)
        j = torch.randint(0, n, (m_edges,), generator=g, device=self.out.device)
        mask = (i != j)
        i, j = i[mask], j[mask]
        idx = torch.stack([i, j], dim=0)
        val = torch.ones(i.numel(), device=self.out.device, dtype=dtype or self.out.dtype)
        A = self._to_sparse(idx, val, (n, n), self.out.device, dtype or self.out.dtype)
        A = self._ensure_symmetric(A)
        return self._symmetric_normalize(A)

    def _build_virtual_adj(self, virtual_mode: str, k: int, sim_thresh: float, rbf_sigma: float, random_edges: int):
        """
        Return the sparse adjacency of virtual edges (already symmetrically normalized);
        in full mode, return None (handled by mean channel).
        """
        X = self.x  # use input features for graph construction
        if virtual_mode == "full":
            return self._build_virtual_adj_full()
        elif virtual_mode == "knn":
            return self._build_virtual_adj_knn(X, k=k)
        elif virtual_mode == "thresh":
            return self._build_virtual_adj_thresh(X, thresh=sim_thresh)
        elif virtual_mode == "rbf":
            return self._build_virtual_adj_rbf(X, k=k, sigma=rbf_sigma)
        elif virtual_mode == "random":
            if random_edges is None:
                random_edges = max(self.n_nodes * 10, 1000)
            return self._build_virtual_adj_random(self.n_nodes, random_edges, dtype=self.x.dtype)
        elif virtual_mode in (None, "none"):
            return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long, device=self.out.device),
                                           torch.zeros((0,), dtype=self.out.dtype, device=self.out.device),
                                           size=(self.n_nodes, self.n_nodes),
                                           device=self.out.device, dtype=self.out.dtype).coalesce()
        else:
            raise ValueError(f"Unknown virtual_mode: {virtual_mode}")


    def fp(self, out: torch.Tensor = None, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def fp_analytical_solution(self, **kw) -> torch.Tensor:
        adj = self.adj.to_dense()

        assert self.know_mask.dtype == torch.int64
        know_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        know_mask[self.know_mask] = True
        unknow_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        unknow_mask[self.know_mask] = False

        A_uu = adj[unknow_mask][:, unknow_mask]
        A_uk = adj[unknow_mask][:, know_mask]

        L_uu = torch.eye(unknow_mask.sum()) - A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = self.out.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), self.out_k_init)

        return out * self.std + self.mean

    def pr(self, out: torch.Tensor = None, alpha: float = 0.85, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def ppr(self, out: torch.Tensor = None, alpha: float = 0.85, weight: torch.Tensor = None, num_iter: int = 1,
            **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        if weight is None:
            weight = self.mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * weight
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def mtp(self, out: torch.Tensor = None, beta: float = 0.85, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def mtp_analytical_solution(self, beta: float = 0.85, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        eta = (1 / beta - 1)
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

    def arb(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1,
             **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def arb2(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma: float = 0.75,
              num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma * (alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)) + (1 - gamma) * out
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def arb_analytical_solution(self, alpha: float = 0.85, beta: float = 0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (1 - 1 / self.n_nodes) * (1 / alpha - 1)
        eta = (1 / beta - 1) / alpha
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes) * (n_nodes / (n_nodes - 1)) - torch.ones(n_nodes, n_nodes) / (n_nodes - 1)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik + theta * L1), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

    def arb_ve( 
        self,
        out: torch.Tensor = None,
        alpha: float = 0.99,
        beta: float = 0.9,
        num_iter: int = 20,
        virtual_mode: str = "full",  # 'full' | 'knn' | 'thresh' | 'rbf' | 'random' | 'none'
        gamma: float = 1.0,          # weight of the virtual-edge channel (added into the graph)
        k: int = 10,                 # k for knn/rbf
        sim_thresh: float = 0.8,     # threshold for 'thresh' mode
        rbf_sigma: float = None,     # sigma for 'rbf' mode
        random_edges: int = None     # number of edges for 'random' mode
    ) -> torch.Tensor:
        if out is None:
            out = self.out
        device, dtype = out.device, out.dtype

        # Standardization
        out = (out - self.mean) / (self.std + 1e-12)

        # Real graph (assumed to be a symmetrically normalized propagation matrix)
        A_real = self.adj.coalesce()

        # Build virtual edges
        A_virtual = self._build_virtual_adj(virtual_mode, k, sim_thresh, rbf_sigma, random_edges)

        # Merge adjacencies & normalize consistently
        if A_virtual is not None and A_virtual._nnz() > 0 and gamma > 0:
            if A_virtual.device != A_real.device or A_virtual.dtype != A_real.dtype:
                A_virtual = torch.sparse_coo_tensor(
                    A_virtual.indices(),
                    A_virtual.values().to(device=device, dtype=dtype),
                    size=A_virtual.shape, device=device, dtype=dtype
                ).coalesce()
            A_tot = (A_real + gamma * A_virtual).coalesce()
            A_tot = self._symmetric_normalize(A_tot)
            use_fc_mean = False
        else:
            A_tot = self._symmetric_normalize(A_real)
            use_fc_mean = (virtual_mode == "full")

        # Iterative propagation
        for _ in range(num_iter):
            prop = torch.sparse.mm(A_tot, out)
            if use_fc_mean:
                out = alpha * prop + (1 - alpha) * out.mean(dim=0, keepdim=True)  # full: use the global-mean channel
            else:
                # Non-'full': virtual edges already merged into the graph;
                # alpha=1.0 is recommended (you may keep alpha for ablation/contrast)
                out = alpha * prop + (1 - alpha) * out.mean(dim=0, keepdim=True)
            # Reset boundary conditions
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init

        # De-standardization
        return out * (self.std + 1e-12) + self.mean


class arbLabel:

    def __init__(self, edge_index: Adj, x: torch.Tensor, y: torch.Tensor, know_mask: torch.Tensor, is_binary: bool):
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self._adj = None

        self._label_adj = None
        self._label_adj_25 = None
        self._label_adj_50 = None
        self._label_adj_75 = None
        self._label_adj_all = None
        self._label_mask = know_mask
        self._label_mask_25 = None
        self._label_mask_50 = None
        self._label_mask_75 = None

        self.know_mask = know_mask
        self.mean = 0 if is_binary else self.x[self.know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask]-self.mean) / self.std
        # init self.out without normalized
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def label_adj(self):
        if self._label_adj is None:
            edge_index = get_edge_index_from_y(self.y, self.know_mask)
            self._label_adj = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj, self._label_mask
    
    def label_adj_25(self):
        if self._label_adj_25 is None:
            _, label_mask_50 = self.label_adj_50()
            self._label_mask_25 = torch.tensor(random.sample(label_mask_50.tolist(), int(0.5*label_mask_50.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_25)
            self._label_adj_25 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_25, self._label_mask_25

    def label_adj_50(self):
        if self._label_adj_50 is None:
            _, label_mask_75 = self.label_adj_75()
            self._label_mask_50 = torch.tensor(random.sample(label_mask_75.tolist(), int(0.75*label_mask_75.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_50)
            self._label_adj_50 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_50, self._label_mask_50

    def label_adj_75(self):
        if self._label_adj_75 is None:
            edge_index, self._label_mask_75 = get_edge_index_from_y_ratio(self.y, 0.75)
            self._label_adj_75 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_75, self._label_mask_75

    @property
    def label_adj_all(self):
        if self._label_adj_all is None:
            edge_index = get_edge_index_from_y(self.y)
            self._label_adj_all = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_all

    def arb(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def _arb_label(self, adj: Adj, mask:torch.Tensor, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1):
        G = torch.ones(self.n_nodes)
        G[mask] = gamma
        G = G.unsqueeze(1)
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = G*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-G)*torch.spmm(adj, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def arb_label_25(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_25()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_50(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_50()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_75(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_75()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_100(self, out: torch.Tensor = None, alpha: float = 1.0, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    
    def arb_label_all(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*torch.spmm(self.adj, out) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean


class arbLoss(nn.Module):
    def __init__(self, edge_index:Adj, raw_x:torch.Tensor, know_mask:torch.Tensor, alpha, beta, is_binary:bool, **kw):
        super().__init__()
        num_nodes = raw_x.size(0)
        self.n_nodes = num_nodes
        num_attrs = raw_x.size(1)
        self.know_mask = know_mask

        self.mean = 0 if is_binary else raw_x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (raw_x[know_mask]-self.mean) / self.std

        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
        self.L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to_dense().to(edge_index.device)
        self.avg_L = num_nodes/(num_nodes-1)*torch.eye(num_nodes) - 1/(num_nodes-1)*torch.ones(num_nodes, num_nodes)
        self.x = nn.Parameter(torch.zeros(num_nodes, num_attrs))
        self.x.data[know_mask] = raw_x[know_mask].clone().detach().data
        if alpha == 0:
            alpha = 0.00001
        if beta == 0:
            beta = 0.00001
        self.theta = (1 - 1/num_nodes) * (1/alpha - 1)
        self.eta = (1/beta - 1)/alpha
        print(alpha, beta, self.theta, self.eta)

    def get_loss(self, x):
        x = (x - self.mean)/self.std
        dirichlet_loss = to_dirichlet_loss(x, self.L)
        avg_loss = to_dirichlet_loss(x, self.avg_L)
        recon_loss = nn.functional.mse_loss(x[self.know_mask], self.out_k_init, reduction="sum")
        return dirichlet_loss + self.eta * recon_loss + self.theta * avg_loss

    def forward(self):
        return self.get_loss(self.x)
    
    def get_out(self):
        return self.x


class arbwithParams(nn.Module):

    def __init__(self, x: torch.Tensor, y: torch.Tensor, edge_index: Adj, know_mask: torch.Tensor, is_binary: bool):
        super().__init__()
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.n_attrs = x.size(1)
        self.edge_index = edge_index
        self._adj = None
        self.know_mask = know_mask
        self.is_binary = is_binary
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (x[know_mask]-self.mean) / self.std
        # init self.out without ormalized
        self.out = torch.zeros_like(x)
        self.out[know_mask] = x[know_mask]
        # parameters
        self.eta, self.theta = nn.Parameter(torch.zeros(self.n_attrs)), nn.Parameter(torch.zeros(self.n_attrs))

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def forward(self, num_iter: int = 30) -> torch.Tensor:
        alpha = (self.n_nodes-1)/(self.theta*self.n_nodes+self.n_nodes-1)
        beta = 1/alpha / (1/alpha+self.eta)
        out = (self.out.clone().detach() - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    


