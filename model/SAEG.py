# Graph Models for Synthetic Lethality

# Luis F. Iglesias-Martinez, PhD
# Systems Biology Ireland, University College Dublin, Dublin, Ireland.
#
# Copyright 2024 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You can see a copy of the license in the link below.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations.



import torch


class GCN_block(torch.nn.Module):
    """
    Graph Neural Network Block.
    dim_in: Integer with the dimension (default 512).
    hidden_dim: Integer with the dimension for the neural network (default 2048).
    """
    def __init__(self, dim_in = 512, hidden_dim = 512*4):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim_in, hidden_dim, bias = True)
        self.linear2 = torch.nn.Linear(dim_in, hidden_dim, bias = True)
        self.linear3 = torch.nn.Linear(hidden_dim, dim_in, bias = True)
        self.norm1 = torch.nn.RMSNorm(dim_in)
        self.norm2 = torch.nn.RMSNorm(dim_in)
    def forward(self, x, DA):
        h = x + torch.matmul(self.norm1(x).transpose(2,1), DA).transpose(2,1)
        x_n = self.norm2(h)
        x2 = self.linear1(x_n)
        x3 = self.linear2(x_n)
        x =  h + self.linear3(torch.nn.functional.silu(x2)*x3)
        return x


class GAT_block(torch.nn.Module):
    """
    Graph Attention Block.
    dim_in: Integer with the dimension (default 512).
    hidden_dim: Integer with the dimension for the neural network (default 2048).
    heads: Integer with the number of heads for the attention operation (default 8).
    """
    def __init__(self, dim_in = 512, hidden_dim = 512*4, heads = 8):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim_in, hidden_dim, bias = True)
        self.linear2 = torch.nn.Linear(dim_in, hidden_dim, bias = True)
        self.linear3 = torch.nn.Linear(hidden_dim, dim_in, bias = True)
        self.norm1 = torch.nn.RMSNorm(dim_in)
        self.norm2 = torch.nn.RMSNorm(dim_in)
        self.key = torch.nn.Linear(dim_in, dim_in, bias = True) 
        self.value = torch.nn.Linear(dim_in, dim_in, bias = True) 
        self.query = torch.nn.Linear(dim_in, dim_in, bias = True)
        self.heads = heads
        self.d = dim_in
    def forward(self, x, DA):
        xn = self.norm1(x)
        k = self.key(xn)
        q = self.query(xn)
        v = self.value(xn)
        q = q.view(q.shape[0], q.shape[1],self.heads,  q.shape[2]//self.heads)
        q =q.transpose(2,1)
        k = k.view(k.shape[0], k.shape[1],self.heads, k.shape[2]//self.heads)
        k = k.transpose(2,1)
        v = v.view(v.shape[0], v.shape[1], self.heads,v.shape[2]//self.heads)
        v = v.transpose(2,1)
        qk = torch.matmul(q, k.transpose(-2, -1))/(self.d/self.heads)**.5
        qk = qk.masked_fill(DA==0, -1.e20)
        qk = torch.nn.functional.softmax(qk, dim = -1)
        v = torch.matmul(qk,v)
        v = v.transpose(1,2).contiguous().view(x.shape[0], x.shape[1], self.d)
        h  = x + v
        x_n = self.norm2(h)
        x2 = self.linear1(x_n)
        x3 = self.linear2(x_n)
        x =  h + self.linear3(torch.nn.functional.silu(x2)*x3)
        return x





class somatic_emb(torch.nn.Module):
    """
    Somatic Embedding Layer.
    no_genes: integer that refers to the number of genes in the system
    no_aminos: Number of amino acids for the amino acid embedding layer . 
    max_pos: The largest protein mutation position to consider. 
    """
    def __init__(self, no_genes, no_aminos, max_pos, dim = 100):
        super().__init__()
        self.gene_emb = torch.nn.Embedding(no_genes, dim)
        self.semb  = snv_embedder(no_aminos, max_pos, dim)
        self.cnemb = torch.nn.Linear(1, dim)
    def forward(self, genes, muts, cnas ):
        x1 = self.gene_emb(genes)
        x2 = self.semb(muts)
        x3 = self.cnemb(cnas)
        x = torch.concatenate((x1, x2, x3), dim = 2)
        return x




class GCN(torch.nn.Module):
    """
    SAEG Graph Neural Network.
    no_genes: integer that refers to the number of genes in the system
    no_aminos: integerr with number of amino acids for the amino acid embedding layer . 
    max_pos: integer The largest protein mutation position to consider. 
    dim: integer with the embedding dimension.
    layer: integer with the number of layers in the GCN block.
    """
    def __init__(self, no_genes,  no_aminos, max_pos, out, dim = 100,layers = 3):
        super().__init__()
        self.semb = somatic_emb(no_genes , no_aminos, max_pos, dim)
        gcn_list = list()
        for i in range(layers):
            gcn_list.append(GCN_block(dim*6, dim*6))
        self.gcn = torch.nn.ModuleList(gcn_list)
        self.out1 = torch.nn.Linear(dim*6, out, bias = False)
    def forward(self, x_gi, x_mi, x_ci,  DA):
        x = self.semb(x_gi, x_mi, x_ci)
        for i in self.gcn:
            x = i(x, DA)
        x = self.out1(x) 
        return x


class GAT(torch.nn.Module):
    """
    SAEG Graph Attention Neural Network.
    no_genes: integer that refers to the number of genes in the system
    no_aminos: integerr with number of amino acids for the amino acid embedding layer . 
    max_pos: integer The largest protein mutation position to consider. 
    dim: integer with the embedding dimension.
    layer: integer with the number of layers in the GCN block.
    """
    def __init__(self, no_genes,  no_aminos, max_pos, out, dim = 100,layers = 3):
        super().__init__()
        self.semb = somatic_emb(no_genes , no_aminos, max_pos, dim)
        gcn_list = list()
        for i in range(layers):
            gcn_list.append(GAT_block(dim*6, dim*6, heads = 1))
        self.gcn = torch.nn.ModuleList(gcn_list)
        self.out1 = torch.nn.Linear(dim*6, out, bias = True)
    def forward(self, x_gi, x_mi, x_ci,  DA):
        x = self.semb(x_gi, x_mi, x_ci)
        for i in self.gcn:
            x = i(x, DA)
        x = self.out1(x) 
        return x



class snv_embedder(torch.nn.Module):
    """
    Embedding Layer to produce a float embedding for amino-acids and 
    protein change position. 
    The layers use a padding index of 0, so when no change in protein
    structure is known a tensor of 0s is produced.
    To initiate the layer, number of amino acids and maximum position 
    for protein position of mutation are required.
    The input for the forward method is an integer tensor with 3 dimensions
    The first dimension is an integer that represents the original amino acid,
    The second dimension; an integer that represents the original amino acid,
    The third dimension the position of change. 
    """
    def __init__(self, no_aminos, max_pos, dim):
        super().__init__()
        self.mut_emb = torch.nn.Embedding(2, dim, padding_idx = 0)
        self.aemb =  torch.nn.Embedding(no_aminos, dim, padding_idx = 0)
        self.pemb = pemb(max_pos, dim, 1)
    def forward(self, x):
        me = torch.flatten(self.mut_emb(x[:,:,0:1]),  2)
        a1 = torch.flatten(self.aemb(x[:,:,1:2]),  2)
        a2 = torch.flatten(self.aemb(x[:,:,2:3]),  2)
        p = self.pemb[x[:,:,3]]
        x = torch.concat((me,a1,a2,p), dim = 2)
        return x




def pemb(max_pos, dim, start = 0):
    """
    Positional Embedding layer using sine and cosine layers
    for mutation sequence.
    """
    pe = torch.zeros(max_pos, dim)
    for pos in range(start, max_pos):
        for i in range(0, dim, 2):
            pe[pos, i] = torch.math.sin(pos / (10000 ** ((2 * i) / dim)))
            pe[pos, i + 1] = torch.math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))
    return pe




class GCN_b(torch.nn.Module):
    def __init__(self, no_genes,  no_aminos, max_pos, out, dim_g = 400, dim_m = 100, dim_a = 50, dim_p = 100, dim_c = 1,  layers = 3):
        super().__init__()
        self.semb = somatic_emb_b(no_genes , no_aminos, max_pos,dim_m,  dim_g, dim_a, dim_p, dim_c)
        gcn_list = list()
        dim = dim_g + dim_m + 2*dim_a + dim_p + dim_c
        for i in range(layers):
            gcn_list.append(GCN_block(dim, dim))
        self.gcn = torch.nn.ModuleList(gcn_list)
        self.out1 = torch.nn.Linear(dim, out, bias = False)
    def forward(self, x_gi, x_mi, x_ci,  DA):
        x = self.semb(x_gi, x_mi, x_ci)
        for i in self.gcn:
            x = i(x, DA)
        x = self.out1(x) 
        return x



class snv_embedder_b(torch.nn.Module):
    """
    Embedding Layer to produce a float embedding for amino-acids and 
    protein change position. 
    The layers use a padding index of 0, so when no change in protein
    structure is known a tensor of 0s is produced.
    To initiate the layer, number of amino acids and maximum position 
    for protein position of mutation are required.
    The input for the forward method is an integer tensor with 3 dimensions
    The first dimension is an integer that represents the original amino acid,
    The second dimension; an integer that represents the original amino acid,
    The third dimension the position of change. 
    """
    def __init__(self, no_aminos, max_pos, dim_m, dim_a, dim_p):
        super().__init__()
        self.mut_emb = torch.nn.Embedding(2, dim_m, padding_idx = 0)
        self.aemb =  torch.nn.Embedding(no_aminos, dim_a, padding_idx = 0)
        print(dim_p)
        self.pemb = pemb(max_pos, dim_p, 1)
    def forward(self, x):
        me = torch.flatten(self.mut_emb(x[:,:,0:1]),  2)
        a1 = torch.flatten(self.aemb(x[:,:,1:2]),  2)
        a2 = torch.flatten(self.aemb(x[:,:,2:3]),  2)
        p = self.pemb[x[:,:,3]]
        x = torch.concat((me,a1,a2,p), dim = 2)
        return x




class somatic_emb_b(torch.nn.Module):
    def __init__(self, no_genes, no_aminos, max_pos, dim_g = 400, dim_m = 50, dim_a = 50, dim_p = 100, dim_c = 100):
        super().__init__()
        self.gene_emb = torch.nn.Embedding(no_genes, dim_g)
        self.semb  = snv_embedder_b(no_aminos, max_pos, dim_m, dim_a, dim_p)
        self.cnemb = torch.nn.Linear(1, dim_c)
    def forward(self, genes, muts, cnas ):
        x1 = self.gene_emb(genes)
        x2 = self.semb(muts)
        x3 = self.cnemb(cnas)
        x = torch.concatenate((x1, x2, x3), dim = 2)
        return x


