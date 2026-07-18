import torch

from torch import Tensor
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Legacy attention utilities retained for compatibility with older experiments.


def transpose_qkv(X,num_heads):
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X=X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

class Attention(nn.Module):
    def __init__(
        self,
        in_channels: int
        
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, input: Tensor) -> Tensor:
        
        batch_size, channels, height, width = input.size()

        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)

        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)

        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)

        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> B, H * W, H * W
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)
 
        return self.gamma * out + input
    

class MultiheadAttention(nn.Module):
    def __init__(self, dim_in ,n_heads=4, dropout=0.1) -> None:
        super(MultiheadAttention,self).__init__()

        self.dim_in=dim_in
        self.hid_dim=torch.tensor((dim_in//8))
        self.n_head=n_heads
        self.gamma = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(p=dropout)

        assert self.hid_dim % n_heads == 0, "hidden dimension must be divisible by n_heads"

        self.query = nn.Conv2d(self.dim_in, self.hid_dim, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(self.dim_in, self.hid_dim, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(self.dim_in, self.hid_dim, kernel_size = 1, stride = 1)
        self.scale = 1/ torch.sqrt(self.hid_dim//n_heads)

        self.concat =nn.Linear(self.hid_dim,self.dim_in)

    def forward(self,x):
        
        batch, dim_in, height, width = x.size()
        #check_dim=self.dim_in(0)
        assert dim_in == self.dim_in

        nh=self.n_head
        dk=self.hid_dim//nh

        q=self.query(x)
        q=q.reshape(batch,-1, nh, dk).transpose(1,2)
        k=self.key(x)
        k=k.reshape(batch,-1,nh,dk).transpose(1,2)
        v=self.value(x)
        v=v.reshape(batch,-1,nh,dk).transpose(1,2)
        
        dist=torch.matmul(q,k.transpose(2,3))*self.scale
        dist=torch.softmax(dist,dim=-1)#batch,nh,n,n

        dist=self.dropout(dist)

        att=torch.matmul(dist,v)#batch,nh,n,dv
        att=att.transpose(1,2).reshape(batch,-1,self.hid_dim)

        out=self.concat(att)
        out=out.view(batch,-1, height, width)

        return out+x


class Danet_PositAttention(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Danet_PositAttention,self).__init__()

        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, input: Tensor) -> Tensor:
        
        batch_size, _ , height, width = input.size()

        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(input).view(batch_size, -1, height * width)
        v = self.value(input).view(batch_size, -1, height * width)

        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)
 
        return self.alpha * out + input



class Danet_ChannAttention(nn.Module):
    def __init__(self) -> None:
        super(Danet_ChannAttention,self).__init__()

        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,input:Tensor) -> Tensor:

        batch_size, _ , height, width = input.size()
        q_c=input.view(batch_size,-1,height*width)
        p_c=input.view(batch_size,-1,height*width).permute(0,2,1)
        attention = torch.bmm(q_c, p_c)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, q_c).view(batch_size, -1, height, width)
        out = self.beta * feat_e + input

        return self.beta * out + input
