import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pdb

class Triaffine(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, input_size_1, input_size_2, input_size_3, init_std, rank=257, factorize = False, flag=True, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(Triaffine, self).__init__()
        self.flag = flag
        if flag:
            self.input_size_1 = input_size_1+1
            self.input_size_2 = input_size_2+1
            self.input_size_3 = input_size_3
        else:
            self.input_size_1 = input_size_1
            self.input_size_2 = input_size_2
            self.input_size_3 = input_size_3
        self.rank = rank
        self.init_std = init_std
        self.factorize = factorize
        if not factorize:
            self.W = Parameter(torch.Tensor(self.input_size_1, self.input_size_2, self.input_size_3, self.input_size_3))
        else:
            self.W_1 = Parameter(torch.Tensor(self.input_size_1, self.input_size_2, self.input_size_3))
            self.W_2 = Parameter(torch.Tensor(self.input_size_1, self.input_size_2, self.input_size_3))
            self.W_3 = Parameter(torch.Tensor(self.input_size_3, self.input_size_3))
            #self.W_1 = Parameter(torch.Tensor(self.input_size_1, self.rank))
            #self.W_2 = Parameter(torch.Tensor(self.input_size_2, self.rank))
            #self.W_3 = Parameter(torch.Tensor(self.input_size_3, self.rank))
            #self.W_4 = Parameter(torch.Tensor(self.input_size_3, self.rank))
        # if self.biaffine:
        #     self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        # else:
        #     self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        if not self.factorize:
            #nn.init.zeros_(self.W)
            nn.init.xavier_normal_(self.W)
        else:
            nn.init.xavier_normal_(self.W_1, gain=self.init_std)
            nn.init.xavier_normal_(self.W_2, gain=self.init_std)
            nn.init.xavier_normal_(self.W_3, gain=self.init_std)
            #nn.init.xavier_normal_(self.W_4, gain=self.init_std)
    def forward(self, layer1, layer2, h1, h2):
        """
        Args:
            
        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert layer1.size(0) == layer2.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        layer_shape = layer1.size()
        one_shape = list(layer_shape[:2])+[1]
        ones = torch.ones(one_shape).cuda()
        if self.flag:
            layer1 = torch.cat([layer1,ones],-1)
            layer2 = torch.cat([layer2,ones],-1)
            #layer3 = torch.cat([layer3,ones],-1)
        if not self.factorize:
            # layer = torch.einsum('nia,njb,abc,nkc->nijk', layer1, layer2, self.W, layer3)
            if layer1.dim()==3: layer = torch.einsum('nia,abcd,njb,nic,njd->nij', layer1, self.W, layer2, h1, h2)
            else: layer = torch.einsum('snia,abc,snjb,snic,snjb->snij', layer1, self.W, layer2, h1, h2)
            # layer1_temp = torch.einsum('nia,abc->nibc', layer1, self.W)
            # layer12_temp = torch.einsum('njb,nibc->nijc', layer2, layer1_temp)
            # layer = torch.einsum('nkc,nijc->nijk', layer3, layer12_temp)
        else:
            # layer = torch.einsum('nia,njb,nkc,al,bl,cl->nijk', layer1, layer2, layer3, self.W_1, self.W_2, self.W_3)
            # pdb.set_trace()
            # layer = torch.einsum('nia,al,njb,bl,nkc,cl->nijk', layer1, self.W_1, layer2, self.W_2, layer3, self.W_3)
            # nia * al -> nil * bl -> nibl * njb -> nijl * cl -> nijc * nkc -> nijk
            if layer1.dim()==3: 
                layer_l = torch.einsum('nia,abc,njb,nic->nijc', layer1, self.W_1, layer2, h1)
                layer_r = torch.einsum('nia,abc,njb,nic->nijc', layer1, self.W_2, layer2, h2)
                layer = torch.einsum('nijc,cd,nijd->nij', layer_l, self.W_3, layer_r)
            else: 
                layer_l = torch.einsum('snia,abc,snjb,snic->snijc', layer1, self.W_1, layer2, h1)
                layer_r = torch.einsum('snia,abc,snjb,snic->snijc', layer1, self.W_2, layer2, h2)
                layer = torch.einsum('snijc,cd,snijd->snij', layer_l, self.W_3, layer_r)
                #layer = torch.einsum('snia,abc,snjb,snic,cd,snjd->snij', layer1, self.W, layer2, h1, self.W2, h2)

            #if layer1.dim()==3: layer = torch.einsum('al,nia,bl,njb,cl,nic,dl,njd ->nij', self.W_1, layer1, self.W_2, layer2, self.W_3, h1, self.W_4, h2)
            #else: layer = torch.einsum('al,snia,bl,snjb,cl,snic,dl,snjd ->snij', self.W_1, layer1, self.W_2, layer2, self.W_3, h1, self.W_4, h2)
            # nil * njl -> nijl * nkl -> nijk
            # # (n x m x d) * (d x k) -> (n x m x k)
            # layer1_temp = torch.matmul(layer1, self.W_1)
            # layer2_temp = torch.matmul(layer2, self.W_2)
            # layer3_temp = torch.matmul(layer3, self.W_3)
            # layer12_temp = torch.einsum('nak,nbk->nabk',(layer1_temp,layer2_temp))
            # layer = torch.einsum('nabk,nck->nabc',(layer12_temp,layer3_temp))
        
        return layer
