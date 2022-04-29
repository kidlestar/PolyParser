# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TrilinearScorer, Triaffine
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                                                pad_sequence)
from torch.distributions.gumbel import Gumbel
import pdb
import torch.nn.functional as F


from torch.autograd import grad


def pargmax(E, mask):
    nE = E.clone()
    mask = mask[:,1:,1:].transpose(-1,-2)
    nE[~mask] = -float('inf')
    preds = nE.argmax(1)
    ptree = preds.new_zeros(nE.size())
    ptree.scatter_(1,preds.unsqueeze(-2),1)
    ptree[~mask] = 0
    return ptree


class Model(nn.Module):

        def __init__(self, args):
                super(Model, self).__init__()
                #self.gumbel = Gumbel(0, 1)
                self.args = args
                # the embedding layer
                self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                                                           embedding_dim=args.n_embed)
                if args.use_char:
                        self.char_embed = CHAR_LSTM(n_chars=args.n_char_feats,
                                                                                n_embed=args.n_char_embed,
                                                                                n_out=args.n_feat_embed)
                if args.use_bert:
                        self.bert_embed = BertEmbedding(model=args.bert_model,
                                                                                        n_layers=args.n_bert_layers,
                                                                                        n_out=args.n_feat_embed)
                if args.use_pos:
                        self.pos_embed = nn.Embedding(num_embeddings=args.n_pos_feats,
                                                                                   embedding_dim=args.n_feat_embed)
                self.embed_dropout = IndependentDropout(p=args.embed_dropout)

                # the word-lstm layer
                self.lstm = BiLSTM(input_size=args.n_feat_embed*(args.use_char+args.use_bert+args.use_pos)+args.n_embed,
                                                   hidden_size=args.n_lstm_hidden,
                                                   num_layers=args.n_lstm_layers,
                                                   dropout=args.lstm_dropout)
                self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

                # the MLP layers
                self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
                                                         n_hidden=args.n_mlp_arc,
                                                         dropout=args.mlp_dropout)
                self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
                                                         n_hidden=args.n_mlp_arc,
                                                         dropout=args.mlp_dropout)
                self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                                                         n_hidden=args.n_mlp_rel,
                                                         dropout=args.mlp_dropout)
                self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                                                         n_hidden=args.n_mlp_rel,
                                                         dropout=args.mlp_dropout)

                # the Biaffine layers
                self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                                                 bias_x=True,
                                                                 bias_y=False)
                self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                                                 n_out=args.n_rels,
                                                                 bias_x=True,
                                                                 bias_y=True)
                self.binary = args.binary
                # the Second Order Parts
                if self.args.use_second_order:
                        self.use_sib = args.use_sib
                        self.use_cop = args.use_cop
                        self.use_gp = args.use_gp
                        #regard sibling as second order part
                        if args.use_sib:
                                self.mlp_sib_h = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.mlp_sib_d = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.mlp_sib_ch = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.mlp_sib_cd = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.trilinear_sib = Triaffine(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = True)
                        if args.use_cop:
                                self.mlp_cop_h = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.mlp_cop_d = MLP(n_in=args.n_lstm_hidden*2,
                                                                 n_hidden=args.n_mlp_sec,
                                                                 dropout=args.mlp_dropout,identity=self.binary)
                                self.trilinear_cop = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
                        #regard gp as third order part
                        #if args.use_gp:
                                #self.mlp_gp_h = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                #self.mlp_gp_d = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                #self.mlp_gp_s = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                
                                #self.mlp_gp_th = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                #self.mlp_gp_td = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                #self.mlp_gp_ts = MLP(n_in=args.n_lstm_hidden*2,
                                #                                 n_hidden=args.n_mlp_sec,
                                #                                 dropout=args.mlp_dropout,identity=self.binary)
                                #for calculating the linear energy
                                #self.wh_gp = nn.Parameter(torch.Tensor(args.n_mlp_sec, args.n_mlp_sec))
                                #nn.init.eye_(self.wh_gp)
                                #transformation to have vector to third order features
                                #self.wd_gp = nn.Parameter(torch.Tensor(args.n_mlp_sec, args.n_mlp_sec))
                                #nn.init.eye_(self.wd_gp)
                                #self.ws_gp = nn.Parameter(torch.Tensor(args.n_mlp_sec, args.n_mlp_sec))
                                #nn.init.eye_(self.ws_gp)
                                #self.trilinear_gp = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
                                #self.trilinear_tgp = TrilinearScorer(args.n_mlp_sec,args.n_mlp_sec,args.n_mlp_sec,init_std=args.init_std, rank = args.n_mlp_sec, factorize = args.factorize)
                                
                self.pad_index = args.pad_index
                self.unk_index = args.unk_index

        def load_pretrained(self, embed=None):
                if embed is not None:
                        self.pretrained = nn.Embedding.from_pretrained(embed)
                        nn.init.zeros_(self.word_embed.weight)

                return self

        def forward(self, words, feats):
            if self.training:
                batch_size, seq_len = words.shape
                # get the mask and lengths of given batch
                
                mask = words.ne(self.pad_index)
                self.rmask = mask.clone()
                self.rmask[:,0] = 0
                lens = mask.sum(dim=1)
                # set the indices larger than num_embeddings to unk_index
                ext_mask = words.ge(self.word_embed.num_embeddings)
                ext_words = words.masked_fill(ext_mask, self.unk_index)
                # get outputs from embedding layers
                word_embed = self.word_embed(ext_words)
                if hasattr(self, 'pretrained'):
                        word_embed += self.pretrained(words)
                feat_embeds=[word_embed]
                feats_index=0
                # pdb.set_trace()
                if self.args.use_char:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        char_embed = self.char_embed(input_feats[mask])
                        char_embed = pad_sequence(char_embed.split(lens.tolist()), True)
                        # char_embed = self.embed_dropout(char_embed)
                        feat_embeds.append(char_embed)
                if self.args.use_bert:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        bert_embed = self.bert_embed(*input_feats)
                        # bert_embed = self.embed_dropout(bert_embed)
                        feat_embeds.append(bert_embed)
                if self.args.use_pos:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        pos_embed = self.pos_embed(input_feats)
                        # pos_embed = self.embed_dropout(pos_embed)
                        feat_embeds.append(pos_embed)
                feat_embeds=self.embed_dropout(*feat_embeds)
                # for i in range(len(feat_embeds)):
                #       feat_embeds[i]=self.embed_dropout(feat_embeds[i])

                # word_embed = self.embed_dropout(word_embed)
                # concatenate the word and feat representations
                embed = torch.cat(feat_embeds, dim=-1)

                x = pack_padded_sequence(embed, lens.cpu(), True, False)
                x, _ = self.lstm(x)
                x, _ = pad_packed_sequence(x, True, total_length=seq_len)
                x = self.lstm_dropout(x)

                # apply MLPs to the BiLSTM output states
                arc_h = self.mlp_arc_h(x)
                arc_d = self.mlp_arc_d(x)
                rel_h = self.mlp_rel_h(x)
                rel_d = self.mlp_rel_d(x)

                # get arc and rel scores from the bilinear attention
                # [batch_size, seq_len, seq_len]
                s_arc = self.arc_attn(arc_d, arc_h)
                #s_arc += self.gumbel.sample(s_arc.size()).cuda()
                # [batch_size, seq_len, seq_len, n_rels]
                s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
                #s_rel += self.gumbel.sample(s_rel.size()).cuda()
                # add second order using mean field variational inference
                if self.args.use_second_order:
                        self.mask_unary, self.mask_sib, self.mask_cop, self.mask_gp = self.from_mask_to_3d_mask(mask)
                        self.unary = self.mask_unary*s_arc
                        self.arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
                        #self.arc_sib = (arc_sib[2], arc_sib[3])
                        #arc_sib = (arc_sib[0], arc_sib[1])
                        #self.arc_gp = (arc_gp[3], arc_gp[4], arc_gp[5])
                        #arc_gp = (arc_gp[0], arc_gp[1], arc_gp[2])
                        #self.layer_sib, layer_cop, self.layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, self.mask_sib, self.mask_cop, self.mask_gp)
                        #layer_sib, layer_cop, layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
                        #s_arc = self.mean_field_variational_infernece(unary, mask_unary, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
                        #s_arc = self.run()
                        s_arc = self.mean_field_variational_infernece()
                        #s_arc = self.mean_field_variational_infernece(unary, layer_sib, layer_cop, layer_gp, mask_unary) 
                # set the scores that exceed the length of each sentence to -inf
                s_arc.masked_fill_(~mask.unsqueeze(1), float(-999999999))

                return s_arc, s_rel
            with torch.no_grad():
                batch_size, seq_len = words.shape
                # get the mask and lengths of given batch
                mask = words.ne(self.pad_index)
                self.rmask = mask.clone()
                self.rmask[:,0] = 0
                lens = mask.sum(dim=1)
                # set the indices larger than num_embeddings to unk_index
                ext_mask = words.ge(self.word_embed.num_embeddings)
                ext_words = words.masked_fill(ext_mask, self.unk_index)
                # get outputs from embedding layers
                word_embed = self.word_embed(ext_words)
                if hasattr(self, 'pretrained'):
                        word_embed += self.pretrained(words)
                feat_embeds=[word_embed]
                feats_index=0
                # pdb.set_trace()
                if self.args.use_char:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        char_embed = self.char_embed(input_feats[mask])
                        char_embed = pad_sequence(char_embed.split(lens.tolist()), True)
                        # char_embed = self.embed_dropout(char_embed)
                        feat_embeds.append(char_embed)
                if self.args.use_bert:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        bert_embed = self.bert_embed(*input_feats)
                        # bert_embed = self.embed_dropout(bert_embed)
                        feat_embeds.append(bert_embed)
                if self.args.use_pos:
                        input_feats=feats[feats_index]
                        feats_index+=1
                        pos_embed = self.pos_embed(input_feats)
                        # pos_embed = self.embed_dropout(pos_embed)
                        feat_embeds.append(pos_embed)
                feat_embeds=self.embed_dropout(*feat_embeds)
                # for i in range(len(feat_embeds)):
                #       feat_embeds[i]=self.embed_dropout(feat_embeds[i])

                # word_embed = self.embed_dropout(word_embed)
                # concatenate the word and feat representations
                embed = torch.cat(feat_embeds, dim=-1)
                x = pack_padded_sequence(embed, lens.cpu(), True, False)
                x, _ = self.lstm(x)
                x, _ = pad_packed_sequence(x, True, total_length=seq_len)
                x = self.lstm_dropout(x)

                # apply MLPs to the BiLSTM output states
                arc_h = self.mlp_arc_h(x)
                arc_d = self.mlp_arc_d(x)
                rel_h = self.mlp_rel_h(x)
                rel_d = self.mlp_rel_d(x)

                # get arc and rel scores from the bilinear attention
                # [batch_size, seq_len, seq_len]
                s_arc = self.arc_attn(arc_d, arc_h)
                # [batch_size, seq_len, seq_len, n_rels]
                s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

                # add second order using mean field variational inference
                if self.args.use_second_order:
                        self.mask_unary, self.mask_sib, self.mask_cop, self.mask_gp = self.from_mask_to_3d_mask(mask)
                        self.unary = self.mask_unary*s_arc
                        self.arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
                        #self.arc_sib = (arc_sib[2], arc_sib[3])
                        #arc_sib = (arc_sib[0], arc_sib[1])
                        #self.arc_gp = (arc_gp[3], arc_gp[4], arc_gp[5])
                        #arc_gp = (arc_gp[0], arc_gp[1], arc_gp[2])
                        #self.layer_sib, layer_cop, self.layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, self.mask_sib, self.mask_cop, self.mask_gp)
            #s_arc = self.mean_field_variational_infernece(unary, mask_unary, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
            #s_arc = self.run()
            s_arc = self.mean_field_variational_infernece()
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~mask.unsqueeze(1), float(-999999999))

            return s_arc, s_rel


        @classmethod
        def load(cls, path):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                state = torch.load(path, map_location=device)
                model = cls(state['args'])

                model.load_pretrained(state['pretrained'])
                model.load_state_dict(state['state_dict'], False)
                model.to(device)

                return model

        def save(self, path):
                state_dict, pretrained = self.state_dict(), None
                if hasattr(self, 'pretrained'):
                        pretrained = state_dict.pop('pretrained.weight')
                state = {
                        'args': self.args,
                        'state_dict': state_dict,
                        'pretrained': pretrained
                }
                torch.save(state, path)
        
        def encode_second_order(self, memory_bank):

                if self.use_sib:
                        edge_node_sib_h = self.mlp_sib_h(memory_bank)
                        edge_node_sib_m = self.mlp_sib_d(memory_bank)
                        edge_node_sib_ch = self.mlp_sib_ch(memory_bank)
                        edge_node_sib_cm = self.mlp_sib_cd(memory_bank)
                        arc_sib=(edge_node_sib_h, edge_node_sib_m, edge_node_sib_ch, edge_node_sib_cm)
                else:
                        arc_sib=None

                if self.use_cop:
                        edge_node_cop_h = self.mlp_cop_h(memory_bank)
                        edge_node_cop_m = self.mlp_cop_d(memory_bank)
                        edge_node_cop_s = self.mlp_cop_s(memory_bank)
                        arc_cop=(edge_node_cop_h, edge_node_cop_m, edge_node_cop_s)
                else:
                        arc_cop=None

                if self.use_gp:
                        arc_gp = None
                        #edge_node_gp_h = self.mlp_gp_h(memory_bank)
                        #edge_node_gp_m = self.mlp_gp_d(memory_bank)
                        #edge_node_gp_s = self.mlp_gp_s(memory_bank)
                        #edge_node_gp_th = self.mlp_gp_th(memory_bank)
                        #edge_node_gp_tm = self.mlp_gp_td(memory_bank)
                        #edge_node_gp_ts = self.mlp_gp_ts(memory_bank)
                        #arc_gp=(edge_node_gp_h, edge_node_gp_m, edge_node_gp_s)
                        #arc_gp=(edge_node_gp_h, edge_node_gp_m, edge_node_gp_s, edge_node_gp_th, edge_node_gp_tm, edge_node_gp_ts)
                else:
                        arc_gp=None

                return arc_sib, arc_cop, arc_gp

        def get_edge_second_order_node_scores(self, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp):

                if self.use_sib:
                        edge_node_sib_h, edge_node_sib_m = arc_sib
                        layer_sib = self.trilinear_sib(edge_node_sib_h, edge_node_sib_m, edge_node_sib_m) * mask_sib
                        # keep (ma x mb x mc) -> (ma x mb x mb)
                        #layer_sib = 0.5 * (layer_sib + layer_sib.transpose(3,2))

                        one_mask=torch.ones(layer_sib.shape[-2:]).cuda()
                        tril_mask=torch.tril(one_mask,-1)
                        triu_mask=torch.triu(one_mask,1)
                        layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0).unsqueeze(0) #+ (layer_sib*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])

                        idx = torch.arange(layer_sib.size(-1))
                        #layer_sib = layer_sib.clone()
                        layer_sib[:,idx,idx,:] = 0
                        layer_sib[:,idx,:,idx] = 0
                        layer_sib[:,:,idx,idx] = 0


                else:
                        layer_sib = None
                if self.use_cop:
                        edge_node_cop_h, edge_node_cop_m = arc_cop
                        layer_cop = self.trilinear_cop(edge_node_cop_h, edge_node_cop_m, edge_node_cop_h) * mask_cop
                        # keep (ma x mb x mc) -> (ma x mb x ma)
                        one_mask=torch.ones(layer_cop.shape[-2:]).cuda()
                        tril_mask=torch.tril(one_mask,-1)
                        triu_mask=torch.triu(one_mask,1)
                        layer_cop=layer_cop.transpose(1,2)
                        layer_cop = layer_cop-layer_cop*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_cop*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
                        layer_cop=layer_cop.transpose(1,2)
                else:
                        layer_cop = None

                if self.use_gp:
                        layer_gp = None
                        #edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m = arc_gp
                        #layer_gp = self.trilinear_gp(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m) * mask_gp
                        #idx = torch.arange(layer_gp.size(-1))
                        #layer_gp = layer_gp.clone()
                        #layer_gp[:,idx,idx,:] = 0
                        #layer_gp[:,:,idx,idx] = 0

                else:
                        layer_gp = None

                return layer_sib,layer_cop,layer_gp
       
        #batchified energy
        def BNetEnergy(self, tree, bmask=None, grad=True, nsample=6):
            if bmask is None: bmask = torch.tensor([True]*tree.size(0)).cuda()
            unary = self.unary.repeat(nsample,1,1)[bmask]
            if not grad: unary = unary.detach()
            mask_unary = self.mask_unary.repeat(nsample,1,1)[bmask]
            unary = unary.transpose(1,2)
            unary_potential = unary.clone()
            q_value = torch.zeros(tree.size(0), tree.size(1)+1, tree.size(2)+1).cuda()
            idx = torch.arange(tree.size(-1))
            q_value[:,1:,1:] = tree
            q_value[:,0,1:] = tree[:,idx,idx]
            q_value[:,idx+1,idx+1] = 0
            rmask = self.rmask[bmask]
            if self.use_sib:
                me_sib = self.layer_sib.repeat(nsample,1,1,1)[bmask]
                if not grad: me_sib = me_sib.detach()
                E_sib = ((q_value.unsqueeze(-1))*(q_value.unsqueeze(-2))*me_sib).sum(dim=(-1,-2,-3))
            if self.use_gp:
                me_gp = self.layer_gp.repeat(nsample,1,1,1)[bmask]
                if not grad: me_gp = me_gp.detach()
                E_gp = ((q_value.unsqueeze(-1))*(q_value.unsqueeze(-3))*me_gp).sum(dim=(-1,-2,-3))
            E = (q_value*unary_potential).sum(dim=(-1,-2)) + E_sib + E_gp
            return E

        def NetEnergy(self, tree, bmask=None, grad=True, verse=False):
                #tree is masked
                if bmask is None: bmask = torch.tensor([True]*tree.size(0)).cuda()
                unary = self.unary[bmask]
                if not grad: unary = unary.detach()
                mask_unary = self.mask_unary[bmask]
                unary = unary.transpose(1,2)
                unary_potential = unary.clone()
                q_value = torch.zeros(tree.size(0), tree.size(1)+1, tree.size(2)+1).cuda()
                idx = torch.arange(tree.size(-1))
                q_value[:,1:,1:] = tree
                q_value[:,0,1:] = tree[:,idx,idx]
                q_value[:,idx+1,idx+1] = 0
                rmask = self.rmask[bmask]
                if self.use_sib:
                    #me_sib = self.layer_sib[bmask]
                    edge_node_sib_h_l, edge_node_sib_m_l,  edge_node_sib_h_r, edge_node_sib_m_r = self.arc_sib
                    edge_node_sib_h_l = edge_node_sib_h_l[bmask]
                    edge_node_sib_m_l = edge_node_sib_m_l[bmask]
                    edge_node_sib_h_r = edge_node_sib_h_r[bmask]
                    edge_node_sib_m_r = edge_node_sib_m_r[bmask]
                    if not grad:
                        #me_sib = me_sib.detach()
                        edge_node_sib_h_l = edge_node_sib_h_l.detach()
                        edge_node_sib_m_l = edge_node_sib_m_l.detach()
                        edge_node_sib_h_r = edge_node_sib_h_r.detach()
                        edge_node_sib_m_r = edge_node_sib_m_r.detach()
                    #E_sib = ((q_value.unsqueeze(-1))*(q_value.unsqueeze(-2))*me_sib).sum(dim=(-1,-2,-3))
                    edge_node_sib_h_l = torch.matmul(q_value.transpose(-1,-2),edge_node_sib_h_l)
                    edge_node_sib_h_r = torch.matmul(q_value.transpose(-1,-2),edge_node_sib_h_r)
                    layer_sib = self.trilinear_sib(edge_node_sib_m_l, edge_node_sib_m_r, edge_node_sib_h_l, edge_node_sib_h_r)

                    #layer_sib_r = self.trilinear_sib_r(edge_node_sib_m_r, edge_node_sib_m_r, edge_node_sib_h_r)
                    #layer_sib_l = (layer_sib_l*(q_value.transpose(-1,-2).unsqueeze(-2))).sum(-1)
                    #layer_sib_r = (layer_sib_r*(q_value.transpose(-1,-2).unsqueeze(-3))).sum(-1)
                    #layer_sib = layer_sib_l * layer_sib_r
                    one_mask=torch.ones(layer_sib.shape[-2:]).cuda()
                    tril_mask=torch.tril(one_mask,-1)
                    if tree.dim()==3:
                        layer_sib[:,idx,idx] = 0
                        layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0)
                    else:
                        layer_sib[:,:,idx,idx] = 0
                        layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0).unsqueeze(0)


                if self.use_gp:
                    pass
                    #me_gp = self.layer_gp[bmask]
                    #edge_node_gp_h, edge_node_gp_m, edge_node_gp_s = self.arc_gp
                    #edge_node_gp_h, edge_node_gp_m = self.arc_gp
                    #edge_node_gp_h = edge_node_gp_h[bmask]
                    #edge_node_gp_m = edge_node_gp_m[bmask]
                    #edge_node_gp_s = edge_node_gp_s[bmask]
                    #mask_gp = self.mask_gp[bmask]
                    #if not grad:
                        #me_gp = me_gp.detach()
                        #edge_node_gp_h = edge_node_gp_h.detach()
                        #edge_node_gp_m = edge_node_gp_m.detach()
                        #edge_node_gp_s = edge_node_gp_s.detach()
                    #E_gp = ((q_value.unsqueeze(-1))*(q_value.unsqueeze(-3))*me_gp).sum(dim=(-1,-2,-3))
                    #edge_node_gp_h = torch.matmul(q_value.transpose(1,2),edge_node_gp_h)*torch.matmul(edge_node_gp_h,self.wh_gp)
                    #edge_node_gp_m = torch.matmul(q_value.transpose(1,2),edge_node_gp_m)*torch.matmul(edge_node_gp_m,self.wd_gp)
                    #edge_node_gp_s = torch.matmul(q_value.transpose(1,2),edge_node_gp_s)*torch.matmul(edge_node_gp_s,self.ws_gp)
                    #edge_node_gp_h = torch.matmul(q_value.transpose(1,2),edge_node_gp_h)
                    #gp linear energy
                    #E_gp_lin = (edge_node_gp_h*torch.matmul(edge_node_gp_m, self.wh_gp)).sum(-1)
                    #E_gp_lin[~rmask] = 0
                    #edge_node_gp = edge_node_gp_h*torch.matmul(edge_node_gp_m, self.wd_gp)
                    #layer_gp = self.trilinear_tgp(edge_node_gp_h, edge_node_gp_m, edge_node_gp_s) * mask_gp
                    #layer_gp = self.trilinear_gp(edge_node_gp, edge_node_gp, edge_node_gp) * mask_gp
                    #idx = torch.arange(layer_gp.size(-1))
                    #layer_gp = layer_gp.clone()
                    #layer_gp[:,idx,idx,:] = 0
                    #layer_gp[:,:,idx,idx] = 0
                    #layer_gp[:,idx,:,idx] = 0
                    #one_mask=torch.ones(layer_gp.shape[-2:]).cuda()
                    #tril_mask=torch.tril(one_mask,-1)
                    #layer_gp = layer_gp-layer_gp*tril_mask.unsqueeze(0).unsqueeze(0)
                    #layer_gp = layer_gp-layer_gp*tril_mask.unsqueeze(0).unsqueeze(-1)
                    #layer_gp = layer_gp-layer_gp*tril_mask.unsqueeze(0).unsqueeze(-2)

                
                if not verse: 
                    E = (q_value*unary_potential).sum(dim=(-1,-2)) + layer_sib.sum(dim=(-1,-2)) #+ layer_gp.sum(dim=(-1,-2,-3)) #+ E_sib_lin.sum(-1) + E_gp_lin.sum(-1)
                    return E
                else: return (q_value*unary_potential).sum(dim=(-1,-2)), layer_sib.sum(dim=(-1,-2)), 0#+layer_gp.sum(dim=(-1,-2,-3)) 

        def GGS(self, ttree, vmask=None):
            idx = torch.arange(ttree.size(-1))
            ttree = ttree.requires_grad_()
            if vmask is None:
                bmask = torch.tensor([True]*ttree.size(0)).cuda()
                vmask = torch.tensor([True]*ttree.size(0)).cuda()
            else:
                bmask = torch.tensor([True]*(vmask.sum().item())).cuda()
            ovmask = vmask.clone()
            for i in range(100):
                E = self.NetEnergy(ttree, vmask, False)
                if i==0:
                    bE = E.detach()
                    btree = ttree.detach().clone()
                else:
                    tbmask = E > bE[bmask]
                    if not(tbmask.any()): break

                    vmask[vmask.clone()] = tbmask
                    bmask[bmask.clone()] = tbmask

                    bE[bmask] = E.detach()[tbmask]

                    btree[bmask] = ttree.detach()[tbmask]

                #if self.training: print('GGS energy: ' + str(bE.sum().item()))
                ptree, = grad(E.sum(), ttree, retain_graph=False)
                if not(i==0): ptree = ptree[tbmask]
                ptree = pargmax(ptree, self.mask_unary[vmask]).float()
                ttree = ptree.requires_grad_()
            return btree.detach(), bE.detach(), i

        def SEvolution(self, ptree, vmask=None):
            ptree, pE, i = self.GGS(ptree, vmask)
            #if self.training: print('number of iterations of GGS: ' + str(i))
            if vmask is None:
                bmask = torch.tensor([True]*ptree.size(0)).cuda()
                vmask = torch.tensor([True]*ptree.size(0)).cuda()
            else:
                bmask = torch.tensor([True]*(vmask.sum().item()),device=self.device)
            ovmask = vmask.clone()
            idx = torch.arange(ptree.size(-1))
            for i in range(1000):
                ntree = ptree[bmask].clone().requires_grad_()
                
                E = self.NetEnergy(ntree, vmask, False)
                E, = grad(E.sum(), ntree)
                E = E - (E*ntree.detach()).sum(-2).unsqueeze(-2)
                mE = E.reshape(E.size(0),-1)
                mindices = mE.argmax(-1)
                pntree = mE.new_zeros(mE.size())
                pntree[torch.arange(mE.size(0)), mindices] = 1
                pntree = pntree.view(E.size(0), E.size(1), E.size(2))
                pmask = (pntree.sum(1)!=0).unsqueeze(-2).repeat(1,pntree.size(-1),1)
                ntree = ntree.detach()
                ntree[pmask] = 0
                ntree[pmask] = pntree[pmask]

                pmask = torch.abs(ntree-ptree[bmask]).sum(dim=(-1,-2))==0
                if pmask.all()==True: break

                bmask[bmask.clone()] = ~pmask
                vmask[ovmask] = bmask

                ptree[bmask] = ntree[~pmask]

            #if self.training: print('number of iteration of MGS: ' + str(i), flush=True)
            return ptree

        def mean_field_variational_infernece(self):
            unary_potential = self.unary[:,1:,1:].transpose(-1,-2)
            idx = torch.arange(unary_potential.size(-1))
            unary_potential[:,idx, idx] = self.unary[:,1:,0]
            ptree = pargmax(unary_potential, self.mask_unary).float()
            ptree = self.SEvolution(ptree)
            ptree = ptree.requires_grad_()
            E = self.NetEnergy(ptree)
            if self.training: 
                E = self.NetEnergy(ptree)
                E, = grad(E.sum(), ptree, create_graph=True)
            else: 
                El, Es, Et = self.NetEnergy(ptree, bmask=None, grad=False, verse=True)
                E = El+Es+Et
                E, = grad(E.sum(), ptree)
            E[~self.mask_unary[:,1:,1:].transpose(-1,-2)] = -float('inf')
            E = E - torch.logsumexp(E, dim=-2).unsqueeze(-2)
            E[~self.mask_unary[:,1:,1:].transpose(-1,-2)] = 0
            nE = torch.zeros(E.size(0), E.size(1)+1, E.size(2)+1).cuda()
            nE[:,1:,1:] = E.transpose(-1,-2)
            nE[:,1:,0] = E[:,idx,idx]
            return nE 
            



        """
        def mean_field_variational_infernece(self, unary, mask_unary, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp):
                unary = unary.transpose(1,2)
                unary_potential = unary.clone()
                q_value = unary_potential.clone()
                if self.use_sib:
                    edge_node_sib_h, edge_node_sib_m = arc_sib
                if self.use_gp:
                    edge_node_gp_h, edge_node_gp_m, edge_node_gp_s = arc_gp
                for i in range(self.args.iterations):
                    q_value[~mask_unary.transpose(1,2)] = -float('inf')
                    idx = torch.arange(q_value.size(-1)-1)
                    temp = q_value[:,1:,1:]
                    temp[:,idx,idx] = q_value[:,0,1:]
                    ptemp = F.softmax(temp,1)
                    nq = torch.zeros(q_value.size()).cuda()
                    nq[:,1:,1:] = ptemp
                    nq[:,0,1:] = ptemp[:,idx,idx]
                    nq[:,idx+1,idx+1] = 0
                    nq[~mask_unary.transpose(1,2)] = 0
                    q_value = nq
                    if not self.training: q_value = q_value.detach().requires_grad_()
                    if self.use_sib:
                        edge_node_sib_h = torch.matmul(q_value.transpose(1,2),edge_node_sib_h)*torch.matmul(edge_node_sib_h, self.wh_sib)
                        edge_node_sib_m = torch.matmul(q_value.transpose(1,2),edge_node_sib_m)*torch.matmul(edge_node_sib_m, self.wd_sib)
                        layer_sib = self.attn_sib(edge_node_sib_m, edge_node_sib_h) * mask_unary
                        # keep (ma x mb x mc) -> (ma x mb x mb)
                        #layer_sib = 0.5 * (layer_sib + layer_sib.transpose(3,2))
                        idx = torch.arange(layer_sib.size(-1))
                        layer_sib = layer_sib.clone()
                        layer_sib[:,idx,idx] = 0

                        
                    else:
                        layer_sib = None
                    if self.use_cop:
                        layer_cop = self.trilinear_cop(edge_node_cop_h, edge_node_cop_m, edge_node_cop_h) * mask_cop
                        # keep (ma x mb x mc) -> (ma x mb x ma)
                        one_mask=torch.ones(layer_cop.shape[-2:]).cuda()
                        tril_mask=torch.tril(one_mask,-1)
                        triu_mask=torch.triu(one_mask,1)
                        layer_cop=layer_cop.transpose(1,2)
                        layer_cop = layer_cop-layer_cop*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_cop*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
                        layer_cop=layer_cop.transpose(1,2)
                    else:
                        layer_cop = None

                    if self.use_gp:
                        edge_node_gp_h, edge_node_gp_m, edge_node_gp_s = arc_gp
                        edge_node_gp_h = torch.matmul(q_value.transpose(1,2),edge_node_gp_h)*torch.matmul(edge_node_gp_h,self.wh_gp)
                        edge_node_gp_m = torch.matmul(q_value.transpose(1,2),edge_node_gp_m)*torch.matmul(edge_node_gp_m,self.wd_gp)
                        edge_node_gp_s = torch.matmul(q_value.transpose(1,2),edge_node_gp_s)*torch.matmul(edge_node_gp_s,self.ws_gp)
                        layer_gp = self.trilinear_gp(edge_node_gp_h, edge_node_gp_m, edge_node_gp_s) * mask_gp
                        idx = torch.arange(layer_gp.size(-1))
                        layer_gp = layer_gp.clone()
                        layer_gp[:,idx,idx,:] = 0
                        layer_gp[:,:,idx,idx] = 0
                        layer_gp[:,idx,:,idx] = 0

                    else:
                        layer_gp = None
                    E = (q_value*unary_potential).sum(dim=(-1,-2)) + layer_sib.sum(dim=(-1,-2)) + layer_gp.sum(dim=(-1,-2,-3))
                    if self.training: q_value, = grad(E.sum(), q_value, create_graph=True)
                    else: q_value, = grad(E.sum(), q_value)
                return q_value.transpose(1,2)
        """

        def from_mask_to_3d_mask(self,token_weights):
                root_weights = token_weights.clone()
                root_weights[:,0] = 0
                token_weights3D = token_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
                token_weights2D = root_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
                # abc -> ab,ac
                #token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
                #abc -> ab,cb
                if self.use_cop:
                        token_weights_cop = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * token_weights.unsqueeze(1).unsqueeze(1)
                        token_weights_cop[:,0,:,0] = 0
                else:
                        token_weights_cop=None
                #data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
                #abc -> ab, bc
                if self.use_gp:
                        token_weights_gp = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
                else:
                        token_weights_gp = None

                if self.use_sib:
                        #abc -> ca, ab
                        if self.use_gp:
                                token_weights_sib = token_weights_gp.clone()
                        else:
                                token_weights_sib = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
                else:
                        token_weights_sib = None
                return token_weights3D.transpose(1,2), token_weights_sib, token_weights_cop, token_weights_gp
