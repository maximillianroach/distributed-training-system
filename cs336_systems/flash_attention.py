import torch
from config import TILE_SIZE_KV, TILE_SIZE_Q
import math
from einops import einsum, rearrange, reduce

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        lenQ = Q.shape[-2]
        lenKV = K.shape[-2]
        d = Q.shape[-1]
        batch_size = Q.shape[-3]
        scale = 1 / math.sqrt(d)

        # define the number of Q blocks
        Tq = math.ceil(lenQ / TILE_SIZE_Q)
        # define the number of KV blocks
        Tkv = math.ceil(lenKV / TILE_SIZE_KV)

        # create O and L in global memory 
        O = torch.empty((batch_size, lenQ, d))
        L = torch.empty((batch_size, lenQ))


        for batch in range(batch_size):
            for i in range(Tq):
                # get the ith block of Q
                q_start = i * TILE_SIZE_Q
                q_end= (i + 1) * TILE_SIZE_Q

                Qi = Q[batch, q_start: q_end] # Bq x d
                Oi0 = torch.zeros((TILE_SIZE_Q, d)) # Bq x d
                Oi_prev = Oi0

                li0 = torch.zeros((TILE_SIZE_Q,)) # Bq
                li_prev = li0

                mi0 = torch.full((TILE_SIZE_Q, ), -torch.inf) # Bq
                mi_prev = mi0

                for j in range(Tkv):
                    kv_start = j * TILE_SIZE_KV
                    kv_end = (j + 1) * TILE_SIZE_KV

                    Kj = K[batch, kv_start: kv_end] # Bk x d
                    Vj = V[batch, kv_start: kv_end] # Bk x d

                    # compute block pre-soft expression
                    Sij = einsum(Qi, Kj, 'Bq d, Bk d -> Bq Bk') * scale # Bq x Bk

                    Sij_maxes = torch.max(Sij, dim=-1).values # Bq

                    # stack Sij_maxes and mij so we're finding pairwise maxes
                    mij = torch.max(torch.stack((Sij_maxes, mi_prev), dim=0), dim=0).values # Bq

                    Pij = torch.exp(Sij - mij.unsqueeze(-1)) # Bq x Bk

                    lij = torch.exp(mi_prev - mij) * li_prev + torch.sum(Pij, dim=-1) # Bq

                    PV = einsum(Pij, Vj, 'Bq Bk, Bk d -> Bq d') # Bq x d

                    Oij = torch.exp(mi_prev - mij).unsqueeze(-1) * Oi_prev + PV # Bq x d

                    mi_prev = mij
                    li_prev = lij
                    Oi_prev = Oij

                true_Oi = Oi_prev / li_prev.unsqueeze(-1)
                Li = mi_prev + torch.log(li_prev)

                O[batch, q_start: q_end] = true_Oi
                L[batch, q_start: q_end] = Li

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

















    def backward(ctx):
        raise NotImplementedError