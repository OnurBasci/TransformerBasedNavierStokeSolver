from model.Transolver_Structured_Mesh_2D import Model as transolver_model
import torch
import torch.nn as nn


class SOL_Transolver_Structured_Mesh_2D(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85,
                 step = 1, #the number of scalar field by t (for example 2 for a 2d vector field)
                 look_ahead = 5 #the number of times that we call the transolver in the forward function
                 ):
        super(SOL_Transolver_Structured_Mesh_2D, self).__init__()

        self.transolver_model = transolver_model(space_dim=space_dim,
                                      n_layers=n_layers,
                                      n_hidden=n_hidden,
                                      dropout=dropout,
                                      n_head=n_head,
                                      Time_Input=Time_Input,
                                      act=act,
                                      mlp_ratio=mlp_ratio,
                                      fun_dim=fun_dim,
                                      out_dim=out_dim,
                                      slice_num=slice_num,
                                      ref=ref,
                                      unified_pos=unified_pos,
                                      H=H,
                                      W=W,
                                      )
        self.n = look_ahead
        self.step = step
    
    def forward(self, x, fx):
        for k in range(0,self.n):
            u = self.transolver_model(x, fx=fx)  # B , 4096 , 1
            #from frame (t: t+T) to (t+1: t+T+1) 
            fx = torch.cat((fx[..., self.step:], u), dim=-1)  #we concatante the prediction to the input
        return u


