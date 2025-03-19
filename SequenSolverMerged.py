"""In this script we try to learn slice weight and code tokens together"""
import argparse
import torch
import numpy as np
import torch.nn as nn
from model import Transolver_Structured_Mesh2D_Encoder
import scipy.io as scio
import matplotlib.pyplot as plt
#from phi.flow import *
from einops import rearrange
from utils.testloss import TestLoss
import os
#from SliceLearner import SliceLearner
#from LearnSlice import LearnSlice
import torch.nn.functional as F


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class SequenSolver(nn.Module):

    def __init__(self, transolver_path, T, W, H, M, C, B, mlp_ratio = 4, layers = 5, act='gelu',dropout=0.):
        super(SequenSolver, self).__init__()
        self.T = T #sequence number
        self.W = W #width
        self.H = H #height
        self.M = M #slices
        self.C = C #dimension for transolver
        self.N = H * W
        self.B = B
        self.dim = M * C #dimension for transolver
        self.scale = self.dim ** -0.5
        self.Head = 1 # Head is 1 for now
        self.layers = layers #number of repetion for the attention of sequences

        #load transolver model
        self.encoder = Transolver_Structured_Mesh2D_Encoder.Model(space_dim=2,
                                  n_layers=8,
                                  n_hidden=32,
                                  dropout=0.0,
                                  n_head=1,
                                  slice_num=16,
                                  Time_Input=False,
                                  fun_dim=1,
                                  out_dim=1,
                                  unified_pos=1,
                                  H=64, W=64).cuda()
        
        self.encoder.load_state_dict(torch.load(transolver_path, weights_only=True), strict=False)

        #self.slice_learner = SliceLearner(space_dim=1, n_hidden=64, fun_dim=T, unified_pos=1, H=64, W=64, slice_num=16)

        #freeze the model since it is not included in the trainig
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)
        self.softmax_attention = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.slice_weights = torch.from_numpy(np.zeros((B, 1, self.N, self.M), dtype=np.float32)).cuda()
        self.slice_weights_t = 0    

        #for repeting attention
        self.ln_1 = nn.LayerNorm(self.dim)
        self.ln_2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim, self.dim * mlp_ratio, self.dim, n_layers=0, res=False, act=act)

        #for slice learning
        self.fundemental = 74 #ref*ref + Tin
        n_hidden = 256 #feature vector (with temporal analysis)
        self.preprocess = MLP(self.fundemental, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act).cuda()
        kernel = 3
        self.in_project_x = nn.Conv2d(n_hidden, n_hidden, kernel, 1, kernel // 2).cuda()
        self.softmax_vort = nn.Softmax(dim=-1).cuda()
        self.concatenated = n_hidden + (self.M * self.C)
        self.in_project_slice = MLP(self.concatenated, self.concatenated//2, self.M).cuda() #mlp for concatenated vector vort + code
        self.temperature = nn.Parameter(torch.ones([1, 1, 1, 1]) * 0.5).cuda()

        #for decoding
        self.ln_3 = nn.LayerNorm(self.C)
        self.mlp2 = nn.Linear(self.C, 1) #to get the output N*C -> N*1

    def forward(self, spatial_pos, fx, y, use_gt=True):
        """
        spatial_pos: spatial encoding
        fx: vorticity of T previous sequence so [B, W* H, T]
        """
        B, _, _ = fx.shape
        tokens = torch.from_numpy(np.zeros((B, self.Head, self.T, self.M * self.C), dtype=np.float32)).cuda() # B H T M*C

        #get tokens
        for i in range(self.T):
            token = self.encoder.encode(spatial_pos, fx[:,:,i:i+1])
            token = token.reshape(B, self.M * self.C).contiguous() #[B, H, 1, M*C]
            tokens[:, :, i, :] = token.unsqueeze(1) #[B, H, T, M*C]

        #get the targeted slice
        if use_gt:
            self.encoder.encode(spatial_pos, y)
            target_slice = self.encoder.get_attention_slice()
            self.slice_weights = target_slice

        #learn slice by like in the article
        #self.slice_weights = self.slice_learner(spatial_pos, fx)

        #attention
        for i in range(self.layers):
            tokens = self.attention(self.ln_1(tokens)) + tokens
            tokens = self.mlp(self.ln_2(tokens)) + tokens
        #get the last toke as result
        code = tokens[:,:,-1:,].reshape(B, self.Head, self.M, self.C).contiguous() # B, H, M, C 
        self.code = code

        #3 learn slice from token
        self.slice_weights = self.forward_slice(spatial_pos, fx, code)
        

        #print(f"slice weight all {self.slice_weights}")

        #call weigth projection one time over 

        #decode
        decoded = self.decode(code) #B, N, C
        output = self.mlp2(self.ln_3(decoded)) # B, N, C -> B, N, 1
        return output


    def forward_slice(self, x, fx, code):
        """
        This function takes the previous position, vorticity information and the token representing the predicted frames. And predicts the
        slice weights
        x: positional information [B, W*H, ref*ref]
        fx: vorticity information [B, W*H, Tin]
        code: the predicted token [B, 1, M, C]
        """

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            #if the code is added before
            """code = code.reshape(code.shape[0], 1 ,code.shape[2]*code.shape[3]).contiguous() # B, 1, M, C -> B, 1, M * C
            code = code.expand(-1, fx.shape[1], -1) #B, 1, M*C -> B, W*H, M*C
            fx = torch.cat((fx, code), -1)"""
            fx = self.preprocess(fx) #the size of fx becomes B * N * C where C is the size of the embedded token from the article
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        #learn slices
        B, N, C = fx.shape
        fx = fx.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W
        x_mid = self.in_project_x(fx).permute(0, 2, 3, 1).contiguous().reshape(B, N, 1, C) \
            .permute(0, 2, 1, 3).contiguous()  # B 1 N C
        
        #concatenate code withe the output of the vorticity analysis
        code = code.reshape(code.shape[0], 1 , 1, code.shape[2]*code.shape[3]).contiguous() # B, 1, M, C -> B, 1, 1, M * C
        code = self.z_score_normalization(code)
        code = code.expand(-1, -1, x_mid.shape[2], -1)
        x_mid = self.z_score_normalization(x_mid)
        x_mid = torch.cat((x_mid, code), -1)
        slice_weights = self.softmax_vort(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G

        #print(f"forward slice output {slice_weights.shape}")

        return slice_weights

    def get_code(self, spatial_pos, fx, y):
        B, _, _ = fx.shape
        tokens = torch.from_numpy(np.zeros((B, self.Head, self.T, self.M * self.C), dtype=np.float32)).cuda() # B H T M*C

        #get tokens
        for i in range(self.T):
            token = self.encoder.encode(spatial_pos, fx[:,:,i:i+1])
            token = token.reshape(B, self.M * self.C).contiguous() #[B, H, 1, M*C]
            tokens[:, :, i, :] = token.unsqueeze(1) #[B, H, T, M*C]

        for i in range(self.layers):
            tokens = self.attention(self.ln_1(tokens)) + tokens
            tokens = self.mlp(self.ln_2(tokens)) + tokens
        #get the last toke as result
        code = tokens[:,:,-1:,].reshape(B, self.Head, self.M, self.C).contiguous() # B, H, M, C 

        return code
    
    def get_last_slice_weight(self, spatial_pos, fx):

        #call the encoder of the last frame to generate the slice
        self.encoder.encode(spatial_pos, fx[:,:,-1:])
        return self.encoder.get_attention_slice()


    def attention(self, tokens):
        #print(f"attention input {tokens.shape}")
        # Attention among sequential tokens
        q_slice_token = self.to_q(tokens)
        k_slice_token = self.to_k(tokens)
        v_slice_token = self.to_v(tokens)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale

        #apply masking
        mask = torch.tril(torch.ones(self.T, self.T, device=dots.device))
        dots = dots.masked_fill(mask==0, float('-inf'))

        attn = self.softmax_attention(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        #print(f"attention output {out_slice_token.shape}")
        return out_slice_token


    def decode(self, code):
        #print(f"decode input {tokens.shape}") #B, H, T, M*C
        #take the last token representing the new slice, and split by slices
        #B = tokens.shape[0]
        #code = tokens[:,:,-1:,].reshape(B, self.Head, self.M, self.C).contiguous() # B, H, M, C 
        #decoded1 = self.encoder.decode(code) #for now we used the learned slice

        #learn slice
        slice_weight = self.slice_weights # self.slice_projection(self.slice_weights)
        #print(f"self.slice_weights_t {self.slice_weights_t.shape}")
        #print(f"slice_weight {slice_weight.shape}")
        #slice_weight = slice_weight.reshape(B, 1, self.N, self.M).contiguous()
        #print(f"slice_weight 2 {slice_weight.shape}")

        #self.encoder.set_attention_slice(slice_weight)
        #decoded2 = self.encoder.decode(code)
        
        #decode with slice
        decoded = torch.einsum("bhgc,bhng->bhnc", code, slice_weight)
        decoded = rearrange(decoded, 'b h n d -> b n (h d)')

        #print(f"decoded output {decoded.shape}")
        return decoded

    def z_score_normalization(self, x):
        mean = torch.mean(x)
        std = torch.std(x, unbiased=False)  # Use unbiased=False for population std

        return (x - mean) / (std + 1e-8)  # Avoid division by zero

    def freeze_attention(self):
        self.to_q.eval()
        self.to_k.eval()
        self.to_v.eval()
        self.softmax_attention.eval()

        self.mlp.eval()

        self.ln_1.eval()
        self.ln_2.eval()

        # Freeze the parameters
        for param in self.to_q.parameters():
            param.requires_grad = False

        for param in self.to_k.parameters():
            param.requires_grad = False

        for param in self.to_v.parameters():
            param.requires_grad = False

        for param in self.mlp.parameters():
            param.requires_grad = False

        for param in self.ln_1.parameters():
            param.requires_grad = False

        for param in self.ln_2.parameters():
            param.requires_grad = False
        

def get_grid(batchsize=1):
        size_x, size_y = 64, 64
        ref = 8
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, ref * ref).contiguous()
        
        return pos    


def train(eval = False):

    print(f"args {args.eval}")

    batch_size = 1
    epochs = args.epochs
    lr = 0.001
    weight_decay = 1e-5
    save_name = args.save_name

    ntrain = args.sim_num
    ntest = 10
    Tin = 10 #the size of the input sequence
    Tout = 10 #the number of frames to predict

    unified_pos = 1

    #load data
    data_path = r"./data/fno/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
    data = scio.loadmat(data_path)
    data = data['u'] #get the velocity component

    print(f"data {data.shape}")

    train_a = data[:ntrain, :, :, :Tin]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)

    train_u = data[:ntrain,:,:, Tin:Tin +Tout]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data[-ntest:, :, :, :Tin]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)

    test_u = data[-ntest:,:,:, Tin:Tin + Tout]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    print(f"train a {train_a.shape}")
    print(f"train u {train_u.shape}")

    #spatial positional encoding
    h = 64
    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0).cuda()
    if unified_pos:
        uni_grid = get_grid()
        pos=uni_grid.repeat(pos.shape[0], 1, 1, 1).reshape(pos.shape[0], 64 * 64, 8 * 8)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    #define loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=batch_size, shuffle=False)
    
    #get model
    transolver_path = r"./sequential_checkpoints/encoder_ep20_head_1.pt"
    model = SequenSolver(transolver_path, T=Tin, H=64, W=64, M=16, C=32, B=batch_size, layers=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    
    myloss = TestLoss(size_average=False)

    #train
    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./sequential_checkpoints/" + save_name + ".pt", weights_only=True), strict=False)
        model.eval()

        #slice_learner_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\slice_ep1_sim20_unified_vort.pt"
        #slice_learner_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\buff.pt"
        #slice_learner_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\slice_vorticity_code_ep10_sim200_b1.pt"
        #slice_learner_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\slice_ep10_sim75_unified_vort.pt"

        test_l2_full = 0
        with torch.no_grad():
            for i, (x, fx, yy) in enumerate(test_loader):
                print(f"i {i}")
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, Tout):
                    print(f"t {t}")
                    y = yy[..., t:t+1]
                    im = model(x, fx, y, use_gt=False)
                    #im = model.solve_with_slice_learner(slice_learner_path, x, fx, y, unified_pos=unified_pos, use_vorticity=use_vorticity, use_previous_slice=False, learn_from_vort=False, use_code_for_vorticity=use_code_for_vorticity)

                    fx = torch.cat((fx[..., 1:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print(test_l2_full / ntest)

    else:
        for ep in range(epochs):
            model.train()
            train_l2_step = 0
            train_l2_full = 0
            use_gt = False
            print(f"train loader size {len(train_loader)}")
            for i, (x, fx, yy) in enumerate(train_loader):
                print(i)
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                bsz = x.shape[0]

                #print(f"x {x.shape}, fx {fx.shape}, yy {yy.shape}")
                for t in range(0, Tout):
                    y = yy[..., t:t+1]
                    im = model(x, fx, y, use_gt=use_gt)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    # we add ground truth to the fx
                    fx = torch.cat((fx[..., 1:], y), dim=-1)
                
                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            model.eval()

            test_l2_step = 0
            test_l2_full = 0

            with torch.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                    bsz = x.shape[0]

                    for t in range(0, Tout):
                        y = yy[..., t:t+1]
                        im = model(x, fx, y, use_gt=True)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        fx = torch.cat((fx[..., 1:], im), dim=-1)

                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()    

                print(
                    "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(
                        ep, train_l2_step / ntrain / Tin, train_l2_full / ntrain, test_l2_step / ntest / Tin,
                            test_l2_full / ntest))
                
        if not os.path.exists('./sequential_checkpoints'):
            os.makedirs('./sequential_checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))
                

if __name__ == "__main__":
    #inference_example()
    parser = argparse.ArgumentParser('Training Transformer')

    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_name', type=str, default='buff')
    parser.add_argument('--sim_num', type=int, default=10)

    args = parser.parse_args()

    train(eval=args.eval)
