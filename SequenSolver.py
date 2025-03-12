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
        #different approach for obtaining slices
        #1 get slice from the previous frames slice
        self.slice_projection =  nn.Linear(self.M, self.M) #MLP(self.M, self.M * mlp_ratio, self.M, n_layers=0, res=False, act=act)
        #2 get slice from fram slices of T previous frame
        self.temporal_slice_projection = MLP(self.T, self.T * mlp_ratio, 1) #nn.Linear(self.T, 1)
        #3 get slice from the learned tokens we need N MLP's of size C
        self.code = None

        self.token_to_slice_list = []
        for _ in range(self.N):
            self.token_to_slice_list.append(nn.Linear(self.C+2, 1).cuda())
        self.weight_projection = MLP(self.C+2, 64, 1)# nn.Linear(self.C+2, 1).cuda()

        self.softmax_slice = nn.Softmax(dim=-1)

        #for repeting attention
        self.ln_1 = nn.LayerNorm(self.dim)
        self.ln_2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim, self.dim * mlp_ratio, self.dim, n_layers=0, res=False, act=act)

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
        slice_weights_t = torch.from_numpy(np.zeros((self.B, self.N, self.M, self.T), dtype=np.float32)).cuda()
        for i in range(self.T):
            token = self.encoder.encode(spatial_pos, fx[:,:,i:i+1])
            token = token.reshape(B, self.M * self.C).contiguous() #[B, H, 1, M*C]
            tokens[:, :, i, :] = token.unsqueeze(1) #[B, H, T, M*C]
            #slice_weight = self.encoder.get_attention_slice().reshape(B, self.N, self.M, 1).contiguous()
            #slice_weights_t[:, :, :, i:i+1] = slice_weight
        
        #GET CORRECT SLICE WEIGHT
        #self.slice_weights_t = slice_weights_t

        #get the slice of the last frame
        #self.slice_weights = self.encoder.get_attention_slice()
        #print(f"last frame slice {self.slice_weights}")

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
        #if the slice weight is not given predict it
        
        if not use_gt:
            concat_total = torch.from_numpy(np.zeros((B, self.N, self.M, 34), dtype=np.float32)).cuda()
            for i in range(self.N):
                concat = torch.cat((code, spatial_pos.unsqueeze(1)[:, :, i:i+1, :].expand(-1,-1,16,-1)), dim=-1)
                concat_total[:,i,:,:] = concat
                #p_i = self.softmax_slice(self.weight_projection(concat).reshape(1, 1, 1, 16))
                #print(f"pi {p_i.shape}")
                #self.slice_weights[:,:,i,:] = p_i
                #p_i = weight_projection(code).reshape(1, 1, 1, 16) #probabilities of slice i 1, 1, 16, 1
                #self.slice_weights[:,:,i,:] = p_i

            self.slice_weights = self.softmax_slice(self.weight_projection(concat_total).permute(0, 3, 1, 2).contiguous()) #B N M 1 -> B 1 N M
        

        #print(f"slice weight all {self.slice_weights}")

        #call weigth projection one time over 

        #decode
        decoded = self.decode(code) #B, N, C
        output = self.mlp2(self.ln_3(decoded)) # B, N, C -> B, N, 1
        return output
    
    def solve_with_slice_learner(self, slice_learner_path, spatial_pos, fx, y, unified_pos=0, use_vorticity=0,use_previous_slice=False, learn_from_vort = False):
        #get sequential tokens
        B, _, _ = fx.shape
        tokens = torch.from_numpy(np.zeros((B, self.Head, self.T, self.M * self.C), dtype=np.float32)).cuda() # B H T M*C

        #get tokens
        for i in range(self.T):
            token = self.encoder.encode(spatial_pos, fx[:,:,i:i+1])
            token = token.reshape(B, self.M * self.C).contiguous() #[B, H, 1, M*C]
            tokens[:, :, i, :] = token.unsqueeze(1) #[B, H, T, M*C]

        #get the next token
        for i in range(self.layers):
            tokens = self.attention(self.ln_1(tokens)) + tokens
            tokens = self.mlp(self.ln_2(tokens)) + tokens
        #get the last toke as result
        code = tokens[:,:,-1:,].reshape(B, self.Head, self.M, self.C).contiguous() # B, H, M, C 
        self.code = code

        #get the slice weigth with the slice learner
        #load and freeze the model
        learn_slice_model = LearnSlice(unified_pos=unified_pos, use_vorticity=use_vorticity)
        learn_slice_model.load_state_dict(torch.load(slice_learner_path, weights_only=True), strict=False)

        learn_slice_model.eval()
        for param in learn_slice_model.parameters():
            param.requires_grad = False

        #get current slice weight from the previous slice weight
        if use_previous_slice:
            prev_slice_weight = self.get_last_slice_weight(spatial_pos, fx)
            token = code
            self.slice_weights = learn_slice_model.forward_previous_slice(prev_slice_weight, token)
        elif learn_from_vort:
            print(f"spatial pos {spatial_pos.shape}")
            self.slice_weights = learn_slice_model.forward_from_vorticity(spatial_pos, fx)
        else:
            #get the slice weight with the transolver article method
            self.slice_weights = learn_slice_model.get_slice_weight(code, spatial_pos, fx, use_vorticity)

        #slice weight gt
        self.encoder.encode(spatial_pos, y)
        slice_gt = self.encoder.get_attention_slice()

        #gt learned slice comp
        """print(f"slice gt {slice_gt.shape}")
        print(f"self.slice_weights {self.slice_weights.shape}")

        loss = 0
        for i in range(4096):
            #print(slice_gt[0,0,i,:])
            #print(self.slice_weights[0,0,i,:])
            #we compute the probabilities of all j for a given i
            loss += F.mse_loss(slice_gt[0,0,i,:], self.slice_weights[0,0,i,:])
        
        print(f"mse {loss}")"""

        #decode with token and slice
        decoded = self.decode(code) #B, N, C
        output = self.mlp2(self.ln_3(decoded)) # B, N, C -> B, N, 1
        return output


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
        #slice_weight = self.temporal_slice_projection(self.slice_weights_t)
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

def inference_example():
    transolver_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\codes\\Transolver\\model_weights\\encoder_ep50_head_3.pt"
    
    model = SequenSolver(transolver_path, T=10, H=64, W=64, M=16, C=32, B=1, layers=8).cuda()

    #spatial positional encoding
    h = 64
    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0).cuda()

    #get the data
    data_path = r"C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\data\\fno\\NavierStokes_V1e-5_N1200_T20\\NavierStokes_V1e-5_N1200_T20.mat"
    data = scio.loadmat(data_path)
    data = data['u'] #get the velocity component

    #adjust the data shape
    test_data = data[1, :, :, :10]
    test = test_data.reshape(1, -1, test_data.shape[-1]) #1, 4096, T
    test = torch.from_numpy(test)

    print(f"test size {test.shape}")

    #inference
    fx = test.cuda()
    print(f"fx {fx.shape}")
    print(f"pos {pos.shape}")

    pred_o = model(pos, fx)

    pred = pred_o.reshape(64,64)

    vort = tensor(pred, spatial('x,y'))

    # Create a StaggeredGrid for the velocity field
    f = CenteredGrid(math.tensor(vort), extrapolation.PERIODIC)
    plot(f)
    plt.show()

    #loss
    gt_o = data[1, :, :, 11]
    gt = gt_o.reshape(1, -1, gt_o.shape[-1]) #1, 4096, T
    gt = torch.from_numpy(gt).cuda()

    gt = gt.cuda()
    gt = gt.reshape(64, 64)
    
    vort_gt = tensor(gt, spatial('x, y'))

    f = CenteredGrid(math.tensor(vort_gt), extrapolation.PERIODIC)
    plot(f)
    plt.show()

    loss = torch.sum(torch.pow(pred_o.reshape(1, -1).cpu() - gt.reshape(1, -1).cpu(),2))
    print(loss)

def train(eval = False):

    batch_size = 1
    epochs = 10
    lr = 0.001
    weight_decay = 1e-5
    save_name = "tokenizer_ep10_sim10_2"

    ntrain = 10
    ntest = 2
    Tin = 10 #the size of the input sequence
    Tout = 10 #the number of frames to predict

    unified_pos = 1
    use_vorticity = 1

    #load data
    data_path = r"C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\data\\fno\\NavierStokes_V1e-5_N1200_T20\\NavierStokes_V1e-5_N1200_T20.mat"
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
    transolver_path = r"C:\Users\onurb\master\PRJ_4ID22_TP\Transolver\PDE-Solving-StandardBenchmark\sequential_checkpoints\encoder_ep20_head_1.pt"
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
        slice_learner_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\buff.pt"

        test_l2_full = 0
        with torch.no_grad():
            for i, (x, fx, yy) in enumerate(test_loader):
                print(f"i {i}")
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, Tout):
                    print(f"t {t}")
                    y = yy[..., t:t+1]
                    #im = model(x, fx, y, use_gt=True)
                    im = model.solve_with_slice_learner(slice_learner_path, x, fx, y, unified_pos=unified_pos, use_vorticity=use_vorticity, use_previous_slice=True, learn_from_vort=False)

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
            use_gt = True
            if ep > 5:
                model.freeze_attention()
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
                    im = model(x, fx, y, use_gt=True)
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
                test_l2_full = myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()    

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
    train(eval=True)
