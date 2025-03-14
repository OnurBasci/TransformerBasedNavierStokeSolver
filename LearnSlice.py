import torch
import numpy as np
import torch.nn as nn
from SequenSolver import SequenSolver
import scipy.io as scio
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


class LearnSlice(nn.Module):
    """
    this class try to predict slice weight from the learned token w_i,j = f(z_i, p_i)
    """
    def __init__(self, unified_pos=0, use_vorticity = 0, use_code_for_vorticity = False):
        super(LearnSlice, self).__init__()
        self.C = 32
        self.N = 4096
        self.M=16
        self.unified_pos = unified_pos
        self.use_vorticity = use_vorticity
        if self.unified_pos:
            self.pos = 64
            if use_vorticity:
                self.pos = 74
        else:
            self.pos = 2
            if use_vorticity:
                self.pos = 12
        self.weight_projection = MLP(self.C+self.pos, 64, 1).cuda()# nn.Linear(self.C+2, 1).cuda()
        self.act = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.unified_pos = unified_pos

        #modules to predict from previous slice
        self.weight_projection_form_slice = MLP(self.M + self.M * self.C, (self.M + self.M * self.C)*4, self.M, 1).cuda()
        #self.weight_projection_form_slice = MLP(self.M, self.M*4, self.M, 1).cuda()

        #modules to predict from vorticity + pos
        n_hidden = 256
        act='gelu'
        self.H = 64
        self.W = 64
        self.fundemental = 10
        if self.unified_pos:
            self.fundemental += 64
        else:
            self.fundemental += 2
        if use_code_for_vorticity:
            self.concatenated = n_hidden + (self.M * self.C)
        else:
            self.concatenated = n_hidden
        self.preprocess = MLP(self.fundemental, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act).cuda()
        kernel = 3
        self.in_project_x = nn.Conv2d(n_hidden, n_hidden, kernel, 1, kernel // 2).cuda()
        self.softmax_vort = nn.Softmax(dim=-1).cuda()
        self.in_project_slice = MLP(self.concatenated, self.concatenated//2, self.M).cuda()
        #self.in_project_slice = nn.Linear(self.concatenated, self.M).cuda()
        self.temperature = nn.Parameter(torch.ones([1, 1, 1, 1]) * 0.5).cuda()

        #modules to predict from vorticity + pos seperate
        self.preprocess_seperate = MLP(106, 212, 64, n_layers=0, res=False, act=act).cuda()
        kernel = 3
        self.in_project_x_seperate = nn.Conv2d(64, 64, kernel, 1, kernel // 2).cuda()
        self.softmax_vort_seperate = nn.Softmax(dim=-1).cuda()
        self.in_project_slice_seperate = nn.Linear(64, 1).cuda()
        self.temperature_seperate = nn.Parameter(torch.ones([1, 1, 1, 1]) * 0.5).cuda()

    
    def forward(self, code, spatial_pos):
        """
        code: [M, 32]
        spatial pos: [1,2] 
        output: [1, M] w_ij
        """
        #input is concatenated code and spatial pos
        M=code.shape[0]
        spatial_pos = spatial_pos.expand(16,-1)
        code = torch.cat((code, spatial_pos), dim=-1)
        w = self.softmax(self.weight_projection(code)).reshape(1,M).contiguous()
        #w = self.act(w)
        return w
    
    def forward_all(self, concatenated):
        """
        concatenated: [N, M, 34]
        output: [1,N, M]
        """
        #input is concatenated code and spatial pos
        M=concatenated.shape[1]
        w = self.softmax(self.weight_projection(concatenated)).permute(2,0,1).contiguous()
        #w = self.act(w)
        return w
    
    def forward_previous_slice(self, prev_slice_weight, token):
        """
        in this function we try to predict the next timestep slice from the previous one and token
        prev_slice_weight: [1, 1, 4096, 16]
        token: [1, 1, 16, 32]
        """
        flatten = token.reshape(1, 1, 1, token.shape[2]*token.shape[3]).contiguous() #1, 1, 16 , 32 -> 1, 1, 1 512
        flatten = flatten.expand(-1,-1,prev_slice_weight.shape[2], -1)
        concatenated = torch.cat((prev_slice_weight, flatten), -1)
        return self.weight_projection_form_slice(concatenated)

    def get_slice_weight(self, tokens, spatial_pos, fx, use_vorticity=0):
        """
        this function given the slice tokens and spatial position returns the slice weight
        tokens: [1, 1, 16, 32]
        spatial pos: [1, 4096, 2]
        """
        slice_weight = torch.from_numpy(np.zeros((1, 1, self.N, self.M), dtype=np.float32)).cuda()
        for i in range(self.N):
            code_i = tokens[0,0]
            #spatial_pos_i = spatial_pos[0, i:i+1,:]
            x = spatial_pos[0,i:i+1,:]
            if use_vorticity:
                x = torch.cat((spatial_pos, fx), -1)
                x = x[0,i:i+1,:]
            w_i = self.forward(code_i, x)
            slice_weight[:,:, i, :] = w_i.reshape(1,1, w_i.shape[0], w_i.shape[1]).contiguous()
        
        return slice_weight
    
    def forward_from_vorticity(self, x, fx, code = None):
        """
        x: [1, 4096, 64]
        fx: [1, 4096, 10]
        code: [1, 1, 16, 32]
        """

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            """if code is not None:
                code = code.reshape(code.shape[0], 1 ,code.shape[2]*code.shape[3]).contiguous() # B, 1, M, C -> B, 1, M * C
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
        if code is not None:
            code = code.reshape(code.shape[0], 1 , 1, code.shape[2]*code.shape[3]).contiguous() # B, 1, M, C -> B, 1, 1, M * C
            code = self.z_score_normalization(code)
            code = code.expand(-1, -1, x_mid.shape[2], -1)
            x_mid = self.z_score_normalization(x_mid)
            x_mid = torch.cat((x_mid, code), -1)
        slice_weights = self.softmax_vort(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G

        return slice_weights
    
    def z_score_normalization(self, x):
        mean = torch.mean(x)
        std = torch.std(x, unbiased=False)  # Use unbiased=False for population std

        return (x - mean) / (std + 1e-8)  # Avoid division by zero
        

    
    def forward_from_vorticity_seperate(self, x, fx, code):
        """
        x: [1, 4096, 64]
        fx: [1, 4096, 10]
        code: [1, 1, 16, 32]
        """

        slice_weights = None
        for i in range(self.M):
            if fx is not None:
                fx_i = torch.cat((x, fx), -1)
                
                code_i = code[:,:,i,:] # B, 1, M, C -> B, 1, C
                code_i = code_i.expand(-1, fx.shape[1], -1) #B, 1, M*C -> B, W*H, M*C
                fx_i = torch.cat((fx_i, code_i), -1)
                fx_i = self.preprocess_seperate(fx_i) 

                #learn slices
                B, N, C = fx_i.shape
                fx_i = fx_i.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W
                x_mid = self.in_project_x_seperate(fx_i).permute(0, 2, 3, 1).contiguous().reshape(B, N, 1, C) \
                    .permute(0, 2, 1, 3).contiguous()  # B H N G
                slice_weights_i = self.softmax_vort_seperate(
                    self.in_project_slice_seperate(x_mid) / torch.clamp(self.temperature_seperate, min=0.1, max=5))  # B H N G
                
                if slice_weights == None:
                    slice_weights = slice_weights_i
                else:
                    slice_weights = torch.cat((slice_weights, slice_weights_i), dim=-1)

        return slice_weights


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

def buff():
    transolver_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\codes\\Transolver\\model_weights\\encoder_ep20_head_1.pt"
    model = SequenSolver(transolver_path, T=10, H=64, W=64, M=16, C=32, B=1, layers=8).cuda()
    
    SequenSolver_path = "C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\sequential_checkpoints\\tokenizer_ep10_sim10_2.pt"
    model.load_state_dict(torch.load(SequenSolver_path, weights_only=True), strict=False)

    SliceLearner = LearnSlice()

    #saptial positioning
    h = 64
    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0).cuda()

    #get the data
    data_path = "/tempory/TransformerBasedNavierStokeSolver/data/fno/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
    data = scio.loadmat(data_path)
    data = data['u'] #get the velocity component
    data = data.reshape(data.shape[0], -1, data.shape[-1])
    print(data.shape)

    #adjust the data shape
    test = data[1:2, :, :10]
    y = data[1:2,:,10:11]
    test = torch.from_numpy(test)
    y = torch.from_numpy(y).cuda()

    print(f"test size {test.shape}")

    #inference
    fx = test.cuda()

    code = model.get_code(pos, fx, y)

    print(f"get code {code.shape}")
    print(f"pos {pos.shape}")
    i = 0 #pos
    j = 0 #slice
    w_ij = SliceLearner(code[0,0], pos[0,i:i+1,:])

    #get the original slice
    model.encoder.encode(pos, y)
    target_slice = model.encoder.get_attention_slice()
    print(f"w_ij {w_ij.shape}")
    print(f"target slice {target_slice.shape}")
    print(f"target slice {target_slice[0,0:1,i].shape}")
    
def load_data(ntrain, ntest, Tin, Tout):
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

    return train_a, train_u, test_a, test_u

def encode_spatial_positionning(ntrain, ntest, unified_pos):
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

    print(f"pos {pos.shape}")

    return pos_train, pos_test, pos

def train(eval = False):

    #DEFINE PARAMETERS
    N = 4096
    M = 16

    batch_size = 1
    epochs = 1
    lr = 0.001
    weight_decay = 1e-5
    #save_name = "slice_ep2_sim20"
    #save_name = "slice_ep2_sim20_unified"
    #save_name = "slice_ep1_sim20_unified_vort"
    #save_name = "slice_ep1_sim20_unified_vort2"
    #save_name = "slice_learner_unified"
    #save_name = "slice_ep1_sim50_unified_vort_encoder_ep50"
    #save_name = "slice_ep4_sim50_unified_vort"
    save_name = "buff"
    #save_name = "slice_ep5_sim50_unified_vort"
    #save_name = "slice_ep10_sim75_unified_vort"

    ntrain = 10
    ntest = 10
    Tin = 10 #the size of the input sequence
    Tout = 10 #the number of frames to predict

    unified_pos = 1
    use_vorticity = 1

    #LOAD DATA
    train_a, train_u, test_a, test_u = load_data(ntrain, ntest, Tin, Tout)

    print(f"train a {train_a.shape}")
    print(f"train u {train_u.shape}")

    #SPATIAL POSITION ENCODING
    pos_train, pos_test, pos = encode_spatial_positionning(ntrain, ntest, unified_pos)

    #DEFINE LOADERS
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=batch_size, shuffle=False)
    
    #GET MODEL
    transolver_path = "./sequential_checkpoints/encoder_ep20_head_1.pt"
    sequen_solver = SequenSolver(transolver_path, T=10, H=64, W=64, M=16, C=32, B=1, layers=8).cuda()
    
    SequenSolver_path = "./sequential_checkpoints/tokenizer_ep10_sim10_2.pt"
    sequen_solver.load_state_dict(torch.load(SequenSolver_path, weights_only=True), strict=False)
    
    #freeze sequenSolver
    sequen_solver.eval()
    for param in sequen_solver.parameters():
        param.requires_grad = False

    model = LearnSlice(unified_pos=unified_pos, use_vorticity=use_vorticity)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #for each epoch we have train loader * N * M
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=(len(train_loader)*Tout))
    
    mse_loss = nn.MSELoss()

    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./sequential_checkpoints/" + save_name + ".pt", weights_only=True), strict=False)
        model.eval()
        print(f"TEST")
        with torch.no_grad():
            loss_overall = []
            mean_loss_slice = 0
            for x, fx, yy in test_loader:
                loss_sim = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                original_x = x
                loss_t = 0
                loss_slice = 0
                for t in range(0, Tout):
                    print(t)
                    y = yy[..., t:t+1]

                    #get the original slice
                    target_code = sequen_solver.encoder.encode(pos, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()

                    #get the code from sequen solver
                    code = sequen_solver.get_code(pos, fx, y)

                    #GET AND SHOW THE PREDICTED SLICE
                    predicted_slice = model.get_slice_weight(code, original_x, fx, use_vorticity=use_vorticity)

                    loss_slice += mse_loss(predicted_slice, target_slice)

                    #LOSS EVALUATION FOR EACH ROW
                    #for each position and slice we predict the weight
                    loss = 0
                    """
                    for i in range(N):
                        #we compute the probabilities of all j for a given i
                        x = pos[0,i:i+1,:]
                        if use_vorticity:
                            x = torch.cat((pos, fx), -1)
                            x = x[0,i:i+1,:]
                        
                        w_i = model(code[0,0], x)
                        loss += F.mse_loss(w_i, target_slice[0,0:1,i])"""
                        
                    sequen_solver.slice_weights = target_slice
                    decoded = sequen_solver.decode(code)
                    pred = sequen_solver.mlp2(sequen_solver.ln_3(decoded))
                    fx = torch.cat((fx[..., 1:], pred), dim=-1)

                    #print(loss)
                    loss_t += loss
                loss_sim = loss_t/Tout
                loss_overall.append(loss_sim)
                print(f"diff {loss_slice}")
                mean_loss_slice += loss_slice
            mean_loss = sum(loss_overall)/len(loss_overall)
            mean_loss_slice = mean_loss_slice/len(loss_overall)
            print(f"mean_loss {mean_loss}")
            print(f"mean diff {mean_loss_slice}")
            print(f"loss by sim {loss_overall}")
    else:
        losses = []
        for ep in range(epochs):
            print(f"ep: {ep}")
            model.train()
            loss_epoch = 0
            print(f"train loader size {len(train_loader)}")
            for i, (x, fx, yy) in enumerate(train_loader):
                print(f"i {i}")
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                bsz = x.shape[0]

                #print(f"x {x.shape}, fx {fx.shape}, yy {yy.shape}")
                loss_t = 0
                for t in range(0, Tout):
                    print(f"t {t}")
                    y = yy[..., t:t+1]

                    #get the original slice
                    sequen_solver.encoder.encode(pos, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()

                    #get the code from sequen solver
                    code = sequen_solver.get_code(pos, fx, y)

                    #for each position and slice we predict the weight
                    loss = torch.tensor(0.0, device="cuda")
                    #concat_total = torch.from_numpy(np.zeros((N, M, 34), dtype=np.float32)).cuda()  #[N, M, 34]
                    for i in range(N):
                        #loss total
                        #concat = torch.cat((code[0,0], pos[0,i:i+1,:].expand(16,-1)), dim=-1)
                        #concat_total[i,:,:] = concat
                        #loss by geometry
                        #concatenate vorticity and position
                        x = pos[0,i:i+1,:]
                        if use_vorticity:
                            x = torch.cat((pos, fx), -1)
                            x = x[0,i:i+1,:]
                        w_i = model(code[0,0], x)
                        loss += F.mse_loss(w_i, target_slice[0,0:1,i])
                        del x, w_i
                    #update fx
                    fx = torch.cat((fx[..., 1:], y), dim=-1)
                    #loss total
                    #w = model.forward_all(concat_total).unsqueeze(0)
                    #loss += F.mse_loss(w, target_slice)
                    #print(f"w.shape {w.shape}")
                    #print(f"target {target_slice.shape}")
                    loss_t += loss
                    losses.append(loss.item())
                    print(f"train loss {loss}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                loss_sim = loss_t/Tout
                loss_epoch += loss_sim
                torch.cuda.empty_cache()
                print(f"mean loss sim {loss_sim}")
            loss_epoch /= len(train_loader)
            
            print(f"loss epoch {ep}: {loss_epoch}")
            losses.append(loss_epoch.item())
            print(losses)
            if not os.path.exists('./sequential_checkpoints'):
                os.makedirs('./sequential_checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))

            model.eval()
            print(f"TEST")
            with torch.no_grad():
                overall_loss = 0
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                    bsz = x.shape[0]

                    loss_t = 0
                    for t in range(0, Tout):
                        print(t)
                        y = yy[..., t:t+1]

                        #get the original slice
                        sequen_solver.encoder.encode(pos, y)
                        target_slice = sequen_solver.encoder.get_attention_slice()

                        #get the code from sequen solver
                        code = sequen_solver.get_code(pos, fx, y)

                        #for each position and slice we predict the weight
                        loss = 0
                        for i in range(N):
                            #we compute the probabilities of all j for a given i
                            x = pos[0,i:i+1,:]
                            if use_vorticity:
                                x = torch.cat((pos, fx), -1)
                                x = x[0,i:i+1,:]
                            w_i = model(code[0,0], x)
                            loss += F.mse_loss(w_i, target_slice[0,0:1,i])
                            #we compute the probabilities of all i andf j
                            """for j in range(M):
                                code_j = code[0,0,j,:]
                                w_ij = model(code_j, pos_i)[0]
                                print(f"w_ij {w_ij}")
                                print(f"target_slice {target_slice[0,0,i,j]}")
                                loss += F.mse_loss(w_ij, target_slice[0,0,i,j])"""
                        sequen_solver.slice_weights = target_slice
                        decoded = sequen_solver.decode(code)
                        pred = sequen_solver.mlp2(sequen_solver.ln_3(decoded))
                        fx = torch.cat((fx[..., 1:], pred), dim=-1)
                        loss_t += loss
                        print(f"test loss {loss}")
                    loss_sim = loss_t/Tout
                    overall_loss += loss_sim
                    print(f"mean loss of a simulation {loss_sim}")
                overall_loss /= len(train_loader)
                print(f"overall loss {overall_loss}")
        
def train_from_previous(eval=False):
    N = 4096
    M = 16

    batch_size = 1
    epochs = 50
    lr = 0.001
    weight_decay = 1e-5
    save_name = "buff"
    #save_name = "buff2"
    #save_name = "slice_ep2_sim20"
    #save_name = "slice_ep2_sim20_unified"
    #save_name = "slice_ep1_sim20_unified_vort"
    #save_name = "slice_ep1_sim20_unified_vort2"
    #save_name = "slice_learner_unified"
    #save_name = "slice_ep1_sim50_unified_vort_encoder_ep50"
    #save_name = "slice_ep4_sim50_unified_vort"
    #save_name = "buff"
    save_name = "slice_ep5_sim50_unified_vort"

    unified_pos = 1
    use_vorticity = 1

    ntrain = 50
    ntest = 10
    Tin = 10 #the size of the input sequence
    Tout = 10 #the number of frames to predict

    train_a, train_u, test_a, test_u = load_data(ntrain, ntest, Tin, Tout)

    #spatial positional encoding
    pos_train, pos_test, pos = encode_spatial_positionning(ntrain, ntest, unified_pos)

    #define loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=batch_size, shuffle=False)
    
    #get model
    transolver_path = "./sequential_checkpoints/encoder_ep20_head_1.pt"
    sequen_solver = SequenSolver(transolver_path, T=10, H=64, W=64, M=16, C=32, B=1, layers=8).cuda()
    
    SequenSolver_path = "./sequential_checkpoints/tokenizer_ep10_sim10_2.pt"
    sequen_solver.load_state_dict(torch.load(SequenSolver_path, weights_only=True), strict=False)
    
    #freeze sequenSolver
    sequen_solver.eval()
    for param in sequen_solver.parameters():
        param.requires_grad = False

    model = LearnSlice(unified_pos=unified_pos, use_vorticity=use_vorticity)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #for each epoch we have train loader * N * M
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=(len(train_loader)))
    
    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./sequential_checkpoints/" + save_name + ".pt", weights_only=True), strict=False)
        model.eval()
        print(f"TEST")
        with torch.no_grad():
            mean_loss = 0
            mean_diff = 0
            for x, fx, yy in test_loader:
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                loss = 0
                diff = 0
                slice_weights_gt = []
                prev_slice = 0
                for t in range(0, Tout):
                    print(t)
                    y = yy[..., t:t+1]

                    #get the original slice
                    sequen_solver.encoder.encode(pos, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()
                    slice_weights_gt.append(target_slice)

                    #get the code from sequen solver
                    code = sequen_solver.get_code(pos, fx, y)

                    #get the previous slice weight
                    prev_slice_weight = sequen_solver.get_last_slice_weight(pos, fx)

                    #difference slices (loss from the previous method)
                    predicted_slice = model.get_slice_weight(code, x, fx, use_vorticity=use_vorticity)
                    diff += F.mse_loss(predicted_slice, target_slice)

                    #loss from the original
                    new_slice = model.forward_previous_slice(prev_slice_weight, code)
                    loss += F.mse_loss(new_slice, target_slice)
                    #slice_weights_gt.append(new_slice)
                    
                    """if t > 0:
                        diff = F.mse_loss(new_slice, prev_slice)
                        print(f"diffrence prev current {diff}")
                    #print(prev_slice_weight)
                    #print(new_slice)
                    prev_slice = new_slice
                    np_slice = new_slice.cpu().numpy()[0,0]

                    plt.imshow(np_slice, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title("Tensor Visualization")
                    plt.xlabel("Columns (16)")
                    plt.ylabel("Rows (4096)")
                    plt.show()"""

                    #update fx
                    sequen_solver.slice_weights = new_slice
                    decoded = sequen_solver.decode(code)
                    pred = sequen_solver.mlp2(sequen_solver.ln_3(decoded))
                    fx = torch.cat((fx[..., 1:], pred), dim=-1)

            
                print(f"mean diffrence of simulation {diff}")
                print(f"mean loss of simulation {loss}")
                mean_loss += loss
                mean_diff += diff
            print(f"total mean difference {mean_diff/len(test_loader)}")
            print(f"total mean loss {mean_loss/len(test_loader)}")
    else:
        losses = []
        for ep in range(epochs):
            print(f"ep: {ep}")
            model.train()
            loss_epoch = 0
            print(f"train loader size {len(train_loader)}")
            for i, (x, fx, yy) in enumerate(train_loader):
                print(f"i {i}")
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                bsz = x.shape[0]

                #print(f"x {x.shape}, fx {fx.shape}, yy {yy.shape}")
                loss = 0
                for t in range(0, Tout):
                    y = yy[..., t:t+1]

                    #get the original slice
                    sequen_solver.encoder.encode(pos, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()

                    #get the code from sequen solver
                    code = sequen_solver.get_code(pos, fx, y)

                    prev_slice_weight = sequen_solver.get_last_slice_weight(pos, fx)
                    #print(f"prev slice {prev_slice_weight}")

                    #for each position and slice we predict the weight
                    new_slice = model.forward_previous_slice(prev_slice_weight, code)
                    loss += F.mse_loss(new_slice, target_slice)

                    #update fx
                    fx = torch.cat((fx[..., 1:], y), dim=-1)

                loss_epoch += loss
                print(f"train loss {loss}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(f"loss epoch {ep}: {loss_epoch} ")

            if not os.path.exists('./sequential_checkpoints'):
                os.makedirs('./sequential_checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))

            model.eval()
            print(f"TEST")
            with torch.no_grad():
                overall_loss = 0
                print(f"test loader size {len(test_loader)}")
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()

                    for t in range(0, Tout):
                        y = yy[..., t:t+1]

                        #get the original slice
                        sequen_solver.encoder.encode(pos, y)
                        target_slice = sequen_solver.encoder.get_attention_slice()

                        #get the code from sequen solver
                        code = sequen_solver.get_code(pos, fx, y)

                        prev_slice_weight = sequen_solver.get_last_slice_weight(pos, fx)
                        #print(f"prev slice {prev_slice_weight.shape}")

                        #for each position and slice we predict the weight
                        new_slice = model.forward_previous_slice(prev_slice_weight, code)
                        loss += F.mse_loss(new_slice, target_slice)
                    
                    
                    print(f"mean loss of a simulation {loss}")
                    overall_loss += loss
                #overall_loss /= len(test_loader)
                print(f"overall loss {overall_loss}")


def train_from_vorticity(eval=False):
    N = 4096
    M = 16

    batch_size = 1
    epochs = 5
    lr = 0.001
    weight_decay = 1e-5

    #save_name = "buff"
    #save_name = "buff2"
    #save_name = "slice_vort_ep10_sim200"
    #save_name = "slice_vorticity_code_ep10_sim200_b1"

    unified_pos = 1
    use_vorticity = 1
    use_code_for_vorticity = 1
    code_fx = None

    ntrain = 10
    ntest = 10
    Tin = 10 #the size of the input sequence
    Tout = 10 #the number of frames to predict

    train_a, train_u, test_a, test_u = load_data(ntrain, ntest, Tin, Tout)

    #spatial positional encoding
    pos_train, pos_test, pos = encode_spatial_positionning(ntrain, ntest, unified_pos)

    #define loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=batch_size, shuffle=False)
    
    #get model
    transolver_path = "./sequential_checkpoints/encoder_ep20_head_1.pt"
    sequen_solver = SequenSolver(transolver_path, T=10, H=64, W=64, M=16, C=32, B=1, layers=8).cuda()
    
    SequenSolver_path = "./sequential_checkpoints/tokenizer_ep10_sim10_2.pt"
    sequen_solver.load_state_dict(torch.load(SequenSolver_path, weights_only=True), strict=False)
    
    #freeze sequenSolver
    sequen_solver.eval()
    for param in sequen_solver.parameters():
        param.requires_grad = False

    model = LearnSlice(unified_pos=unified_pos, use_vorticity=use_vorticity, use_code_for_vorticity=use_code_for_vorticity)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #for each epoch we have train loader * N * M
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=(len(train_loader)))
    
    mse_loss = nn.MSELoss()

    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./sequential_checkpoints/" + save_name + ".pt", weights_only=True), strict=False)
        model.eval()
        print(f"TEST")
        with torch.no_grad():
            mean_loss = 0
            mean_diff = 0
            for x, fx, yy in test_loader:
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                loss = 0
                diff = 0
                for t in range(0, Tout):
                    print(t)
                    y = yy[..., t:t+1]

                    #get the original slice
                    sequen_solver.encoder.encode(x, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()

                    #get the code from sequen solver
                    code = sequen_solver.get_code(x, fx, y)

                    if use_code_for_vorticity:
                        code_fx = code

                    slice_from_vorticity = model.forward_from_vorticity(x, fx, code=code_fx)
                    loss += mse_loss(slice_from_vorticity, target_slice)
                    #slice_weights_gt.append(new_slice)
                

                    """if t > 0:
                        diff = F.mse_loss(slice_from_vorticity, prev_slice)
                        print(f"diffrence prev current {diff}")
                    #print(prev_slice_weight)
                    #print(new_slice)
                    prev_slice = slice_from_vorticity
                    np_slice = slice_from_vorticity.cpu().numpy()[0,0]

                    plt.imshow(np_slice, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title("predicted")
                    plt.xlabel("Columns (16)")
                    plt.ylabel("Rows (4096)")
                    plt.show()

                    #show gt
                    gt_slice = target_slice.cpu().numpy()[0,0]

                    plt.imshow(gt_slice, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title("gt")
                    plt.xlabel("Columns (16)")
                    plt.ylabel("Rows (4096)")
                    plt.show()"""

                    #reconstruct and update fx
                    sequen_solver.slice_weights = slice_from_vorticity
                    decoded = sequen_solver.decode(code)
                    pred = sequen_solver.mlp2(sequen_solver.ln_3(decoded))
                    fx = torch.cat((fx[..., 1:], pred), dim=-1)

            
                print(f"mean diffrence of simulation {diff}")
                print(f"mean loss of simulation {loss}")
                mean_loss += loss
                mean_diff += diff
            print(f"total mean difference {mean_diff/len(test_loader)}")
            print(f"total mean loss {mean_loss/len(test_loader)}")
    else:
        losses = []
        for ep in range(epochs):
            print(f"ep: {ep}")
            model.train()
            loss_epoch = 0
            print(f"train loader size {len(train_loader)}")
            for i, (x, fx, yy) in enumerate(train_loader):
                print(f"i {i}")
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()

                #print(f"x {x.shape}, fx {fx.shape}, yy {yy.shape}")
                loss = 0
                for t in range(0, Tout):
                    y = yy[..., t:t+1]

                    #get the original slice
                    sequen_solver.encoder.encode(x, y)
                    target_slice = sequen_solver.encoder.get_attention_slice()
                
                    #get the code from sequen solver
                    code = sequen_solver.get_code(x, fx, y)

                    if use_code_for_vorticity:
                        code_fx = code

                    slice_from_vorticity = model.forward_from_vorticity(x, fx, code=code_fx)
                    loss += mse_loss(slice_from_vorticity, target_slice)

                    #update fx
                    fx = torch.cat((fx[..., 1:], y), dim=-1)

                loss_epoch += loss
                print(f"train loss {loss}")
                losses.append(loss_epoch.item())
                print(losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(f"loss epoch {ep}: {loss_epoch} ")

            if not os.path.exists('./sequential_checkpoints'):
                os.makedirs('./sequential_checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))

            model.eval()
            print(f"TEST")
            with torch.no_grad():
                overall_loss = 0
                print(f"test loader size {len(test_loader)}")
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()

                    for t in range(0, Tout):
                        y = yy[..., t:t+1]

                        #get the original slice
                        sequen_solver.encoder.encode(x, y)
                        target_slice = sequen_solver.encoder.get_attention_slice()

                        #get the code from sequen solver
                        code = sequen_solver.get_code(x, fx, y)

                        if use_code_for_vorticity:
                            code_fx = code

                        slice_from_vorticity = model.forward_from_vorticity(x, fx, code=code_fx)
                        loss += mse_loss(slice_from_vorticity, target_slice)


                        #reconstruct and update fx
                        sequen_solver.slice_weights = slice_from_vorticity
                        decoded = sequen_solver.decode(code)
                        pred = sequen_solver.mlp2(sequen_solver.ln_3(decoded))
                        fx = torch.cat((fx[..., 1:], pred), dim=-1)
                    
                    
                    print(f"mean loss of a simulation {loss}")
                    overall_loss += loss
                #overall_loss /= len(test_loader)
                print(f"overall loss {overall_loss/len(test_loader)}")


if __name__ == "__main__":
    train(eval=True)
    #train_from_previous(eval=False)
    #train_from_vorticity(eval=True)
