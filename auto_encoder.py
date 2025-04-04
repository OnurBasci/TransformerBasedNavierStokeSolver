import os
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ns_2d_UniPDE')
parser.add_argument('--data_path', type=str, default='/data/fno')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

#data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat'
data_path = r"C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\data\\fno\\NavierStokes_V1e-5_N1200_T20\\NavierStokes_V1e-5_N1200_T20.mat"
# data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20.mat'
ntrain = 100
ntest = 20
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    print(torch.cuda.is_available())
    r = args.downsample
    h = int(((64 - 1) / r) + 1)

    data = scio.loadmat(data_path)
    print(data['u'].shape)
    #a is the frames until the time T u is the frames after the time T
    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    train = data['u'][:ntrain, :, :, :]
    train = train.reshape(train.shape[0], -1, train.shape[-1])
    train = train.reshape(train.shape[0] * train.shape[-1], train.shape[1], 1)
    train = torch.from_numpy(train)
    test = data['u'][-ntest:, :, :, :]
    test = test.reshape(test.shape[0], -1, test.shape[-1])
    test = test.reshape(test.shape[0] * test.shape[-1], test.shape[1], 1)
    test = torch.from_numpy(test)

    print(f"train data {train.shape}") #simulation * (Tin+tout) , W*H, 1
    print(f"test data {test.shape}")

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain*20, 1, 1)
    pos_test = pos.repeat(ntest*20, 1, 1)
    
    print(pos_train.shape)
    print(pos_test.shape)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test),
                                              batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=h, W=h).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./sequential_checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 0
        id = 0

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        with torch.no_grad():
            test_l2 = 0
            for x, fx in test_loader:
                id += 1
                print(id)
                loss = 0
                x, fx = x.cuda(), fx.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                im = model(x, fx=fx)
                #loss is calculated with the image itself
                loss += myloss(im.reshape(bsz, -1), fx.reshape(bsz, -1))

                test_l2 += loss.item()
            print(test_l2 / ntest)
    else:
        for ep in range(args.epochs):
            model.train()
            train_l2_step = 0
            train_l2_full = 0

            print(len(train_loader))
            for i, (x, fx) in enumerate(train_loader):
                #print(f"training data {i}")
                loss = 0
                x, fx = x.cuda(), fx.cuda()  # x: B,4096,2    fx: B,4096,T   y: B,4096,T
                bsz = x.shape[0]
                im = model(x, fx=fx)
                #loss is calculated with the image itself
                loss += myloss(im.reshape(bsz, -1), fx.reshape(bsz, -1))

                train_l2_step += loss.item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            test_l2_step = 0
            test_l2_full = 0

            model.eval()

            with torch.no_grad():
                for x, fx in test_loader:
                    loss = 0
                    x, fx = x.cuda(), fx.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                    bsz = x.shape[0]
                    im = model(x, fx=fx)
                    #loss is calculated with the image itself
                    loss += myloss(im.reshape(bsz, -1), fx.reshape(bsz, -1))

                    test_l2_step += loss.item()

            print(f"Epoch {ep} , train_step_loss:{train_l2_step / ntrain / (T / step)} , test_step_loss:{test_l2_step / ntest / (T / step)}")

            if ep % 100 == 0:
                if not os.path.exists('./sequential_checkpoints'):
                    os.makedirs('./sequential_checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))

        if not os.path.exists('./sequential_checkpoints'):
            os.makedirs('./sequential_checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./sequential_checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
