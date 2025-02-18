import os
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model.SOL_Transolver_Structured_Mesh_2D import SOL_Transolver_Structured_Mesh_2D
from phi.torch.flow import *

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=5)
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

data_path = r"C:\\Users\\onurb\\master\\PRJ_4ID22_TP\\Transolver\\PDE-Solving-StandardBenchmark\\data\\fno\\NavierStokes_V1e-5_N1200_T20\\NavierStokes_V1e-5_N1200_T20.mat"

ntrain = 100
ntest = 50
T_in = 10  #the size of training
T = 10  #the size of labels
step = 1 #step is 2 since we have velx, vely
LOOK_AHEAD = 1
max_look_ahead = 10
OFFSET = step * LOOK_AHEAD
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
    look_ahead = LOOK_AHEAD
    offset = OFFSET
    epoch_tresh_look_ahead = args.epochs/2 #the epoch tresh where the lookahead increases

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

    print(train_a.shape)
    print(train_u.shape)

    print(test_a.shape)
    print(test_u.shape)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    print(f"pos train size: {pos_train.shape}")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")

    model = SOL_Transolver_Structured_Mesh_2D(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=T_in,
                                  out_dim=1,                #!!!!Output dimenstion is 2 since we calculate a velocity field
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=h, W=h,
                                  step=step,
                                  look_ahead=look_ahead).cuda()
    
    transolver_model = model.transolver_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        print("evaluation mode")
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 10
        id = 0

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        test_l2_full = 0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                id += 1
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096, 10  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)
                    y = yy[..., t:t + step]

                    #l = torch.sum(torch.pow(im[0].reshape(1, -1) - y[0].reshape(1, -1),2))
                    #print(l)

                    fx = torch.cat((fx[..., step:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                """if id < showcase:
                    print(id)
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(im[0, :, 0].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "case_" + str(id) + "_pred_" + str(20) + ".pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(yy[0, :, t].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt_" + str(20) + ".pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    plt.imshow((im[0, :, 0].reshape(64, 64) - yy[0, :, t].reshape(64, 64)).detach().cpu().numpy(),
                               cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-2, 2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error_" + str(20) + ".pdf"))
                    plt.close()"""
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print(test_l2_full / ntest)
    else:
        #se
        test_losses = []
        look_ahead = LOOK_AHEAD
        offset = step * look_ahead
        model.n = look_ahead
        look_ahead_epoch_thresh = args.epochs/2
        print(f"nb: batches per epoch {len(train_loader)}")
        for ep in range(args.epochs):
            model.train()
            train_l2_step = 0
            train_l2_full = 0

            #for every 2 epoch increase the lookahead
            if ep%look_ahead_epoch_thresh == 0 and ep >= look_ahead_epoch_thresh and look_ahead <= max_look_ahead:
                look_ahead *= 2
                if look_ahead >= max_look_ahead:
                    look_ahead = max_look_ahead
                offset = step * look_ahead
                model.n = look_ahead
                look_ahead_epoch_thresh /= 2
                print(f"look ahead increased {look_ahead}")

            for i, (x, fx, yy) in enumerate(train_loader):
                #print(f"training data {i}")
                #print(f"fx shape {fx.shape}, yy shape {yy.shape}")
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x: B,4096,2    fx: B,4096,T   y: B,4096,T

                bsz = x.shape[0]
                for t in range(0, T-look_ahead+1, look_ahead):
                    y = yy[..., t+offset-step: t+offset]
                    im = model(x, fx)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))

                    y_next = yy[..., t:t + look_ahead]
                    fx = torch.cat((fx[..., look_ahead:], y_next), dim=-1)
                    
                train_l2_step += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                """
                y = yy[..., offset-step: offset]
                im = model(x, fx)
                loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                train_l2_step += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                """

            print("Epoch {} , train_step_loss:{:.5f}".format(ep, train_l2_step)) #loss for all epoch

            test_l2_step = 0
            test_l2_full = 0

            model.eval()

            with torch.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                    bsz = x.shape[0]
                    for t in range(0, T, step): #T
                        y = yy[..., t:t + step]
                        im = transolver_model(x, fx=fx) #here we call transolver model instead of model
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        fx = torch.cat((fx[..., step:], im), dim=-1)
    
                    test_l2_step += loss.item()
                    #test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

                print(
                    "Epoch {} , test_step_loss:{:.5f}".format(
                        ep, test_l2_step / ntest / (T / step)))
                
                test_losses.append(test_l2_step / ntest / (T / step))
            
                """for t in range(0, T-offset+step, step): #ofset = step * lookahead (if look_ahed = 1) upperborn woudld be T
                    #print(f"t is {t}")
                    y = yy[..., t+offset-step:t + offset] 
                    #print(f"x {x.shape}, fx {fx.shape}")
                    im = model(x, fx=fx)  # B , 4096 , 1
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    #print(torch.sum(torch.pow(im.reshape(1, -1) - y.reshape(1, -1),2)))

                    #from frame (t: t+T) to (t+1: t+T+1) 
                    fx = torch.cat((fx[..., step:], y), dim=-1)  # detach() & groundtruth
                    #print(f"fx shape {fx.shape} im shape {im.shape} y shape {y.shape} x shape {x.shape}")

                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            test_l2_step = 0
            test_l2_full = 0"""

            """
            print(
                "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(
                    ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
                        test_l2_full / ntest))
            """
            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(transolver_model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
        
        print(test_losses)

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(transolver_model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
