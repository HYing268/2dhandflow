# from bihand.datasets.handataset import HandDataset
# from bihand.utils.eval.evalutils import AverageMeter
# # from progress.progress.bar import Bar
# from progress.bar import Bar
# from termcolor import colored, cprint

# import bihand.utils.func as func
# import bihand.utils.eval.evalutils as evalutils
# import bihand.utils.misc as misc
# import bihand.utils.imgutils as imutils
# import bihand.losses as losses
# import bihand.models as models
# import os
# import sys
# import argparse
# import time
# import numpy as np
# import torch
# import torch.utils.data
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import matplotlib.pyplot as plt

# # select proper device to run
# # import _init_paths
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #fixme 
# # cudnn.benchmark = True
# # There is BN issue for early version of PyTorch
# # see https://github.com/bearpaw/pytorch-pose/issues/33

# testnum = 36
# seed = 11
# loss_ff_weight = 0.000000000001
# lr = 1e-3
# lrstep = 4000
# device = "cuda:0"
# epoch_num = 5000
# maxoravg = "avg"

# cudnn.benchmark = True

# import random
# # random.seed(seed)
# # np.random.seed(seed)
# # torch.manual_seed(seed)
# # torch.cuda.manual_seed(seed)
# # torch.cuda.manual_seed_all(seed)
# def seed_torch(seed=seed):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True
#     #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
 
# def _init_fn(worker_id):
#     np.random.seed(int(seed)+worker_id)
# seed_torch(seed)

# x_data = []
# y_data = []

# def main(args):
#     best_acc = 0
#     if not os.path.isdir(args.checkpoint):
#         os.makedirs(args.checkpoint)
#     misc.print_args(args)
#     print("\nCREATE NETWORK")
#     model = models.NetBiHand(
#         net_modules=args.net_modules,  # only train hm
#         njoints=21,
#         inp_res=256,
#         out_hm_res=64,
#         out_dep_res=64,
#         upstream_hg_stacks=2,
#         upstream_hg_blocks=1,
#     )
#     model = model.to(device)

#     # define loss function (criterion) and optimizer
#     #fixme
#     # criterion_ups = losses.UpstreamLoss(
#     #     lambda_hm=100.0,
#     #     lambda_mask=1.0
#     # )

#     criterion_ups = losses.FastFlowLoss(
#         device = device,
#         lambda_hm= 100, # 10000.0,
#         lambda_mask= 1, #100.0,
#         lambda_ff= loss_ff_weight #0.01
#     )
#     #fixme end

#     criterion = {
#         'ups': criterion_ups,
#     }
#     #fixme
#     optimizer = torch.optim.Adam(
#         [
#             {
#                 'params': model.upstream.seednet.parameters(),
#                 'initial_lr': args.learning_rate
#             },
#         ],
#         lr=args.learning_rate,
#     )
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, args.lr_decay_step, gamma=args.gamma,
#         last_epoch=args.start_epoch
#     )  #fixme 

#     # optimizer = torch.optim.Adam(
#     #     model.upstream.seednet.parameters(), lr=args.learning_rate, weight_decay = 1e-5
#     # )
#     #fixme end

#     print("\nCREATE DATASET")
#     train_dataset = HandDataset(
#         data_split='train',
#         train=True,
#         scale_jittering=0.2,
#         center_jettering=0.2,
#         max_rot=0.5 * np.pi,
#         subset_name=args.datasets,
#         data_root=args.data_root,
#     )
#     val_dataset = HandDataset(
#         data_split='test',
#         train=False,
#         subset_name=args.datasets,
#         data_root=args.data_root,
#     )
#     print("Total train dataset size: {}".format(len(train_dataset)))
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.train_batch,
#         shuffle=True,
#         num_workers=args.workers,
#         pin_memory=True, worker_init_fn=_init_fn
#         )
    
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=args.test_batch,
#         shuffle=True,
#         num_workers=args.workers,
#         pin_memory=True, worker_init_fn=_init_fn
#     )
#     print("Total test dataset size: {}".format(len(val_dataset)))
#     print("\nLOAD CHECKPOINT")
#     if args.resume_seednet_pth:
#         model.load_checkpoints(ckp_seednet=args.resume_seednet_pth)
#     else:
#         for m in model.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)

#     if args.evaluate_seednet_pth:
#         model.load_checkpoints(ckp_seednet=args.evaluate_seednet_pth)
#         validate(val_loader, model, criterion, args=args)
#         return 0
#     #fixme
#     # model = torch.nn.DataParallel(model) 
#     print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    

#     for epoch in range(args.start_epoch, args.epochs + 1):
#         print('\nEpoch: %d' % (epoch + 1))
#         for i in range(len(optimizer.param_groups)):
#             print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
#         #############  trian for on epoch  ###############
#         train(
#             train_loader,
#             model,
#             criterion,
#             optimizer,
#             args=args,
#         )
#         ##################################################
#         acc_hm = best_acc
#         # if epoch >= 0 and epoch % 1 == 0:
#         if epoch >= 30 and epoch % 5 == 0:
#             x_data.append(epoch)
#         # if epoch == 0:
#             acc_hm = validate(val_loader, model, criterion, args=args)
#             y_data.append(acc_hm)
#         misc.save_checkpoint(
#             {
#                 'epoch': epoch + 1,
#                 'state_dict': model.upstream.seednet.state_dict(),
#                 # 'state_dict': model.module.upstream.seednet.state_dict(),
#             },
#             checkpoint=args.checkpoint,
#             filename='{}.pth.tar'.format(args.saved_prefix),
#             snapshot=args.snapshot,
#             is_best=acc_hm > best_acc
#         )
#         if acc_hm > best_acc:
#             best_acc = acc_hm
#         #fixme 
#         scheduler.step()
#         # optimizer.step()
#         #fixme end
#     cprint('All Done', 'yellow', attrs=['bold'])
#     #fixme
#     data_name = "/data/haoying/bihand_data/data" + str(testnum) + ".txt"
#     with open(data_name, 'w') as file:
#         for i in range(len(x_data)):
#             file.write(f'{x_data[i]},{y_data[i]}\n')
#     #fixme end
#     return 0  # end of main


# def one_forward_pass(metas, model, criterion, args, train=True): #计算模型的前向传播结果，并根据训练模式计算损失
#     ''' prepare infos 准备输入数据：从metas字典中获取关节根部位置、关节骨长、内参矩阵、深度遮罩、热图遮罩等数据 '''
#     joint_root = metas['joint_root'].to(device, non_blocking=True)  # (B, 3)
#     joint_bone = metas['joint_bone'].to(device, non_blocking=True)  # (B, 1)
#     intr = metas['intr'].to(device, non_blocking=True)
#     hm_veil = metas['hm_veil'].to(device, non_blocking=True)
#     dep_veil = metas['dep_veil'].to(device, non_blocking=True)  # (B, 1)
#     ndep_valid = torch.sum(dep_veil).item()
#     infos = {
#         'joint_root': joint_root,
#         'intr': intr,
#         'joint_bone': joint_bone,
#         'hm_veil': hm_veil,
#         'dep_veil': dep_veil,
#         'batch_size': joint_root.shape[0],
#         'ndep_valid': ndep_valid,
#     }
#     ''' prepare targets 准备目标数据：从metas字典中获取颜色图像、热图、深度图、二维关键点、三维关节点等数据'''
#     clr = metas['clr'].to(device, non_blocking=True)
#     hm = metas['hm'].to(device, non_blocking=True)
#     dep = metas['dep'].to(device, non_blocking=True)  # (B, 64, 64)
#     kp2d = metas['kp2d'].to(device, non_blocking=True)
#     joint = metas['joint'].to(device, non_blocking=True)  # (B, 21, 3)
#     jointRS = metas['jointRS'].to(device, non_blocking=True)
#     mask = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)
#     targets = {
#         'clr': clr,
#         'hm': hm,
#         'joint': joint,
#         'kp2d': kp2d,
#         'jointRS': jointRS,
#         'dep': dep,
#         'mask': mask,
#     }
#     ''' ----------------  Forward Pass  执行前向传播：将颜色图像数据和一些额外信息传入模型进行前向传播，得到预测结果---------------- '''
#     results = model(clr, infos) 
#     ''' ----------------  Forward End   ---------------- '''

#     total_loss = torch.Tensor([0]).to(device)
#     # total_loss = torch.Tensor([0]).cuda()   #fixme
#     losses = {}
#     if not train:
#         return results, {**targets, **infos}, total_loss, losses

#     ''' conpute losses '''
#     if args.ups_loss:
#         ups_total_loss, ups_losses = criterion['ups'].compute_loss(
#             results, targets, infos
#         )
#         total_loss += ups_total_loss
#         losses.update(ups_losses)

#     return results, {**targets, **infos}, total_loss, losses


# def train(train_loader, model, criterion, optimizer, args):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     am_loss_hm = AverageMeter()
#     am_loss_mask = AverageMeter()
#     am_loss_all = AverageMeter()

#     last = time.time()
#     # switch to trian
#     model.train()
#     bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
#     for i, metas in enumerate(train_loader):
#         data_time.update(time.time() - last)
#         results, targets, total_loss, losses = one_forward_pass(
#             metas, model, criterion, args, train=True
#         )
#         am_loss_hm.update(
#             losses['ups_hm'].item(), targets['batch_size']
#         )
#         am_loss_mask.update(
#             losses['ups_mask'].item(), targets['batch_size']
#         )
#         am_loss_all.update(
#             total_loss.item(), targets['batch_size']
#         )

#         ''' backward and step '''
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         ''' progress '''
#         batch_time.update(time.time() - last)
#         last = time.time()
#         bar.suffix = (
#             '({batch}/{size}) '
#             'd: {data:.2f}s | '
#             'b: {bt:.2f}s | '
#             't: {total:}s | '
#             'eta:{eta:}s | '
#             'lH: {lossH:.5f} | '
#             'lM: {lossM:.5f} | '
#             'lA: {lossA:.3f} |'
#         ).format(
#             batch=i + 1,
#             size=len(train_loader),
#             data=data_time.avg,
#             bt=batch_time.avg,
#             total=bar.elapsed_td,
#             eta=bar.eta_td,
#             lossH=am_loss_hm.avg,
#             lossM=am_loss_mask.avg,
#             lossA=am_loss_all.avg,
#         )
#         bar.next()
#     bar.finish()


# def validate(val_loader, model, criterion, args, stop=-1):
#     if maxoravg == "max":
#         max_acc_hm = 0
#         model.eval()
#         bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
#         with torch.no_grad():
#             for i, metas in enumerate(val_loader):
#                 results, targets, _1, _2 = one_forward_pass(
#                     metas, model, criterion, args=None, train=False
#                 )
#                 avg_acc_hm, _ = evalutils.accuracy_heatmap(
#                     results['l_hm'][-1],
#                     targets['hm'],
#                     targets['hm_veil']
#                 )
#                 # print("111111",avg_acc_hm)
#                 if avg_acc_hm> max_acc_hm:
#                     max_acc_hm = avg_acc_hm # 获取每个样本的最大值
#                 bar.suffix = (
#                     '({batch}/{size}) '
#                     'accH: {accH:.4f} | '
#                 ).format(
#                     batch=i + 1,
#                     size=len(val_loader),
#                     accH=max_acc_hm,  # 取最大值的平均值
#                 )
#                 # am_accH.update(max_acc_hm, targets['batch_size'])  # 累积最大值
#                 bar.next()
#                 if stop != -1 and i >= stop:
#                     break
#             bar.finish()
#             print("max_accH: {}".format(max_acc_hm))  # 输出累积的最大值
#         return max_acc_hm
#     elif maxoravg == "avg":
#         am_accH = AverageMeter()
#         model.eval()
#         bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
#         with torch.no_grad():
#             for i, metas in enumerate(val_loader):
#                 results, targets, _1, _2 = one_forward_pass(
#                     metas, model, criterion, args=None, train=False
#                 )
#                 avg_acc_hm, _ = evalutils.accuracy_heatmap(
#                     results['l_hm'][-1],
#                     targets['hm'],
#                     targets['hm_veil']
#                 )
#                 bar.suffix = (
#                     '({batch}/{size}) '
#                     'accH: {accH:.4f} | '
#                 ).format(
#                     batch=i + 1,
#                     size=len(val_loader),
#                     accH=avg_acc_hm,
#                 )
#                 am_accH.update(avg_acc_hm, targets['batch_size'])
#                 bar.next()
#                 if stop != -1 and i >= stop:
#                     break
#             bar.finish()
#             print("accH: {}".format(am_accH.avg))
            
#         return am_accH.avg


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='PyTorch Train BiHand Stage 1: SeedNet')
#     # Dataset setting
#     parser.add_argument(
#         '-dr',
#         '--data_root',
#         type=str,
#         default='data',
#         help='dataset root directory'
#     )
#     parser.add_argument(
#         "--datasets",
#         nargs="+",
#         default=['stb', 'rhd'],
#         type=str,
#         help="sub datasets, should be listed in: [rhd|stb]"
#     )

#     # Model Structure
#     ## hourglass:
#     parser.add_argument(
#         '-hgs',
#         '--hg-stacks',
#         default=2,
#         type=int,
#         metavar='N',
#         help='Number of hourglasses to stack'
#     )
#     parser.add_argument(
#         '-hgb',
#         '--hg-blocks',
#         default=1,
#         type=int,
#         metavar='N',
#         help='Number of residual modules at each location in the hourglass'
#     )
#     parser.add_argument(
#         '-nj',
#         '--njoints',
#         default=21,
#         type=int,
#         metavar='N',
#         help='Number of heatmaps calsses (hand joints) to predict in the hourglass'
#     )

#     ####################################
#     # Miscs
#     parser.add_argument(
#         '-ckp',
#         '--checkpoint',
#         default='checkpoints'+str(testnum),
#         type=str,
#         metavar='PATH',
#         help='path to save checkpoint (default: checkpoint)'
#     )
#     parser.add_argument(
#         '--resume_seednet_pth',
#         default='',
#         type=str,
#         metavar='PATH',
#         help='whether to load resume checkpoints pth (default: none)'
#     )
#     parser.add_argument(
#         '--evaluate_seednet_pth',
#         default='',
#         type=str,
#         metavar='PATH',
#         help='whether to load checkpoints pth for evaluation ONLY (default: none)'
#     )
#     parser.add_argument(
#         '-sp',
#         '--saved_prefix',
#         default='ckp_seednet_all',
#         type=str,
#         metavar='PATH',
#         help='path to save checkpoint (default: checkpoint)'
#     )
#     parser.add_argument(
#         '--snapshot',
#         default=5, type=int,
#         help='save models for every #snapshot epochs (default: 0)'
#     )
#     parser.add_argument(
#         '-d', '--debug',
#         dest='debug',
#         action='store_true',
#         default=False,
#         help='show intermediate results'
#     )

#     # Training Parameters
#     parser.add_argument(
#         '-j', '--workers',
#         default=8,
#         type=int,
#         metavar='N',
#         help='number of data loading workers (default: 8)'
#     )
#     parser.add_argument(
#         '--epochs',
#         default=epoch_num,
#         # default=100,
#         type=int,
#         metavar='N',
#         help='number of total epochs to run'
#     )
#     parser.add_argument(
#         '-se', '--start_epoch',
#         default=0,
#         type=int,
#         metavar='N',
#         help='manual epoch number (useful on restarts)'
#     )
#     parser.add_argument(
#         '-b', '--train_batch',
#         default=64,
#         type=int,
#         metavar='N',
#         help='train batchsize'
#     )
#     parser.add_argument(
#         '-tb', '--test_batch',
#         default=32,
#         type=int,
#         metavar='N',
#         help='test batchsize'
#     )
# ##################################
#     parser.add_argument(
#         '-lr', '--learning-rate',
#         # default=1.0e-4,
#         default=lr,    
#         type=float,
#         metavar='LR',
#         help='initial learning rate'
#     )
#     parser.add_argument(
#         "--lr_decay_step",
#         default=lrstep,
#         # default=50,
#         type=int,
#         help="Epochs after which to decay learning rate",
#     )
#     parser.add_argument(
#         '--gamma',
#         type=float,
#         default=0.1,
#         help='LR is multiplied by gamma on schedule.'
#     )
#     parser.add_argument(
#         "--net_modules",
#         nargs="+",
#         default=['seed'],
#         type=str,
#         help="sub modules contained in model"
#     )
#     parser.add_argument(
#         '--ups_loss',
#         dest='ups_loss',
#         action='store_true',
#         help='Calculate upstream loss',
#         default=True
#     )

#     main(parser.parse_args())


from bihand.datasets.handataset import HandDataset
from bihand.utils.eval.evalutils import AverageMeter
# from progress.progress.bar import Bar
from progress.bar import Bar
from termcolor import colored, cprint

import bihand.utils.func as func
import bihand.utils.eval.evalutils as evalutils
import bihand.utils.misc as misc
import bihand.utils.imgutils as imutils
import bihand.losses as losses
import bihand.models as models
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import matplotlib.pyplot as plt

# select proper device to run
# import _init_paths
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #fixme 
# cudnn.benchmark = True
# There is BN issue for early version of PyTorch
# see https://github.com/bearpaw/pytorch-pose/issues/33

testnum =200
seed = 25
loss_ff_weight = 0#0.0000000000001
device = "cuda:2"
maxoravg = "avg"
datasetsname = ['stb']
backbone_name="resnet18"   #resnet18 (256)   deit_base_distilled_patch16_384    wide_resnet50_2     cait_m48_448 (448)

if backbone_name=="resnet18":
    input_size = 256
    flow_steps = 8 
    hidden_ratio = 1.0 #1.0
    resorcait = True
    conv3x3_only = True
    checkpoint_path = './checkpoints33/ckp_seednet_all_20.pth.tar' #res
elif backbone_name=="cait_m48_448":
    input_size = 448
    flow_steps = 20 #8
    hidden_ratio = 0.16 #1.0
    resorcait = False
    conv3x3_only = False
else:
    input_size = 384
    flow_steps = 8 #8
    hidden_ratio = 1 #1.0
    resorcait = False
    conv3x3_only = False
    checkpoint_path = './checkpoints31/ckp_seednet_all_100.pth.tar' #deit

epoch_num = 99    #80
train_batch = 32
lr = 1e-3      #1e-4
lrstep = 100     # 50
cudnn.benchmark = True

import random
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
def seed_torch(seed=seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)
seed_torch(seed)

x_data = 0
y_data = 0

data_name = "/data/sa22010013/bihand_data/data" + str(testnum) + ".txt"

def main(args):
    best_acc = 0
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    misc.print_args(args)
    print("\nCREATE NETWORK")

    model = models.NetBiHand(
        net_modules=args.net_modules,  # only train hm
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=2,
        upstream_hg_blocks=1,
    ) #fixme1

    # model = models.FastFlow(
    #     backbone_name=backbone_name,  #resnet18 (256)   deit_base_distilled_patch16_384    wide_resnet50_2     cait_m48_448 (448)
    #     flow_steps=flow_steps,
    #     input_size=input_size,
    #     conv3x3_only=conv3x3_only,
    #     hidden_ratio=hidden_ratio,
    #     resorcait=resorcait,
    # )

    model = model.to(device)

    # define loss function (criterion) and optimizer
    #fixme
    # criterion_ups = losses.UpstreamLoss(
    #     lambda_hm=100.0,
    #     lambda_mask=1.0
    # )

    criterion_ups = losses.FastFlowLoss(
        device = device,
        lambda_hm= 100, # 10000.0,
        lambda_mask= 1, #100.0,
        lambda_ff= loss_ff_weight #0.01
    )

    #fixme end

    criterion = {
        'ups': criterion_ups,
    }
    #fixme
    optimizer = torch.optim.Adam(
        [
            {
                # 'params': model.parameters(),
                'params': model.upstream.seednet.parameters(),  #fixme4
                'initial_lr': args.learning_rate
            },
        ],
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )  #fixme 

    # optimizer = torch.optim.Adam(
    #     model.upstream.seednet.parameters(), lr=args.learning_rate, weight_decay = 1e-5
    # )
    #fixme end

    print("\nCREATE DATASET")
    train_dataset = HandDataset(
        data_split='train',
        train=True,
        scale_jittering=0.2,
        center_jettering=0.2,
        max_rot=0.5 * np.pi,
        subset_name=args.datasets,
        data_root=args.data_root,
        input_size = input_size,
    )
    val_dataset = HandDataset(
        data_split='test',
        train=False,
        subset_name=args.datasets,
        data_root=args.data_root,
        input_size = input_size,
    )
    print("Total train dataset size: {}".format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, worker_init_fn=_init_fn
        )   
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, worker_init_fn=_init_fn
    )
    print("Total test dataset size: {}".format(len(val_dataset)))
    if args.resume_seednet_pth:
        print("\nLOAD CHECKPOINT")
        # model.load_checkpoints(ckp_seednet=args.resume_seednet_pth)
        # misc.load_checkpoint(model.feature_extractor,args.resume_seednet_pth) #fixmefixme1
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight) #卷积层进行权重初始化

    # if args.evaluate_seednet_pth:
    #     model.load_checkpoints(ckp_seednet=args.evaluate_seednet_pth)
    #     validate(val_loader, model, criterion, args=args)
    #     return 0
    
    #fixme
    # model = torch.nn.DataParallel(model) 
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
 
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for on epoch  ###############
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
        )
        ##################################################
        acc_hm = best_acc
        if epoch >= 0 and epoch % 1 == 0:
            x_data=epoch
        # if epoch == 0:
            acc_hm = validate(val_loader, model, criterion, args=args)
            y_data=acc_hm
        # misc.save_checkpoint(
        #     {
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         # 'state_dict': model.module.upstream.seednet.state_dict(),
        #     },
        #     checkpoint=args.checkpoint,
        #     filename='{}.pth.tar'.format(args.saved_prefix),
        #     snapshot=args.snapshot,
        #     is_best=acc_hm > best_acc
        # )
        if acc_hm > best_acc:
            best_acc = acc_hm
            misc.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    # 'state_dict': model.feature_extractor.state_dict(),
                    'state_dict': model.upstream.seednet.state_dict(),
                },
                checkpoint=args.checkpoint,
                filename='{}.pth.tar'.format(args.saved_prefix),
                snapshot=args.snapshot,
                is_best=acc_hm > best_acc
            )
        scheduler.step()
        with open(data_name, 'a') as file:
            file.write(f'{x_data},{y_data}\n')
    cprint('All Done', 'yellow', attrs=['bold'])
    return 0  # end of main


def one_forward_pass(metas, model, criterion, args, train=True): #计算模型的前向传播结果，并根据训练模式计算损失
    ''' prepare infos 准备输入数据：从metas字典中获取关节根部位置、关节骨长、内参矩阵、深度遮罩、热图遮罩等数据 '''
    joint_root = metas['joint_root'].to(device, non_blocking=True)  # (B, 3)
    joint_bone = metas['joint_bone'].to(device, non_blocking=True)  # (B, 1)
    intr = metas['intr'].to(device, non_blocking=True)
    hm_veil = metas['hm_veil'].to(device, non_blocking=True)
    dep_veil = metas['dep_veil'].to(device, non_blocking=True)  # (B, 1)
    ndep_valid = torch.sum(dep_veil).item()
    infos = {
        'joint_root': joint_root,
        'intr': intr,
        'joint_bone': joint_bone,
        'hm_veil': hm_veil,
        'dep_veil': dep_veil,
        'batch_size': joint_root.shape[0],
        'ndep_valid': ndep_valid,
    }
    ''' prepare targets 准备目标数据：从metas字典中获取颜色图像、热图、深度图、二维关键点、三维关节点等数据'''
    clr = metas['clr'].to(device, non_blocking=True)
    hm = metas['hm'].to(device, non_blocking=True)
    dep = metas['dep'].to(device, non_blocking=True)  # (B, 64, 64)
    kp2d = metas['kp2d'].to(device, non_blocking=True)
    joint = metas['joint'].to(device, non_blocking=True)  # (B, 21, 3)
    jointRS = metas['jointRS'].to(device, non_blocking=True)
    mask = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)
    targets = {
        'clr': clr,
        'hm': hm,
        'joint': joint,
        'kp2d': kp2d,
        'jointRS': jointRS,
        'dep': dep,
        'mask': mask,
    }
    ''' ----------------  Forward Pass  执行前向传播：将颜色图像数据和一些额外信息传入模型进行前向传播，得到预测结果---------------- '''
    results = model(clr, infos)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).to(device)
    # total_loss = torch.Tensor([0]).cuda()   #fixme
    losses = {}
    if not train:
        return results, {**targets, **infos}, total_loss, losses

    ''' conpute losses '''
    if args.ups_loss:
        ups_total_loss, ups_losses = criterion['ups'].compute_loss(
            results, targets, infos
        )
        total_loss += ups_total_loss
        losses.update(ups_losses)

    return results, {**targets, **infos}, total_loss, losses


def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss_hm = AverageMeter()
    am_loss_mask = AverageMeter()
    am_loss_all = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )
        am_loss_hm.update(
            losses['ups_hm'].item(), targets['batch_size']
        )
        am_loss_mask.update(
            losses['ups_mask'].item(), targets['batch_size']
        )
        am_loss_all.update(
            total_loss.item(), targets['batch_size']
        )

        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lH: {lossH:.5f} | '
            'lM: {lossM:.5f} | '
            'lA: {lossA:.3f} |'
        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossH=am_loss_hm.avg,
            lossM=am_loss_mask.avg,
            lossA=am_loss_all.avg,
        )
        bar.next()
    bar.finish()


def validate(val_loader, model, criterion, args, stop=-1):
    if maxoravg == "max":
        max_acc_hm = 0
        model.eval()
        bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
        with torch.no_grad():
            for i, metas in enumerate(val_loader):
                results, targets, _1, _2 = one_forward_pass(
                    metas, model, criterion, args=None, train=False
                )
                avg_acc_hm, _ = evalutils.accuracy_heatmap(
                    results['l_hm'][-1],
                    targets['hm'],
                    targets['hm_veil']
                )
                # print("111111",avg_acc_hm)
                if avg_acc_hm> max_acc_hm:
                    max_acc_hm = avg_acc_hm # 获取每个样本的最大值
                bar.suffix = (
                    '({batch}/{size}) '
                    'accH: {accH:.4f} | '
                ).format(
                    batch=i + 1,
                    size=len(val_loader),
                    accH=max_acc_hm,  # 取最大值的平均值
                )
                # am_accH.update(max_acc_hm, targets['batch_size'])  # 累积最大值
                bar.next()
                if stop != -1 and i >= stop:
                    break
            bar.finish()
            print("max_accH: {}".format(max_acc_hm))  # 输出累积的最大值
        return max_acc_hm
    elif maxoravg == "avg":
        am_accH = AverageMeter()
        model.eval()
        bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
        with torch.no_grad():
            for i, metas in enumerate(val_loader):
                results, targets, _1, _2 = one_forward_pass(
                    metas, model, criterion, args=None, train=False
                )
                avg_acc_hm, _ = evalutils.accuracy_heatmap(
                    results['l_hm'][-1],
                    targets['hm'],
                    targets['hm_veil']
                )
                bar.suffix = (
                    '({batch}/{size}) '
                    'accH: {accH:.4f} | '
                ).format(
                    batch=i + 1,
                    size=len(val_loader),
                    accH=avg_acc_hm,
                )
                am_accH.update(avg_acc_hm, targets['batch_size'])
                bar.next()
                if stop != -1 and i >= stop:
                    break
            bar.finish()
            print("accH: {}".format(am_accH.avg))
            
        return am_accH.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='PyTorch Train BiHand Stage 1: SeedNet')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='data',
        help='dataset root directory'
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=datasetsname,
        type=str,
        help="sub datasets, should be listed in: [rhd|stb]"
    )

    # Model Structure
    ## hourglass:
    parser.add_argument(
        '-hgs',
        '--hg-stacks',
        default=2,
        type=int,
        metavar='N',
        help='Number of hourglasses to stack'
    )
    parser.add_argument(
        '-hgb',
        '--hg-blocks',
        default=1,
        type=int,
        metavar='N',
        help='Number of residual modules at each location in the hourglass'
    )
    parser.add_argument(
        '-nj',
        '--njoints',
        default=21,
        type=int,
        metavar='N',
        help='Number of heatmaps calsses (hand joints) to predict in the hourglass'
    )

    ####################################
    # Miscs
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='checkpoints'+str(testnum),
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--resume_seednet_pth',
        default=checkpoint_path,
        type=str,
        metavar='PATH',
        help='whether to load resume checkpoints pth (default: none)'
    )
    parser.add_argument(
        '--evaluate_seednet_pth',
        default='',
        type=str,
        metavar='PATH',
        help='whether to load checkpoints pth for evaluation ONLY (default: none)'
    )
    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_seednet_all',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--snapshot',
        default=5, type=int,
        help='save models for every #snapshot epochs (default: 0)'
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='show intermediate results'
    )

    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=epoch_num,
        # default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=train_batch,
        type=int,
        metavar='N',
        help='train batchsize'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=16,
        type=int,
        metavar='N',
        help='test batchsize'
    )
##################################
    parser.add_argument(
        '-lr', '--learning-rate',
        # default=1.0e-4,
        default=lr,    
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=lrstep,
        # default=50,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed'],
        type=str,
        help="sub modules contained in model"
    )
    parser.add_argument(
        '--ups_loss',
        dest='ups_loss',
        action='store_true',
        help='Calculate upstream loss',
        default=True
    )

    main(parser.parse_args())
