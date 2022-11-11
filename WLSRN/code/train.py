import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
from option import args
from models import *
from loss import *
import utility
import math


torch.manual_seed(100)
torch.cuda.manual_seed(100)
os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu_id)

device = torch.device('cpu' if args.cpu else 'cuda')   
    
    
def load_teacher():
    print("Loading Teacher ====================================>")
    
    if "Teacher" in args.teacher:
        args.n_resblocks = 20
        args.n_resgroups = 10
        net = wlsrn.WLSRN(args).to(device)
        net.load_state_dict_teacher(torch.load('../teacher_checkpoint/Teacher.pt'))
        if args.precision == 'half':
            net.half()

    for p in net.parameters():
        p.requires_grad = False
    
    return net
    
    
def create_student_model():
    print("Preparing Student ===================================>")
    student_checkpoint = utility.checkpoint(args)
    args.n_resblocks = 3
    args.n_resgroups = 3
    student = wlsrn.WLSRN(args).to(device)
    if args.precision == 'half':
        student.half()
    if args.resume:
        load_from = os.path.join(student_checkpoint.dir, 'model', 'model_latest.pt')
        student.load_state_dict_student(torch.load(load_from))
    return student_checkpoint, student
    
def prepare_criterion():
    criterion = Loss(args, student_ckp)
    if args.resume:
        criterion.load(student_ckp.dir)
    return criterion

def prepare_optimizer():
    optimizer = utility.make_optimizer(args, student)
    if args.resume:
        optimizer.load(student_ckp.dir, epoch=len(student_ckp.log))
    return optimizer

def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]


def train(epoch):
    optimizer.schedule()
    student.train()
    criterion.start_log()
    
    lr = optimizer.get_lr()

    student_ckp.write_log(
        '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, lr)
    )
    
    timer_data, timer_model = utility.timer(), utility.timer()
    for batch, (lr, hr, _, idx_scale) in enumerate(train_loader):
        
        lr, hr = prepare(lr, hr)
        timer_data.hold()
        timer_model.tic()
        
        optimizer.zero_grad()
        student_DWT, student_sr = student(lr)
        teacher_DWT, teacher_sr = teacher(lr)

        total_loss = criterion(student_sr, student_DWT, teacher_DWT, hr)
            
        total_loss.backward()
        optimizer.step()


        timer_model.hold()
        
        if (batch) % args.print_every == 0:
            student_ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch) * args.batch_size,
                        len(train_loader.dataset),
                        criterion.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))

        timer_data.tic()    
    
    criterion.end_log(len(train_loader))


def test(epoch):
    student.eval()
    with torch.no_grad():
        if args.save_results: 
            student_ckp.begin_background()
        
        student_ckp.write_log('\nEvaluation:')    
        student_ckp.add_log(torch.zeros(1, len(test_loader), len(args.scale)))
        
        timer_test = utility.timer()
        

        for idx_data, d in enumerate(test_loader):
            for idx_scale, scale in enumerate(args.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = prepare(lr, hr)
                    dwt, sr = student(lr)
                    sr = utility.quantize(sr, args.rgb_range)
                    
                    save_list = [sr]
                    student_ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)
                    if args.save_gt:
                        save_list.extend([lr, hr])

                    if args.save_results:
                        student_ckp.save_results(filename[0], save_list, scale)

                student_ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = student_ckp.log.max(0)
                student_ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        student_ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        student_ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        student_ckp.write_log('Saving...')

        if args.save_results:
            student_ckp.end_background()

        save(is_best=(best[1][0, 0] + 1 == epoch), epoch=epoch)

        student_ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)


def save(is_best, epoch):
    save_root_path = student_ckp.dir
    
    # save model
    save_dirs = [os.path.join(save_root_path, 'model', 'model_latest.pt')]    
    if is_best:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_best.pt'))        
    if args.save_models:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_{}.pt'.format(epoch)))    
    for s in save_dirs:
        torch.save(student.state_dict(), s)
    
    # save loss
    criterion.save(save_root_path)
    criterion.plot_loss(save_root_path, epoch)
    
    # save optimizer
    optimizer.save(save_root_path)
    
    # save psnr
    student_ckp.plot_psnr(epoch)
    torch.save(student_ckp.log, os.path.join(save_root_path, 'psnr_log.pt'))



def print_args():
    msg = ""
    msg += "Model settings\n"
    msg += "Teachers: %s\n" % args.teacher
    msg += "Student: %s\n" % args.model

    msg += "\n"
    
    msg += "Data Settings\n"
    msg += "RGB range: %d\n" % args.rgb_range
    msg += "Scale: %d\n" % args.scale[0]
    size = args.patch_size / args.scale[0]
    msg += "Input Image Size: (%d, %d, 3)\n" % (size, size)
    msg += "Output Image Size: (%d, %d, 3)\n" % (args.patch_size, args.patch_size)
    msg += "\n"
    
    msg += "Training Settings\n"
    msg += "Epochs: %d\n" % args.epochs
    msg += "Learning rate: %f\n" % args.lr
    msg += "Learning rate decay: %s\n" % args.decay
    msg += "\n"
    
    msg += "Distillation Settings\n"
    if args.alpha == 0 and args.feature_loss_used == 0:
        msg += "No distilation\n"
    else:
        msg += "Distillation type: \n"
        
    msg += "\n\n"    
    
    return msg


if __name__ == "__main__":
    msg = print_args()

    print("Preparing Data ====================================>")
    loader = data.Data(args)
    train_loader = loader.loader_train
    test_loader = loader.loader_test

    teacher = load_teacher()
    student_ckp, student = create_student_model()
    criterion = prepare_criterion()
    optimizer = prepare_optimizer()

    student_ckp.write_log(msg)

    
    epoch = 1
    if args.resume == 1:                
        epoch = len(student_ckp.log) + 1
    

    print("Start Training ======================================>") 
    while epoch < args.epochs + 1:
        print("epoch " + str(epoch))
        train(epoch)
        test(epoch)
        epoch += 1
      

