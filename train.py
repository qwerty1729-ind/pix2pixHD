import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
from torch.cuda.amp import autocast, GradScaler

# Utility function for least common multiple
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0

# Importing necessary modules
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

# Wrap the training code inside the main block for multiprocessing safety
if __name__ == '__main__':
    # Parse training options
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    # Initialize epoch_iter and start_epoch
    start_epoch, epoch_iter = 1, 0  # Default initialization

    # Resume training if applicable
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            pass  # Continue from the default values if loading fails
        print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')
    else:
        print(f'Starting training from epoch {start_epoch}')

    # Debug mode adjustments
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    # Load dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'#training images = {dataset_size}')

    # Create model and visualizer
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # Wrap model with DataParallel if using multiple GPUs
    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    # Initialize GradScaler for AMP (updated to avoid deprecation warning)
    scaler = torch.amp.GradScaler()  # Updated line

    # Initialize optimizer (fixed to handle both single-GPU and multi-GPU cases)
    if len(opt.gpu_ids) > 1:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    else:
        optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D  # Fixed line

    # Training variables
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    # Training loop
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size  # Update epoch_iter if continuing from an older epoch

        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize  # Increment epoch_iter

            # Save fake images conditionally
            save_fake = total_steps % opt.display_freq == display_delta

            # Forward pass with AMP
            with autocast():
                losses, generated = model(
                    Variable(data['label']),
                    Variable(data['inst']),
                    Variable(data['image']),
                    Variable(data['feat']),
                    infer=save_fake,
                )
                losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
                if len(opt.gpu_ids) > 1:
                    loss_dict = dict(zip(model.module.loss_names, losses))  # Fixed line
                else:
                    loss_dict = dict(zip(model.loss_names, losses))  # Fixed line

                # Calculate losses
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            # Backward pass for generator
            optimizer_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            # Backward pass for discriminator
            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # Update scaler
            scaler.update()

            # Display errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # Display output images
            if save_fake:
                visuals = OrderedDict([
                    ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                    ('synthesized_image', util.tensor2im(generated.data[0])),
                    ('real_image', util.tensor2im(data['image'][0])),
                ])
                visualizer.display_current_results(visuals, epoch, total_steps)

            # Save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print(f'saving the latest model (epoch {epoch}, total_steps {total_steps})')
                if len(opt.gpu_ids) > 1:
                    model.module.save('latest')  # Fixed line
                else:
                    model.save('latest')  # Fixed line
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # End of epoch
        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec')

        # Save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
            if len(opt.gpu_ids) > 1:
                model.module.save('latest')  # Fixed line
                model.module.save(epoch)  # Fixed line
            else:
                model.save('latest')  # Fixed line
                model.save(epoch)  # Fixed line
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # Update fixed parameters after certain iterations
        if opt.niter_fix_global != 0 and epoch == opt.niter_fix_global:
            if len(opt.gpu_ids) > 1:
                model.module.update_fixed_params()  # Fixed line
            else:
                model.update_fixed_params()  # Fixed line

        # Linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            if len(opt.gpu_ids) > 1:
                model.module.update_learning_rate()  # Fixed line
            else:
                model.update_learning_rate()  # Fixed line
