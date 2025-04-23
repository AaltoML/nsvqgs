import os
import sys
import pdb
from os.path import join
import datetime
import json
import time
from bitarray import bitarray

import numpy as np
import torch
from random import randint

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.nsvq import QuantizationModule

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# Assistant, Debug.
def write_gpu_stats(writer, step, device):
    reserve = torch.cuda.memory_reserved() / 1024**2
    properties = torch.cuda.get_device_properties(device)
    total = properties.total_memory / 1024**2
    writer.add_scalar(f"GPU/reserve (MB)", reserve, step)
    writer.add_scalar(f"GPU/total (MB)", total, step)
    writer.add_scalar(f"GPU/ratio (%)", reserve / total, step)
    return None

def update_codebooks(iteration, last_cb_update, max_iter):
    update_cb = False
    if iteration > last_cb_update:
        if 20_000 < iteration and iteration <= 22_000:
            update_cb = (iteration % 100 == 0)
        if 22_000 < iteration and iteration <= max_iter -1_000:
            update_cb = (iteration % 500 == 0)
    return update_cb


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    opt.iterations = args.total_iterations
    opt.cb_lr_times= args.cb_lr_times
    if args.fine_tune:
        opt.ft_it=2000
        saving_iterations += [int(opt.iterations - i* opt.ft_it/2) for i in[1, 2]]
        # opt.iterations += opt.ft_it
    
    tb_writer = prepare_output_and_logger(dataset)
    # * 1.1 Initilization: VQ module
    size_cb = {"dc": args.vq_ncls_dc, "sh":  args.vq_ncls_sh, "scale": args.vq_ncls, "rot": args.vq_ncls}
    module_vq = QuantizationModule(args.quant_params, size_cb, device='cuda', last_cb_update = args.vq_start_iter, args = args)
    
    # * 1.1 Initilization: GS model
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, module_vq = module_vq)
    # initialize GS parameters from point clouds
    scene = Scene(dataset, gaussians)
    # setup the gaussians by initilization or loading from checkpoints. info set: gaussian features, create optimizer, load module_vq.parameters into optimizer
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if not args.no_load_cb:
            gaussians.restore(model_params, opt)
        else:
            print('Don\'t load codebook, use the random one')
            gaussians.restore(model_params, opt, False)
    else:
        gaussians.training_setup(opt)
    print(gaussians)
    print(module_vq)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # * 1.2 Initilizationv assistance: tensorboard_logging, progress_bar,
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    num_gaussians_per_iter = []
    
    # * 2. Training loops
    for iteration in range(first_iter, opt.iterations + 1):
        # * 2.1 network GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # * 2.2 updating behavior during training
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() # 2000
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # quantize params
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # * 2.3 Render: traditional or quantized version
        if iteration <= args.vq_start_iter:
            # no quantization, early stage
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        elif args.fine_tune and (iteration > opt.iterations -opt.ft_it):
            # fine tuning
            if iteration == (opt.iterations -opt.ft_it+1):
                print(f'Fine-tuning since iteration {iteration}')
                indices = module_vq.inference(gaussians, save=True)
            module_vq.fine_tune_assign(gaussians, indices)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, vq_mode = 'fine_tune')
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        else:
            # main quantization 
            if iteration == args.vq_start_iter+1:
                print(f'Codebook initialized at iteration {iteration}')
                module_vq.codebooks_initialisation(gaussians)
            update_cb = update_codebooks(iteration, module_vq.last_cb_update, opt.iterations)
            if update_cb:
                unused_count = module_vq.replace_unused_codebooks(iteration - module_vq.last_cb_update)
                module_vq.last_cb_update = iteration
                if tb_writer:
                    for key, value in unused_count.items():
                        # print(key, value)
                        tb_writer.add_scalar(key, value, iteration)
                        
            if args.no_visible_module:
                module_vq.forward(gaussians, radii = None)
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, vq_mode = 'train')
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            else:
                # get visible gaussians
                with torch.no_grad():
                    start_time = time.time()
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                # print(f"rendering time without gradients: {elapsed_time:.6f} seconds")
                # only process visible VQ
                torch.cuda.empty_cache()
                module_vq.forward(gaussians, radii = radii)
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, vq_mode = 'train')
                # codebook updates...
                image = render_pkg["render"]
        # * 2.4 Loss calculation, gradient backpropagation
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # Optionally, use opacity regularization - from iter 15000 to max_prune_iter
        if args.opacity_reg:
            if iteration > args.max_prune_iter or iteration < 15000:
                lambda_reg = 0.
            else:
                lambda_reg = args.lambda_reg
            L_reg_op = gaussians.get_opacity.sum()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (
                    lambda_reg * L_reg_op)
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        # * 2.5 training report, saving results in certain iterations
        with torch.no_grad():
            # * 2.5.1: Log related info: GPU status, loss, rendering visualization on picked images, elapsed time of rendering
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            fq_fresh_progress_bar = 100
            if iteration % fq_fresh_progress_bar == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"}, refresh=False)
                progress_bar.update(fq_fresh_progress_bar)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, radii, module_vq, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, opt)
            psnr_train, psnr_test = psnr['train'], psnr['test']
            # * 2.5.2 (target) save Gaussians and codebook: only pure parameters. For checking the storage cost.. (currently) only save the Gaussians...
            if (iteration in saving_iterations):
                print(args.model_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if iteration >  args.vq_start_iter:
                    if args.fine_tune and (iteration > opt.iterations -opt.ft_it):
                        pass
                    else:
                        module_vq.inference(gaussians, save=True)
                    scene.save(iteration, save_quant = True, quant_params = args.quant_params)
                    # scene.save(iteration, path = os.path.join(scene.model_path, "point_cloud/iteration_{}_continuous".format(iteration)) )
                else:
                    scene.save(iteration) 

            torch.cuda.empty_cache()
            # * 2.6 Densification and pruning
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Prune Gaussians every 1000 iterations from iter 15000 to max_prune_iter if using opacity regularization
            if args.opacity_reg and iteration > opt.densify_until_iter:
                if iteration <= args.max_prune_iter and iteration % 1000 == 0:
                    print('Num Gaussians: ', gaussians._xyz.shape[0])
                    size_threshold = None
                    gaussians.prune(0.005, scene.cameras_extent, size_threshold, radii)
                    print('Num Gaussians after prune: ', gaussians._xyz.shape[0])

            # * 2.7 Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.vq_optimizer.step()
                gaussians.vq_optimizer.zero_grad(set_to_none = True)

            # * 2.8 Save the checkpoint.
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        num_gaussians_per_iter.append(gaussians.get_xyz.shape[0])

    print("Number of Gaussians at the end: ", gaussians._xyz.shape[0])
    np.save(f'{scene.model_path}/num_g_per_iters.npy', np.array(num_gaussians_per_iter))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path, 'tb'))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, radii, module_vq, scene : Scene, renderFunc, renderArgs, train_test_exp, opt):
    '''
    Log information on tensorboard, including training and evaluation related data. (For report training, actually could be canceled if only training needed)
    - training related: GPU status, loss, rendering visualization on picked images, elapsed time of rendering, opacity histogram
    - evaluation related: rendered images on evaluation set, evaluation loss
    '''
    # * 1. log training info
    if tb_writer:
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        num_visible = (radii > 0).sum().item() * 1.0 / num_gaussians
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('nums/total_gaussians', num_gaussians, iteration)
        tb_writer.add_scalar('nums/visible', num_visible, iteration)
        tb_writer.add_scalar('nums/ratio_visible', num_visible/ num_gaussians, iteration)
        if iteration % 100 == 0:
            write_gpu_stats(tb_writer, iteration, device = 'cuda') # costly, better not monitor too frequently
        

    # Report test and samples of training set
    # psnr_test = -1.
    # * 2. log evaluation info
    psnr_out = {'train': -1., 'test': -1}
    if iteration in testing_iterations:        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        if iteration <=  args.vq_start_iter:
            vq_mode = None
        elif args.fine_tune and (iteration > opt.iterations -opt.ft_it):
            vq_mode = 'fine_tune'
        else:
            vq_mode = 'test' 
            module_vq.inference(scene.gaussians, save=True)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, vq_mode=vq_mode)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                        
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                psnr_out[config['name']] = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()
    return psnr_out


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 15_000, 20_000, 22_500, 25_000, 27_500, 30_000])
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[7_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[], help = "iterations during which checkpoints saved")
    parser.add_argument("--start_checkpoint", type=str, default = None, help = "path to the loading checkpoint")
    parser.add_argument('--total_iterations', type=int, default=30000,
                        help='Total iterations of training')

    # Compress3D parameters
    parser.add_argument('--vq_start_iter', type=int, default=30000,
                        help='Start k-Means based vector quantization from this iteration')
    parser.add_argument('--vq_ncls', type=int, default=4096,
                        help='Number of clusters in k-Means quantization')
    parser.add_argument('--vq_ncls_sh', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of spherical harmonics')
    parser.add_argument('--vq_ncls_dc', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of DC component of color')
    parser.add_argument('--grad_thresh', type=float, default=0.0002,
                        help='threshold on xyz gradients for densification')
    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc', 'scale', 'rot'])

    # Opacity regularization parameters
    parser.add_argument('--max_prune_iter', type=int, default=20000,
                        help='Iteration till which pruning is done')
    parser.add_argument('--opacity_reg', action='store_true', default=False,
                        help='use opacity regularization during training')
    parser.add_argument('--lambda_reg', type=float, default=0.,
                        help='Weight for opacity regularization in loss')
    
    parser.add_argument('--no_load_cb', action='store_true', default=False, help='If loading, do not load the codebook parameters.')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='if true, fine tuning the final 3000 iterations.')
    parser.add_argument('--cb_lr_times', type=float, default=1, help='the times of learning rate in codebook')
    parser.add_argument('--no_visible_module', action='store_true', default=False, help='if true, get all visible gaussians for training NSVQ')



    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    args.save_iterations.append(args.total_iterations)
    # args.test_iterations = list(np.arange(0, args.total_iterations, 100))
    # args.quant_params =['dc', 'scale', 'rot']
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if torch.cuda.is_available():
        print("CUDA is available! PyTorch can use the GPU.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. PyTorch will use the CPU.")
    
    
    outfile = join(args.model_path, 'train_args.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as fp:
        json.dump(vars(args), fp, indent=4, default=str)
    print('Quantized Params: ', args.quant_params)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
