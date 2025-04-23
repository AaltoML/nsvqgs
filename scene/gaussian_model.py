# Code modified based on Inria-3DGS repo
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from os.path import join
import pdb
from tqdm import tqdm
from bitarray import bitarray
from collections import OrderedDict

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass
class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        
        # self.scaling_activation = self._act_scale
        # self.scaling_inverse_activation = self._inverse_act_scale
    
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    def _act_scale(self, scaling):
        # out = torch.sigmoid(scaling)*5
        out = torch.log(1+ torch.exp(scaling))
        # out = torch.exp(scaling)
        return out
    def _inverse_act_scale(self, out):
        # scaling = torch.logit(out/5)
        scaling = torch.log( torch.clamp(torch.exp(out) - 1, min=1e-20 ) )
        # out = torch.log(out)
        return scaling

    def __init__(self, sh_degree, optimizer_type="default", module_vq: nn.Module = None):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._xyz_q = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_dc_q = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_rest_q = torch.empty(0)
        self._scaling = torch.empty(0)
        self._scaling_q = torch.empty(0)
        self._rotation = torch.empty(0)
        self._rotation_q = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.vq_mode = None 
        self.module_vq = module_vq
        self.vq_optimizer = None
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation =  lambda x: torch.log(torch.clamp(x, min = 1e-20))
        self.setup_functions()

    def __str__(self):
        l = ['GaussianModel, with features']
        key = 'xyz'
        for key in [ 'xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']:
            shape = getattr(self,f'_{key}').shape
            device = self._xyz.device
            l.append(f'[{key}] {shape}, device: {device}')
        return '\n'.join(l)
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.module_vq.state_dict(),
            self.vq_optimizer.state_dict()
        )
    def restore(self, model_args, training_args, load_codebook = True):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        vq_dict,
        vq_opt_dict) = model_args
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # ! issue
        # todo: recover it after debug
        if load_codebook:
            self.module_vq.load_state_dict(vq_dict)
            self.vq_optimizer.load_state_dict(vq_opt_dict)

    @property
    def get_scaling(self):
        if self.vq_mode is None:
            scaling = self._scaling
        elif 'scale' not in self.module_vq.quant_params:
            scaling = self._scaling
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                scaling = scaling[visible]
        else:
            scaling = self._scaling_q
        return self.scaling_activation(scaling)

    @property
    def get_scaling_nq(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # vq_mode: None means origin, train or test requires the quantization part, fine-tuning requires quantization part
        if self.vq_mode is None:
            rotation = self._rotation
        elif 'rot' not in self.module_vq.quant_params:
            rotation = self._rotation
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                rotation = rotation[visible]
        else:
            rotation = self._rotation_q
        return self.rotation_activation(rotation)

    @property
    def get_features(self):
        if self.vq_mode is None:
            features_dc = self._features_dc
        elif 'dc' not in self.module_vq.quant_params:
            features_dc = self._features_dc
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                features_dc = features_dc[visible]
        else:
            features_dc = self._features_dc_q
            
        if self.vq_mode is None:
            features_rest = self._features_rest
        elif 'sh' not in self.module_vq.quant_params:
            features_rest = self._features_rest
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                features_rest = features_rest[visible]
        else:
            features_rest = self._features_rest_q
        # if self.vq_mode is not None:
        #     breakpoint()
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_xyz(self):
        xyz = self._xyz
        if self.vq_mode == "train" and self.module_vq:
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                xyz = xyz[visible]
        return xyz
    
    @property
    def get_opacity(self):
        opacity = self._opacity
        if self.vq_mode == "train" and self.module_vq:
            visible = self.module_vq._temp_visiblity
            if visible is not None:
                opacity = opacity[visible]
        return self.opacity_activation(opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # additional clamp
        dist2 = torch.clamp_max(dist2, 2**2)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if hasattr(self, 'module_vq') and self.module_vq is not None:
            vq_l = []
            lrs = {'dc': training_args.feature_lr, 'sh':training_args.feature_lr / 20.0, 'scale':training_args.scaling_lr, 'rot': training_args.rotation_lr, 'opacity':training_args.opacity_lr}
            for key in self.module_vq.quant_params:
                vq = self.module_vq.nsvqs[key]
                lr = lrs[key]
                vq_l.append({"params": vq.parameters(), "lr": lr * training_args.cb_lr_times, "name": f"vq_{key}"})
            # vq_l = [{"params": self.module_vq.parameters(), "lr": 0.001, "name": "module_vq"}]
            self.vq_optimizer = torch.optim.Adam(vq_l, lr=0.0, eps=1e-15)
        
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    def construct_list_of_attributes(self, save_att=None):
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = []
        if 'xyz' in save_att:
            l += ['x', 'y', 'z']
        if 'normals' in save_att:
            l += ['nx', 'ny', 'nz']
        # All channels except the 3 DC
        if 'dc' in save_att:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
        if 'sh' in save_att:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if 'scale' in save_att:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
        if 'rot' in save_att:
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        return l
    
    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l

    def save_ply(self, path, save_quant= False, quant_params = None):
        mkdir_p(path)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        all_attributes = {'xyz': xyz, 'dc': f_dc, 'sh': f_rest, 'opacities': opacities,
                          'scale': scale, 'rot': rotation}
        # if save_attributes is None:
            # save_attributes = list(all_attributes.keys()
        # - save non-quantized part if quantization
        if save_quant and quant_params is not None:
            save_attributes = [k for k in all_attributes.keys() if k not in quant_params]
        else:
            save_attributes = list(all_attributes.keys())
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_attributes)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(tuple([val for (key, val) in all_attributes.items() if key in save_attributes]), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(path, "point_cloud.ply"))
        print('non-quantized attributes: ', list(save_attributes))
        if save_quant:
            print('quantized attributes: ', quant_params)
            self.module_vq.save_binary(quant_params, path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _bin2dec(self, b, bits):
        """Convert binary b to decimal integer.

        Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        """
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, torch.int64)
        return torch.sum(mask * b, -1)
    
    def load_ply(self, path, load_quant = False):
        plydata = PlyData.read(os.path.join(path, 'point_cloud.ply'))
        quant_params = []
        if load_quant:
            # info loaded
            codebook = torch.load(os.path.join(path, 'codebook.pth'))
            vq_args = np.load(os.path.join(path, 'vq_args.npy'), allow_pickle=True).item()
            quant_params = vq_args['quant_params']
            loaded_bitarray = bitarray()
            with open(join(path, 'vq_inds.bin'), 'rb') as f:
                loaded_bitarray.fromfile(f)
            indices = dict()
            
            # recover indexes from bitarray
            bit_start = 0
            for key in quant_params:
                bit_end = bit_start + vq_args['len'][key]
                n_bit = vq_args['n_bits'][key]
                bits = loaded_bitarray[bit_start:bit_end]
                bit_start = bit_end
                ind = np.reshape(bits.tolist(), (-1, n_bit))
                ind = self._bin2dec(torch.from_numpy(ind), n_bit).cpu().numpy()
                indices[key] = ind
                
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if 'dc' in quant_params:
            features_dc = np.expand_dims(codebook['dc'][indices['dc']].cpu().numpy(), -1)
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if 'sh' in quant_params:
            features_extra = codebook['sh'][indices['sh']].cpu().numpy()
            features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))
            features_extra = features_extra.transpose((0, 2, 1))
        else:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        if 'scale' in quant_params:
            scales = codebook['scale'][indices['scale']].cpu().numpy()
        else:
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if 'rot' in quant_params:
            rots = codebook['rot'][indices['rot']].cpu().numpy()
        else:
            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_nq, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling_nq[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling_nq[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_nq, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,  new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_nq.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        # return tmp_radii

    def prune(self, min_opacity, extent, max_screen_size, radii):
        self.tmp_radii = radii
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_nq.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

