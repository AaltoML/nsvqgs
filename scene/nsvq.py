import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
from bitarray import bitarray
import numpy as np
import os
import json
class QuantizationModule(torch.nn.Module):
    def __init__(self, quant_params: list = ["dc", "sh", "scale", "rot"], size_cb: dict = {"dc": 1024, "sh": 1024, "scale": 1024, "rot": 1024}, device: str = "cpu", last_cb_update = -1, args = None):
        super(QuantizationModule, self).__init__()
        self.nsvqs = torch.nn.ModuleDict()
        self.quant_params = quant_params
        self.attr_maps = {"dc": "features_dc", "sh": "features_rest", "scale": "scaling", "rot": "rotation"}
        self.last_cb_update = last_cb_update
        dim_attr = {"dc": 3, "sh": 45, "scale": 3, "rot": 4}
        self._temp_visiblity = None
        # self.init_mode = 'basic'
        self.init_mode = 'kmeans' # select from ['kmeans', 'basic']
        for key in self.quant_params:
            self.nsvqs[key] = NSVQ(num_embeddings = size_cb[key], embedding_dim = dim_attr[key], device=device, args = args)
            vq = self.nsvqs[key]            
            vq.name = key
        
        # if getattr(args, 'quantization_activation', None):
        self.quantization_activation = getattr(args, 'quantization_activation', None)
            
    def codebooks_initialisation(self, gaussians):
        self.scaling_activation = gaussians.scaling_activation
        self.scaling_inverse_activation = gaussians.scaling_inverse_activation
        print(f'Initializing codebooks by {self.init_mode}...')
        for key in self.quant_params:
            vq = self.nsvqs[key]
            input = getattr(gaussians, f"_{self.attr_maps[key]}").detach()
            input = self._input_preprocess(input , feat = key)
            # if key =='scale':
            #     print(f'initiliaze the codes in activation space for {key}')
            #     vq.activation = gaussians.scaling_activation
            #     vq.inverse_activation = gaussians.scaling_inverse_activation
            vq.codebooks_initialisation(input, mode = self.init_mode)
            print(f'For {key}, codebook size of {vq.num_embeddings} initialized..')
        # breakpoint()
        
            
    def _input_preprocess(self, input, feat):
        if feat == 'sh':
            input = input.reshape(input.shape[0], -1)
        if feat == 'dc':
            input = input.reshape(input.shape[0], -1)
        if feat == 'scale' and self.quantization_activation:
            input = self.scaling_activation(input)
        return input
    def _output_postprocess(self, output, feat):
        if feat == 'sh':
            output = output.reshape(output.shape[0], 15, 3)
        if feat == 'dc':
            output = output.reshape(output.shape[0], 1, 3)
        if feat == 'scale' and self.quantization_activation:
            # print('non-positive values in scaling output', torch.sum(output<1e-40)  )
            # output = torch.clamp(output, min = 1e-40)
            output = self.scaling_inverse_activation(output)
        return output
    def forward(self, gaussians, radii = None):
        # ! find the best parameter for blockwise quantization
        # quantize gaussian attributes, with possible visibility
        torch.cuda.empty_cache()
        for key in self.quant_params:
            # if key in ['hello']:
            vq = self.nsvqs[key]
            input = getattr(gaussians, f"_{self.attr_maps[key]}")
            input = self._input_preprocess(input, feat = key)
            if radii is not None:
                visible = (radii > 0)
                input = input[visible]
                self._temp_visiblity = visible
            else:
                self._temp_visiblity = None
            output, _ = vq(input)
            output = self._output_postprocess(output, feat = key)
            setattr(gaussians, f"_{self.attr_maps[key]}_q", output)
            del output
        torch.cuda.empty_cache()
        
    def fine_tune_assign(self, gaussians, indices):
        for key in self.quant_params:
            vq = self.nsvqs[key]
            output = vq.codebooks[indices[key]]
            output = self._output_postprocess(output, feat = key)
            setattr(gaussians, f"_{self.attr_maps[key]}_q", output)
    
    def inference(self, gaussians, save=False):
        '''test '''
        torch.cuda.empty_cache()
        indices = {}
        for key in self.quant_params:
            vq = self.nsvqs[key]
            input = getattr(gaussians, f"_{self.attr_maps[key]}")
            input = self._input_preprocess(input, feat = key)
            output, min_indices = vq.inference(input)
            output = self._output_postprocess(output, feat = key)
            setattr(gaussians, f"_{self.attr_maps[key]}_q", output)
            indices[key] = min_indices
        if save:
            self.indices = indices
        torch.cuda.empty_cache()
        return indices
    
    def replace_unused_codebooks(self, num_batches):
        unused_count = {}
        for key in self.quant_params:
            vq = self.nsvqs[key]
            k = f'unused_count/{key}'
            unused_count[k] = vq.replace_unused_codebooks(num_batches)
        return unused_count
    
    def save_binary(self, quant_params, path):
        # todo: prepare the quantization saving part...
        bitarray_all = bitarray([])
        vq_args={'n_bits': dict(), 'len': dict(), 'quant_params': quant_params}
        codebook=dict()
        assert set(quant_params) == set(self.quant_params)
        for key in quant_params:
            vq = self.nsvqs[key]
            n_bits = int(np.ceil(np.log2(vq.num_embeddings)))
            assignments = self._dec2binary(self.indices[key], n_bits)
            bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
            bitarray_all.extend(bitarr)
            vq_args['n_bits'][key] = n_bits
            vq_args['len'][key] = len(bitarr)
            codebook[key] = vq.codebooks.detach().cpu()
        with open(os.path.join(path, 'vq_inds.bin'), 'wb') as file:
            bitarray_all.tofile(file)
        np.save(os.path.join(path, 'vq_args.npy'), vq_args)
        torch.save(codebook, os.path.join(path, 'codebook.pth'))
        
        files = [os.path.join(path, f) for f in ['vq_inds.bin', 'vq_args.npy', 'codebook.pth', 'point_cloud.ply'] ]
        file_sizes = [os.path.getsize(f) for f in files]
        assert len(file_sizes) ==4, "the number of files (save codebook) is incorrect"
        num_gs = (self.indices[quant_params[0]]).shape[0]
        info = {'model_size': sum(file_sizes), 'num_gs': num_gs}
        with open(os.path.join(path + "/gs_info.json"), 'w') as fp:
            json.dump(info, fp, indent=True)
        
    def _dec2binary(self, x, n_bits=None):
        """Convert decimal integer x to binary.

        Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        """
        if n_bits is None:
            n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
        mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0)
    

class NSVQ(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=torch.device('cpu'), discarding_threshold=0.01, initialization='normal', args=None):
        super(NSVQ, self).__init__()

        """
        Inputs:
        
        1. num_embeddings = Number of codebook entries
        
        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)
        
        3. device = The device which executes the code (CPU or GPU)
        
        4. discarding_threshold = Percentage threshold for discarding unused codebooks
        
        5. initialization = Initial distribution for codebooks

        """

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.activation = None
        self.inverse_activation = None
        self.activation_init = args.activation_init
        self.activation_distance = args.activation_distance
        if self.activation_init:
            print('activation initialization in cb')
        if self.activation_distance:
            print('activation distance in cb')

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample([self.num_embeddings, self.embedding_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        
        properties = torch.cuda.get_device_properties(device)
        total_memory = properties.total_memory # get the GPU memory in bytes.
        step =  (total_memory - 7 * 1024**3 ) / (self.num_embeddings * 4* 2) #6GB model size, 2 double usage, 4 bytes for floating number   
        self.step = 2**int(np.log2(step))
        # self.step = 2**16
        # Counter variable which contains the number of times each codebook is used
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)
        self.codebooks_freq =  torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)
    
    def __str__(self):
        return f"NSVQ(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, discarding_threshold={self.discarding_threshold}, step=2^{int(np.log2(self.step))})"
    def __repr__(self):
        return self.__str__()
    def codebooks_initialisation(self, input_data, mode = 'basic'):
        '''codebook initialized from input data'''
        print(f'initialize codes for attributes: {self.name}')
        if self.activation_init and self.name =='scale':
            print(f'acitvation func {self.activation}, {self.inverse_activation}')
            input = self.activation(input_data)
        else:
            input = input_data
        with torch.no_grad():
            if mode =='basic':
                codes = self._cb_init_basic(input)
            elif mode == 'kmeans':
                codes = self._cb_init_kmeans(input)
            if self.activation_init and self.name =='scale': 
                codes = torch.clamp(codes,  min = 10**(-8))
                codes = self.inverse_activation(codes)
            self.codebooks *= 0
            self.codebooks += codes
        # breakpoint()
            
    def _cb_init_basic(self, input_data):
        N, D = input_data.shape
        K = self.num_embeddings
        M = N // K
        indices = torch.randperm(N)[ :M*K]
        tmp_input = input_data[indices]
        norms = torch.norm(tmp_input, dim=1)
        sorted_indices = torch.argsort(norms)
        sorted_tensor = tmp_input[sorted_indices]
        avg_tensor = torch.mean(sorted_tensor.view(K, M, D), dim = 1)
        return avg_tensor
    
    def _cb_init_kmeans(self, input_data):
        N, D = input_data.shape
        K = self.num_embeddings
        X = input_data
        kmeans = KMeansTorch(n_clusters=K, max_iter = 10, step = self.step)
        kmeans.fit(X)
        return kmeans.centroids
    def get_codebooks(self):
        return self.codebooks
    
    def _input_quantization(self, input_data):
        """ update input data to be quantized version
        Args:
            input_data (_type_): _description_
        return the indice of the quantization
        """
        codebooks =  self.get_codebooks()
        N = len(input_data)
        step = self.step # hyperparameters for blocking the input 
        min_indices = []
        for i in range(0, N, step):
            batch_data = input_data[i:i+step]
            if self.activation_distance and self.name =='scale':
                distances = torch.cdist(self.activation(batch_data), self.activation(codebooks))
            else:
                distances = torch.cdist(batch_data, codebooks)
            min_indices.append(torch.argmin(distances, dim=1))
            del distances
        return torch.cat(min_indices, dim=0)
        
        
    def forward(self, input_data, mode = 'nsvq'):

        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """
        
        # compute the distances between input and codebooks vectors for the code indices
        codebooks =  self.get_codebooks()
        min_indices = self._input_quantization(input_data.detach())
        hard_quantized_input = codebooks[min_indices]
        
        if mode =='nsvq':
            random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(self.device)
            norm_quantization_residual = torch.linalg.norm(input_data - hard_quantized_input, dim = 1).view(-1,1)
            norm_random_vector = torch.linalg.norm(random_vector, dim = 1).view(-1,1)
            # defining vector quantization error
            vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector
            
            quantized_input = input_data + vq_error
            with torch.no_grad():
                self.codebooks_used[min_indices] += 1
                unique_values, counts = torch.unique(min_indices, return_counts=True)
                self.codebooks_freq[unique_values] +=  counts
            return quantized_input, self.codebooks_used.cpu().numpy()


    def replace_unused_codebooks(self, num_batches):

        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        For more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
         replaced codebooks might increase. However, the main trend must be decreasing after some training time.
         If it is not the case for you, increase the "num_batches" or decrease the "discarding_threshold" to make
         the trend for number of replacements decreasing. Stop calling the function at the latest stages of training
         in order not to introduce new codebook entries which would not have the right time to be tuned and optimized
         until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                # print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                
                # used codes with lower usage gives higher chance to be used
                # used_freq = (self.codebooks_freq.cpu())[used_indices]
                # freq = sum(used_freq) / used_freq
                # freq = (freq/ sum(freq)).tolist()
                # freq[-1] = 1- sum(freq[:-1])
                # index = np.random.choice(used_count, unused_count, p=freq)
                # used_codebooks = used[index]
                # original random use
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used
                
                new_codes = used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()
                
                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += new_codes
            # print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0
            self.codebooks_freq[:] = 0.0
        return unused_count

    def inference(self, input_data):

        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """
        codebooks =  self.get_codebooks()
        with torch.no_grad():
            min_indices = self._input_quantization(input_data)
            quantized_input = codebooks.detach()[min_indices]

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input, min_indices
    
    # def fine_tune_assign(self, input_data):
    #     min_indices = self._input_quantization(input_data)
    #     quantized_input = self.codebooks.detach()[min_indices]
    #     return quantized_input, min_indices





class KMeansTorch:
    def __init__(self, n_clusters: int, max_iter: int = 10, tol: float = 1e-4, device: str = "cuda", step =2**16):
        """
        KMeans clustering using PyTorch.
        :param n_clusters: Number of clusters
        :param max_iter: Maximum number of iterations
        :param tol: Tolerance for convergence
        :param device: Device to use ('cuda' or 'cpu')
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.centroids = None

    def _get_labels(self, X):
        """ update input data to be quantized version
        Args:
            input_data (_type_): _description_
        return the indice of the quantization
        """
        
        N = len(X)
        step = 2**16 # 16384 hyperparameters for blocking the input 
        min_indices = []
        for i in range(0, N, step):
            batch_data = X[i:i+step]
            distances = torch.cdist(batch_data, self.centroids)
            min_indices.append(torch.argmin(distances, dim=1))
            del distances
        return torch.cat(min_indices, dim=0)
    
    def fit(self, X: torch.Tensor):
        """
        Fit the KMeans model to the data.
        :param X: Input data (N x D) tensor
        """
        X = X.to(self.device)
        n_samples, n_features = X.shape

        # Randomly initialize centroids
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices]

        for i in range(self.max_iter):
            # Compute distances from data points to centroids
            # distances = torch.cdist(X, self.centroids)

            # Assign each point to the nearest centroid
            # labels = torch.argmin(distances, dim=1)
            labels = self._get_labels(X)

            # Compute new centroids
            new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(self.n_clusters)])
            
            # Handle empty clusters
            for k in range(self.n_clusters):
                if torch.sum(labels == k) == 0:
                    new_centroids[k] = self.centroids[k]

            # Check for convergence
            if torch.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X: torch.Tensor):
        """
        Predict the cluster for each data point.
        :param X: Input data (N x D) tensor
        :return: Cluster labels
        """
        X = X.to(self.device)
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)
