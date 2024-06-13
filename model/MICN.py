import torch
import torch.nn as nn
from utils.layers import DataEmbedding, series_decomp, series_decomp_multi
import torch.nn.functional as F

class MIC(nn.Module):
    """
    The MIC (Multi-scale Integration Convolution) module designed for time series forecasting, 
    leveraging the power of convolutional neural networks to capture multi-scale temporal dependencies. 
    It integrates various convolution operations, including downsampling, isometric, and upsampling convolutions, 
    to process data at different scales and extract relevant features for forecasting.

    Attributes:
    - feature_size (int): The size of the input and output features for the convolutions.
    - dropout (float): The dropout rate applied to prevent overfitting.
    - decomp_kernel (list of int): The kernel sizes for the series decomposition layers, used to preprocess the input data by decomposing it into trend and seasonal components.
    - conv_kernel (list of int): The kernel sizes for the downsampling convolutional layers, which reduce the dimensionality of the data to capture lower-level features.
    - isometric_kernel (list of int): The kernel sizes for the isometric convolutional layers, designed to process the data at its original scale without changing its dimensionality.
    - device (str): The device (e.g., 'cuda' or 'cpu') on which the model computations will be performed.

    Methods:
    - __init__(self, feature_size=512, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6], device='cuda'): Initializes the MIC model with the specified parameters.
    - conv_trans_conv_ours(self, input, conv1d, conv1d_trans, isometric):
    Applies a sequence of convolution operations on the input tensor, including downsampling, isometric, and upsampling convolutions, with residual connections and normalization.

    The MIC class is suited for the analysis of complex time series data across multiple scales thanks to the integration of different convolution operations,
    such as financial market predictions, weather forecasting, and demand forecasting in supply chain management.
    """

    def __init__(self, feature_size=512, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = nn.LayerNorm(feature_size)
        self.act =  nn.Tanh()
        self.drop = nn.Dropout(0.05)

    def conv_trans_conv_ours(self, input, conv1d, conv1d_trans, isometric):
        
        """
    Performs a sequence of convolution operations on the input tensor, designed for
    multi-scale processing for the MIC (Multi-scale Integration Convolution) layer. This function
    implements downsampling, isometric, and upsampling convolutions
    with residual connections and normalization.

    Parameters:
    - input (torch.Tensor): The input tensor with shape [batch_size, sequence_length, channels].
    - conv1d (torch.nn.modules.conv.Conv1d): The 1D convolutional layer used for downsampling.
    - conv1d_trans (torch.nn.modules.conv.ConvTranspose1d): The 1D transposed convolutional layer used for upsampling.
    - isometric (torch.nn.modules.conv.Conv1d): The 1D convolutional layer used for isometric convolution.

    The process involves the following steps:
    1. Downsampling convolution with a residual connection: Applies `conv1d` to the input tensor and
       activates the result using a non-linear activation function. The result is then passed through
       a dropout layer.
    2. Isometric convolution with zero-padding: Zero-pads the result of the downsampling convolution
       on one side and then applies the `isometric` convolution. The result is activated, passed through
       a dropout layer, and combined with the result from the downsampling step.
    3. Upsampling convolution with truncation: Applies `conv1d_trans` to the combined tensor from the
       previous step, activates the result, and truncates it to match the original sequence length.
    4. Residual connection and normalization: Adds the result from the upsampling step to the original
       input tensor and normalizes the result.

    Returns:
    - torch.Tensor: The output tensor after applying the sequence of convolutions, with the same shape
      as the input tensor [batch_size, sequence_length, channels].
    
    ---------------------------------------------------------------------------------------------
    
    The purpose of the isometric convolution in the conv_trans_conv_ours function is to efficiently
    model global correlations within the input tensor without the computational burden of attention mechanisms.
    This is achieved through a specific convolution operation that is designed to handle multi-scale processing 
    for the MIC (Multi-scale Integration Convolution) layer. The isometric convolution is applied after
    downsampling the input tensor and involves zero-padding the result of the downsampling convolution 
    on one side before applying the isometric convolution. This step is crucial for capturing global 
    features in the data while maintaining computational efficiency, especially in the context of long-term time series forecasting
    where capturing both local and global patterns is essential.
    """
    
        batch_size, seq_len, channels = input.shape
        x = input.permute(0, 2, 1)

        # Downsampling convolution with residual connection
        x_r = conv1d(x)
        x_r = self.act(x_r)
        x_r = self.drop(x_r)
        x_iso = x_r

        # Isometric convolution with zero-padding
        zeros = torch.zeros((x_iso.shape[0], x_iso.shape[1], x_iso.shape[2] - 1), device=self.device)
        padded_input = torch.cat((zeros, x_iso), dim=-1)
        x_iso = isometric(padded_input)
        x_iso = self.act(x_iso)
        x_iso = self.drop(x_iso)

        # Combine and transform back to original shape
        x_up = self.norm((x_iso + x_r).permute(0, 2, 1)).permute(0, 2, 1)

        # Upsampling convolution with truncation
        x_up = conv1d_trans(x_up)
        x_up = self.act(x_up)
        x_up = self.drop(x_up)
        x_up = x_up[:, :, :seq_len]  # Truncate

        # Residual connection and normalization
        x = x_up.permute(0, 2, 1) + input # [B, L, D]
        x = self.norm(x)
        return x
    
    def forward(self, src):
        
        """
        Forward pass of the MIC (Multi-scale Integration Convolution) layer, which processes the input tensor
        using a sequence of convolution operations to capture multi-scale temporal dependencies.
        """
        # Multi-scale processing
        multi_scale_outputs = [
            self.conv_trans_conv_ours(self.decomp[i](src)[0], self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            for i in range(len(self.conv_kernel))
        ]

        # Merge multi-scale outputs
        merged_output = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            merged_output = torch.cat((merged_output, multi_scale_outputs[i].unsqueeze(1)), dim=1)
        merged_output = self.merge(merged_output.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        # Final prediction steps using feedforward network
        normalized_output = self.norm1(merged_output)
        processed_output = self.conv1(normalized_output.transpose(-1, 1))
        processed_output = self.conv2(processed_output).transpose(-1, 1) 
        return self.norm2(merged_output + processed_output) 

class SeasonalPrediction(nn.Module):
    
    """
    The SeasonalPrediction module is designed to predict seasonal patterns in time series data
    using a combination of Multi-scale Integration Convolution (MIC) layers and linear projection
    using a fully connected layer.
    """
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, decomp_kernel=decomp_kernel, 
                                      conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, 
                                      device=device)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)

class Model(nn.Module):
    
    """
    The Model class defines the architecture of the MICN (Multi-scale Integration Convolution Network)
    model for time series forecasting. It combines the MIC (Multi-scale Integration Convolution) module
    with a linear regression layer to predict the trend component of the time series data and generate
    long-term forecasts. The model also includes an imputation mode for missing data imputation tasks.
    
    attributes:
    - task_name (str): The name of the forecasting task ('long_term_forecast' or 'imputation').
    - pred_len (int): The prediction length for the forecasting task.
    - seq_len (int): The sequence length of the input data.
    
    methods:
    - forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec): Performs long-term forecasting using the MICN model.
    - imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask): Performs data imputation using the MICN model.
    - forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None): Implements the forward pass of the MICN model.
    
    """
    
    def __init__(self, configs, conv_kernel=[12, 16]):
        """
        conv_kernel: downsampling and upsampling convolution kernel_size
        """
        super(Model, self).__init__()
        
        # Initialize lists for the kernels of the decomposition operation and isometric convolution
        decomposition_kernels = []
        isometric_kernels = []

        # Iterate over each kernel in the convolution kernels
        for kernel in conv_kernel:
            # If the kernel is even, it needs to be incremented by 1 because decomposition kernels must be odd
            if kernel % 2 == 0:
                decomposition_kernels.append(kernel + 1)
                isometric_kernels.append((configs.seq_len + configs.pred_len + kernel) // kernel)
            else:
                # If the kernel is already odd, it can be used as is
                decomposition_kernels.append(kernel)
                isometric_kernels.append((configs.seq_len + configs.pred_len + kernel - 1) // kernel)

        self.task_name = configs.task_name #forecasting or imputation
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomposition_kernels)

        # embedding
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model,
                                             n_heads=configs.n_heads,
                                             dropout=configs.dropout,
                                             d_layers=configs.d_layers,
                                             decomp_kernel=decomposition_kernels,
                                             c_out=configs.c_out,
                                             conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernels,
                                             device=torch.device('cuda'))
        if self.task_name == 'long_term_forecast':
            self.regression = nn.Linear(configs.seq_len, configs.pred_len)
            self.regression.weight = nn.Parameter(
                (1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]),
                requires_grad=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_components, trend = self.decomp_multi(x_enc)  

        # Trend Prediction (using regression)
        trend_prediction = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # Prepare decoder input
        seasonal_init_dec = torch.cat([seasonal_components[:, -self.seq_len:, :], 
                                    torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],  
                                                device=x_enc.device)], dim=1)

        # Embedding and Convolutional Processing
        embedded_input = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        conv_output = self.conv_trans(embedded_input)

        # Combine and extract prediction horizon
        final_prediction = conv_output[:, -self.pred_len:, :] + trend_prediction[:, -self.pred_len:, :]
        return final_prediction
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Multi-scale Hybrid Decomposition
        seasonal_components, trend = self.decomp_multi(x_enc)

        # Embedding and Convolutional Processing
        embedded_input = self.dec_embedding(seasonal_components, x_mark_dec) 
        conv_output = self.conv_trans(embedded_input) 

        # Combine and extract prediction horizon
        imputed_output = conv_output + trend 
        return imputed_output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out # [B, L, D]