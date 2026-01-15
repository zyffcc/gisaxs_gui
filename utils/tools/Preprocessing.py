import numpy as np
from scipy.ndimage import shift

class Preprocessing:
    def __init__(self, data):
        self.data = np.array(data)
        self.data = self.data.squeeze().astype(np.float32)
# %%

    # def log_and_normalize(self):
    #     '''
    #     Logarithmic normalization algorithm, 
    #     Input data: 
    #     1. Single image, shape = (h, w); 
    #     2. Multiple images, shape = (n, h, w), where n is the number of images.
    #     Returns: normalized_data 
    #     1. Single image, shape = (h, w); 
    #     2. Multiple images, shape = (n, h, w), where n is the number of images.
    #     '''
    #     # logarithmic
    #     log_data = np.log(self.data + 1e-10)  # Add a small value to avoid taking logarithms to zero.

    #     # normalize to [0, 1]
    #     if self.data.ndim == 2: # For single image

    #         min_val = np.min(log_data)
    #         max_val = np.max(log_data)
    #         # normalized_data = (log_data - min_val) / (max_val - min_val)
    #         normalized_data = log_data / max_val
    #     elif self.data.ndim == 3: # For multiple images, shape = (n, h, w), where n is the number of images.
    #         min_val = np.min(log_data, axis=(1, 2)).reshape(-1, 1, 1)
    #         max_val = np.max(log_data, axis=(1, 2)).reshape(-1, 1, 1)
    #         # normalized_data = (log_data - min_val) / (max_val - min_val)
    #         normalized_data = log_data / max_val
    #     else:
    #         raise ValueError("The dimension of the input data is not supported.")
        
    #     return normalized_data
    def log_and_normalize(self):
        '''
        Logarithmic normalization with max-value scaling before log.
        Input:
            - Single image: shape = (h, w)
            - Multiple images: shape = (n, h, w)
        Returns:
            - Normalized image(s) with values in [0, 1], same shape as input.
        '''
        scaled_data = np.array(self.data, dtype=np.float32)

        if self.data.ndim == 2:
            temp = scaled_data.copy()
            temp[:,0:10] = -1
            temp[240:256,:] = -1
            scale_factor = np.exp(1) / (np.max(temp) + 1e-8)
            # scale_factor = np.exp(1) / (np.max(scaled_data) + 1e-8)
            scaled_data = scaled_data * scale_factor
            log_data = np.log(scaled_data + 1e-8)
            log_data[np.isnan(log_data)] = -1
            
        elif self.data.ndim == 3:
            max_val = np.max(scaled_data, axis=(1, 2), keepdims=True)
            scale_factor = np.exp(1) / (max_val + 1e-8)
            scaled_data = scaled_data * scale_factor
            log_data = np.log(scaled_data + 1e-8)
            
        else:
            raise ValueError("The dimension of the input data is not supported.")
        
        return log_data

    def add_gaussian_noise(self, snr=60, mode='normal'):
        """
        Adds Gaussian noise to the input data.
        
        Parameters.
        - use_max_value: whether to use the maximum value of the data to calculate the noise intensity. If False, the average value will be used.
        - snr: signal-to-noise ratio, default is 10000.
        
        Returns: data_noisy
        - data_noisy: output data with Gaussian noise added.
        """
        if mode == 'normal':
            # calculate the Signal Power
            signal_power = np.sum(self.data ** 2) / self.data.size

            # Calculate the required noise power
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            
            # 生成高斯噪声
            noise = np.random.normal(0, np.sqrt(noise_power), self.data.shape)
            
            # 添加噪声到图像
            noisy_image = self.data + noise
        elif mode == 'avoid_negative':
            # calculate the Signal Power
            signal_power = np.sum(self.data ** 2) / self.data.size

            # Calculate the required noise power
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            
            # 生成高斯噪声
            noise = np.random.normal(0, np.sqrt(noise_power), self.data.shape)

            noise[self.data < 0] = 0
            # 添加噪声到图像
            noisy_image = self.data + noise
    
        return np.abs(noisy_image)

    def add_poisson_noise(self, scale=1.0, mode='avoid_negative'):
        """
        Adds Poisson noise to the input data.
        
        Parameters
        ----------
        scale : float, optional
            Scaling factor for input data before generating Poisson noise. 
            Higher scale -> smaller relative noise. Default is 1.0.
        mode : str, optional
            'normal' — directly apply Poisson noise.
            'avoid_negative' — ensure output values are non-negative.
            
        Returns
        -------
        noisy_image : ndarray
            Output data with Poisson noise added.
        """
        # 防止负数或极小值
        data_clipped = np.clip(self.data, 0, None)
        
        # 为了控制噪声强度，可以通过 scale 调整光子计数水平
        scaled_data = data_clipped * scale
        
        # 生成泊松噪声 (逐像素)
        noisy_scaled = np.random.poisson(scaled_data).astype(np.float64)
        
        # 还原缩放
        noisy_image = noisy_scaled / scale
        
        if mode == 'avoid_negative':
            noisy_image = np.clip(noisy_image, 0, None)
        
        return noisy_image


    def translate_pattern(self, shift_x=np.random.uniform(-40,40), shift_y=np.random.uniform(-20,5)):
        # Translate the pattern by the specified shifts
        translated_pattern = shift(self.data, shift=[shift_y, shift_x], mode='nearest',prefilter=False, order=0)
        return translated_pattern

    def crop_pattern(self):
        pattern = self.data
        # set the edge of the pattern to zero, width is np.random.randint(0, 20) pixels
        width1 = np.random.randint(1, 20)
        width2 = np.random.randint(1, 20)
        width3 = np.random.randint(1, 20)
        width4 = np.random.randint(1, 20)
        pattern[:width1, :] = -1
        pattern[-width2:, :] = -1
        pattern[:, :width3] = -1
        pattern[:, -width4:] = -1
        if np.sum(pattern) == 0:
            return crop_pattern()
        return pattern

    def rotate_pattern(self):
        # Rotate the pattern by a random angle
        angle = np.random.uniform(0, 360)
        rotated_pattern = np.rot90(self.data, k=int(angle / 90))
        return rotated_pattern


        # mean = 0.0  # 高斯噪音的均值

        # # Based on the input data, the intensity I
        # if use_max_value:
        #     intensity = np.max(self.data)
        # else:
        #     intensity = np.mean(self.data)
        
        # # Calculate standard deviation
        # std = np.sqrt(intensity / snr)
        
        # # Generating Gaussian Noise
        # noise = np.random.normal(mean, std, self.data.shape)
        
        # # Adding noise to the data
        # data_noisy = self.data + np.abs(noise)
        
        # return data_noisy

    