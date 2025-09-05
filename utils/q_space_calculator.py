"""
Q-space calculation utilities for GISAXS analysis.
"""

import numpy as np
import hashlib


class Detector:
    """
    Detector class for calculating q-vectors in GISAXS geometry.
    """
    def __init__(self, detector_params, beam_center, distance, theta_in_deg, wavelength):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        detector_params : list
            [Nx, width_x, Ny, width_y] in mm
        beam_center : list
            [x, y] beam center position in mm
        distance : float
            Distance from sample to detector in mm
        theta_in_deg : float
            Incident angle in degrees
        wavelength : float
            X-ray wavelength in nm
        """
        self.detector = detector_params
        self.beam_center = beam_center.copy()  # Make a copy to avoid modifying original
        self.distance = distance
        self.theta_in = np.deg2rad(theta_in_deg)  # Convert to radians
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength  # Wavevector

        # Update the beam center by the theta_in
        self.beam_center[1] += self.distance * np.tan(self.theta_in)
        
        # Cache for calculated q-vectors
        self._q_cache = None
        self._q_cache_hash = None

    def _get_parameter_hash(self):
        """Generate a hash for current parameters to check if recalculation is needed."""
        params_str = f"{self.detector}_{self.beam_center}_{self.distance}_{self.theta_in}_{self.wavelength}"
        return hashlib.md5(params_str.encode()).hexdigest()

    def calculate_q_vectors(self, force_recalculate=False):
        """
        Calculate q-vectors for each detector pixel.
        
        Parameters:
        -----------
        force_recalculate : bool
            If True, force recalculation even if cached values exist
        
        Returns:
        --------
        qx, qy, qz, qr : ndarray
            Q-vector components and magnitude
        """
        # Check if we can use cached values
        current_hash = self._get_parameter_hash()
        if not force_recalculate and self._q_cache is not None and self._q_cache_hash == current_hash:
            return self._q_cache
        
        # Calculate pixel positions on the detector
        Nx, width_x, Ny, width_y = self.detector
        x = np.linspace(0, width_x, Nx) - self.beam_center[0]
        y = np.linspace(0, width_y, Ny) - self.beam_center[1]
        X, Y = np.meshgrid(x, y)

        # Calculate the angles psi and theta_sc for each detector pixel
        theta_sc = np.arctan2(Y, self.distance)  # Scattering angle
        psi = np.arctan2(X, self.distance)  # Azimuthal angle

        # Calculate qx, qy, qz using the provided formulas
        qx = self.k0 * (np.cos(theta_sc) * np.cos(psi) - np.cos(self.theta_in))
        qy = self.k0 * (np.cos(theta_sc) * np.sin(psi))
        qz = self.k0 * (np.sin(theta_sc) + np.sin(self.theta_in))

        # Calculate qr, concern the + and - sign
        qr = np.sqrt(qx**2 + qy**2)
        qr = np.copysign(qr, qy)

        # Flip up the qz image
        qz = np.flipud(qz)

        # Cache the results
        self._q_cache = (qx, qy, qz, qr)
        self._q_cache_hash = current_hash
        
        return qx, qy, qz, qr
    
    def get_qr(self):
        """Get qr component only."""
        _, _, _, qr = self.calculate_q_vectors()
        return qr
    
    def get_qz(self):
        """Get qz component only."""
        _, _, qz, _ = self.calculate_q_vectors()
        return qz
    
    def pixel_to_q_space(self, x_pixel, y_pixel):
        """
        将detector像素坐标转换为q空间坐标
        
        Parameters:
        -----------
        x_pixel : float or array
            像素坐标x (水平方向)
        y_pixel : float or array
            像素坐标y (垂直方向)
            
        Returns:
        --------
        qx, qy, qz : float or array
            对应的q空间坐标 (nm^-1)
        """
        # 确保输入是numpy数组
        x_pixel = np.asarray(x_pixel)
        y_pixel = np.asarray(y_pixel)
        
        # 获取detector参数
        Nx, width_x, Ny, width_y = self.detector
        
        # 将像素坐标转换为物理坐标 (mm)
        # x方向：从pixel转换为物理坐标
        x_physical = (x_pixel / Nx) * width_x - self.beam_center[0]
        # y方向：从pixel转换为物理坐标  
        y_physical = (y_pixel / Ny) * width_y - self.beam_center[1]
        
        # 计算散射角度
        theta_sc = np.arctan2(y_physical, self.distance)  # 散射角
        psi = np.arctan2(x_physical, self.distance)  # 方位角
        
        # 计算q向量分量
        qx = self.k0 * (np.cos(theta_sc) * np.cos(psi) - np.cos(self.theta_in))
        qy = self.k0 * (np.cos(theta_sc) * np.sin(psi))
        qz = self.k0 * (np.sin(theta_sc) + np.sin(self.theta_in))
        
        return qx, qy, qz
    
    def q_to_pixel_space(self, qx, qy, qz):
        """
        将q空间坐标转换为detector像素坐标
        
        Parameters:
        -----------
        qx, qy, qz : float or array
            q空间坐标 (nm^-1)
            
        Returns:
        --------
        x_pixel, y_pixel : float or array
            对应的像素坐标
        """
        # 确保输入是numpy数组
        qx = np.asarray(qx)
        qy = np.asarray(qy)
        qz = np.asarray(qz)
        
        # 从q向量反推散射角度
        # qx = k0 * (cos(theta_sc) * cos(psi) - cos(theta_in))
        # qy = k0 * (cos(theta_sc) * sin(psi))
        # qz = k0 * (sin(theta_sc) + sin(theta_in))
        
        # 先计算theta_sc (散射角)
        sin_theta_sc = (qz / self.k0) - np.sin(self.theta_in)
        cos_theta_sc = np.sqrt(1 - sin_theta_sc**2)
        
        # 计算psi (方位角)
        # 从qy计算: qy = k0 * cos(theta_sc) * sin(psi)
        sin_psi = qy / (self.k0 * cos_theta_sc)
        cos_psi = np.sqrt(1 - sin_psi**2)
        
        # 验证qx的一致性并确定cos_psi的符号
        qx_calc = self.k0 * (cos_theta_sc * cos_psi - np.cos(self.theta_in))
        # 如果计算的qx与输入qx符号不同，则cos_psi应该取负值
        mask = (qx_calc * qx) < 0
        cos_psi = np.where(mask, -cos_psi, cos_psi)
        
        # 重新计算角度
        theta_sc = np.arctan2(sin_theta_sc, cos_theta_sc)
        psi = np.arctan2(sin_psi, cos_psi)
        
        # 转换为物理坐标 (mm)
        x_physical = self.distance * np.tan(psi)
        y_physical = self.distance * np.tan(theta_sc)
        
        # 获取detector参数
        Nx, width_x, Ny, width_y = self.detector
        
        # 转换为像素坐标
        x_pixel = ((x_physical + self.beam_center[0]) / width_x) * Nx
        y_pixel = ((y_physical + self.beam_center[1]) / width_y) * Ny
        
        return x_pixel, y_pixel

    def get_qx(self):
        """Get qx component only."""
        qx, _, _, _ = self.calculate_q_vectors()
        return qx
    
    def get_qy(self):
        """Get qy component only."""
        _, qy, _, _ = self.calculate_q_vectors()
        return qy  
    
    def get_q(self):
        """Get total q magnitude."""
        qx, qy, qz, _ = self.calculate_q_vectors()
        return np.sqrt(qx**2 + qy**2 + qz**2)
    
    def get_qy_qz_meshgrids(self):
        """
        Get qy and qz meshgrids for pcolormesh plotting.
        
        Returns:
        --------
        qy_mesh, qz_mesh : ndarray
            2D meshgrids of qy and qz coordinates
        """
        _, qy, qz, _ = self.calculate_q_vectors()
        return qy, qz
    
    def get_q_extents(self):
        """
        Get the extents of q-space for plotting.
        
        Returns:
        --------
        qr_extent : list
            [qr_min, qr_max] for horizontal axis
        qz_extent : list
            [qz_min, qz_max] for vertical axis
        """
        qr = self.get_qr()
        qz = self.get_qz()
        
        qr_extent = [qr.min(), qr.max()]
        qz_extent = [qz.min(), qz.max()]
        
        return qr_extent, qz_extent
    
    def clear_cache(self):
        """Clear the cached q-vectors."""
        self._q_cache = None
        self._q_cache_hash = None


def create_detector_from_image_and_params(image_shape, pixel_size_x, pixel_size_y, 
                                        beam_center_x, beam_center_y, distance, 
                                        theta_in_deg, wavelength, crop_params=None):
    """
    Create a Detector object from image dimensions and experimental parameters.
    
    Parameters:
    -----------
    image_shape : tuple
        (height, width) of the image in pixels
    pixel_size_x, pixel_size_y : float
        Pixel size in micrometers
    beam_center_x, beam_center_y : float
        Beam center in pixels
    distance : float
        Sample-to-detector distance in mm
    theta_in_deg : float
        Incident angle in degrees
    wavelength : float
        X-ray wavelength in nm
    crop_params : list or None
        [left, right, top, bottom] crop parameters in pixels
        
    Returns:
    --------
    detector : Detector
        Configured Detector object
    """
    height, width = image_shape
    
    # Convert pixel size from micrometers to mm
    pixel_size_x_mm = pixel_size_x / 1000.0
    pixel_size_y_mm = pixel_size_y / 1000.0
    
    # Apply crop parameters if provided
    if crop_params is not None:
        crop_left, crop_right, crop_top, crop_bottom = crop_params
        width = width - crop_left - crop_right
        height = height - crop_top - crop_bottom
        beam_center_x = beam_center_x - crop_left
        beam_center_y = beam_center_y - crop_top
    
    # Calculate detector parameters
    detector_params = [
        width,  # Nx
        width * pixel_size_x_mm,  # width_x in mm
        height,  # Ny  
        height * pixel_size_y_mm  # width_y in mm
    ]
    
    # Calculate beam center in mm
    beam_center = [
        beam_center_x * pixel_size_x_mm,
        beam_center_y * pixel_size_y_mm
    ]
    
    return Detector(detector_params, beam_center, distance, theta_in_deg, wavelength)


def get_q_axis_labels_and_extents(detector):
    """
    Get appropriate axis labels and extents for q-space plotting.
    
    Parameters:
    -----------
    detector : Detector
        Configured Detector object
        
    Returns:
    --------
    xlabel : str
        Label for x-axis (qr)
    ylabel : str  
        Label for y-axis (qz)
    extent : list
        [qr_min, qr_max, qz_min, qz_max] for imshow extent
    """
    qr_extent, qz_extent = detector.get_q_extents()
    
    xlabel = r'$q_r$ (nm$^{-1}$)'
    ylabel = r'$q_z$ (nm$^{-1}$)'
    
    # For imshow, extent is [left, right, bottom, top]
    extent = [qr_extent[0], qr_extent[1], qz_extent[0], qz_extent[1]]
    
    return xlabel, ylabel, extent


class Mask:
    """
    Mask generation class for GISAXS data processing.
    """
    def __init__(self, detector, beam_center, distance, theta_in_deg, wavelength):
        """
        Initialize the mask generator.
        
        Parameters same as Detector class.
        """
        self.detector = detector
        self.beam_center = beam_center
        self.distance = distance
        self.theta_in = np.deg2rad(theta_in_deg)  # Convert to radians
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength  # Wavevector

        # Update the beam center by the theta_in
        self.beam_center[1] += self.distance * np.tan(self.theta_in)

    def mask_center(self, radius=5, bar='False'):
        """Create a mask around the beam center."""
        beam_center_pixel = [int(self.beam_center[0] / self.detector[1] * self.detector[0]),
                             self.detector[2] - int(self.beam_center[1] / self.detector[3] * self.detector[2])]
        mask = np.ones((self.detector[2], self.detector[0]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - beam_center_pixel[1])**2 + (j - beam_center_pixel[0])**2 < radius**2:
                    mask[i, j] = 0

        if bar == 'True':
            # Define a bar shape
            bar_para = np.array([[beam_center_pixel[1], beam_center_pixel[0], 2.5, np.deg2rad(np.random.randint(-90, 90))]])
            cos_angle = np.cos(bar_para[:, 3])
            sin_angle = np.sin(bar_para[:, 3])

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    dx = i - bar_para[:, 0]
                    dy = j - bar_para[:, 1]

                    rotated_x = cos_angle * dx + sin_angle * dy
                    rotated_y = -sin_angle * dx + cos_angle * dy

                    if abs(rotated_y) < bar_para[:, 2] / 2 and rotated_x >= 0:
                        mask[i, j] = 0

        return mask.astype(bool)

    def mask_reflect(self, incident_angle, radius=5, bar='False'):
        """Create a mask for the reflected beam."""
        beam_center_pixel_re = [int(self.beam_center[0] / self.detector[1] * self.detector[0]),
                             self.detector[2] - int(self.beam_center[1] / self.detector[3] * self.detector[2] + self.distance * 2 * np.tan(incident_angle))]
        mask = np.ones((self.detector[2], self.detector[0]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - beam_center_pixel_re[1])**2 + (j - beam_center_pixel_re[0])**2 < radius**2:
                    mask[i, j] = 0
        if bar == 'True':
            bar_para = np.array([[beam_center_pixel_re[1], beam_center_pixel_re[0], 2.5, np.deg2rad(np.random.randint(90, 270))]])
            cos_angle = np.cos(bar_para[:, 3])
            sin_angle = np.sin(bar_para[:, 3])

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    dx = i - bar_para[:, 0]
                    dy = j - bar_para[:, 1]

                    rotated_x = cos_angle * dx + sin_angle * dy
                    rotated_y = -sin_angle * dx + cos_angle * dy

                    if abs(rotated_y) < bar_para[:, 2] / 2 and rotated_x >= 0:
                        mask[i, j] = 0

        return mask.astype(bool)

    def mask_bs_line(self, width=5):
        """Create a beamstop line mask."""
        position = int(self.beam_center[0] / self.detector[1] * self.detector[0])
        mask = np.ones((self.detector[2], self.detector[0]))
        mask[:, position - width // 2:position + width // 2] = 0
        return mask.astype(bool)

    def mask_gap(self, v_number, h_number, gap_size, model='random'):
        """Create gap masks."""
        if model == 'random':
            position_v = np.random.randint(0, self.detector[0] - gap_size - 1, v_number)
            position_h = np.random.randint(0, self.detector[2] - gap_size - 1, h_number)
            mask = np.ones((self.detector[2], self.detector[0]))
        elif model == 'uniform':
            v_number = v_number + 1
            position_v = np.linspace(gap_size, self.detector[0], v_number, endpoint=False).astype(int)
            position_v = np.delete(position_v, 0)
            v_number = v_number - 1
            h_number = h_number + 1
            position_h = np.linspace(0, self.detector[2], h_number, endpoint=False).astype(int)
            position_h = np.delete(position_h, 0)
            h_number = h_number - 1
            mask = np.ones((self.detector[2], self.detector[0]))
        for i in range(v_number):
            mask[position_v[i]:position_v[i] + gap_size, :] = 0
        for i in range(h_number):
            mask[:, position_h[i]:position_h[i] + gap_size] = 0
        return mask.astype(bool)

    def add_mask(self, prob=0.5):
        """Add combined masks."""
        mask = self.mask_center(np.random.randint(5, 10), bar='True')

        if np.random.rand() < prob:
            mask &= self.mask_reflect(self.theta_in, np.random.randint(5, 6), bar='True')

        if np.random.rand() < prob:
            mask &= self.mask_bs_line(np.random.randint(3, 8))

        if np.random.rand() < prob:
            mask &= self.mask_gap(np.random.randint(1, 4), 1, gap_size=np.random.randint(3, 5), model='uniform')

        if np.random.rand() < prob:
            mask &= self.mask_gap(0, 1, gap_size=np.random.randint(3, 5), model='random')

        return mask
