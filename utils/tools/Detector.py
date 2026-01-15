import numpy as np

class Detector:
    def __init__(self, detector, beam_center, distance, theta_in_deg, wavelength):
        self.detector = detector
        self.beam_center = beam_center
        self.distance = distance
        self.theta_in = np.deg2rad(theta_in_deg)  # Convert to radians
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength  # Wavevector

        # Update the beam center by the theta_in
        self.beam_center[1] += self.distance * np.tan(self.theta_in)

    def calculate_q_vectors(self):
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

        return qx, qy, qz, qr
    
    def get_qr(self):
        # qx qy qz is none
        _, _, _, qr = self.calculate_q_vectors()
        return qr
    
    def get_qz(self):
        # qx qy qz is none
        _, _, qz, _ = self.calculate_q_vectors()
        return qz
    
    def get_qx(self):
        # qx qy qz is none
        qx, _, _, _ = self.calculate_q_vectors()
        return qx
    
    def get_qy(self):
        # qx qy qz is none
        _, qy, _, _ = self.calculate_q_vectors()
        return qy  
    
    def get_q(self):
        # qx qy qz is none
        qx, qy, qz, _ = self.calculate_q_vectors()
        return np.sqrt(qx**2 + qy**2 + qz**2)


class Mask:
    def __init__(self, detector, beam_center, distance, theta_in_deg, wavelength):
        self.detector = detector
        self.beam_center = beam_center
        self.distance = distance
        self.theta_in = np.deg2rad(theta_in_deg)  # Convert to radians
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength  # Wavevector

        # Update the beam center by the theta_in
        self.beam_center[1] += self.distance * np.tan(self.theta_in)

    def mask_center(self, radius=5, bar='False'):
        beam_center_pixel = [int(self.beam_center[0] / self.detector[1] * self.detector[0]),
                             self.detector[2] - int(self.beam_center[1] / self.detector[3] * self.detector[2])]
        mask = np.ones((self.detector[2], self.detector[0]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - beam_center_pixel[1])**2 + (j - beam_center_pixel[0])**2 < radius**2:
                    mask[i, j] = 0

        if bar == 'True':
            # 定义一个宽度为2的bar形状为长方形,起点位于beam_center_pixel,角度为-180,0之间,长度延伸到探测器外面
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
        beam_center_pixel_re = [int(self.beam_center[0] / self.detector[1] * self.detector[0]),
                             self.detector[2] - int(self.beam_center[1] / self.detector[3] * self.detector[2] + self.distance * 2 * np.tan(incident_angle))]
        mask = np.ones((self.detector[2], self.detector[0]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - beam_center_pixel_re[1])**2 + (j - beam_center_pixel_re[0])**2 < radius**2:
                    mask[i, j] = 0
        if bar == 'True':
            # 定义一个宽度为2的bar形状为长方形,起点位于beam_center_pixel,角度为-180,0之间,长度延伸到探测器外面
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
        position = int(self.beam_center[0] / self.detector[1] * self.detector[0])
        mask = np.ones((self.detector[2], self.detector[0]))
        mask[:, position - width // 2:position + width // 2] = 0
        return mask.astype(bool)

    def mask_gap(self, v_number, h_number, gap_size, model='random'):
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

    

# # Example usage:
# detector_params = [128, 172.0, 128, 172.0]
# beam_center = [86.0, 10.0]
# distance = 3000
# theta_in_deg = 0.2
# wavelength = 0.1

# detector = Detector(detector_params, beam_center, distance, theta_in_deg, wavelength)
# qx, qy, qz, qr = detector.calculate_q_vectors()

# import matplotlib.pyplot as plt
# plt.imshow(qr)
# plt.show()