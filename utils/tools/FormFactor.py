from tqdm import tqdm
import bornagain as ba
from bornagain import deg, nm, R3
import numpy as np

class ellipsoid:
    def __init__(self):
        pass
    
    @classmethod
    def get_sample(cls,R,h):
        # Define materials
        material_Particle = ba.RefractiveMaterial("Particle", 0.0006, 2e-08)
        material_Substrate = ba.RefractiveMaterial("Substrate", 6e-06, 2e-08)
        material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

        # Define form factors
        ff = ba.Spheroid(R*nm, h*nm)

        # Define particles
        particle = ba.Particle(material_Particle, ff)
        particle_rotation = ba.RotationEuler(0*deg, 0*deg, 0*deg)
        particle.rotate(particle_rotation)
        particle_position = R3(0*nm, 0*nm, 0*nm)
        particle.translate(particle_position)

        # Define particle layouts
        layout = ba.ParticleLayout()
        layout.addParticle(particle, 1.0)
        layout.setTotalParticleSurfaceDensity(0.01)
        
        # Define roughness
        roughness = ba.LayerRoughness(3, 0.3, 5*nm)

        # Define layers
        layer_1 = ba.Layer(material_Vacuum)
        layer_1.addLayout(layout)
        layer_2 = ba.Layer(material_Substrate)

        # Define sample
        sample = ba.MultiLayer()
        sample.addLayer(layer_1)
        sample.addLayerWithTopRoughness(layer_2, roughness)

        return sample

    @classmethod
    def get_simulation(cls, sample):

        # Define GISAS simulation:
        beam = ba.Beam(100000000.0, 0.1*nm, 0.2*deg)

        detector = ba.RectangularDetector(128, 172.0, 128, 172.0)
        detector.setPerpendicularToDirectBeam(3000.0, 86.0, 10.0)
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        return simulation

    @classmethod
    def generate_gaussian_matrix(cls, size=20):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        x, y = np.meshgrid(x, y)
        
        # Flatten the coordinates
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        
        # Randomly generated parameters
        mu1, mu2 = np.random.uniform(0, 1, 2)
        sigma1, sigma2 = np.random.uniform(0.01, 0.5, 2)
        covariance = np.random.uniform(0, 1)
        
        # covariance matrix
        cov_matrix = np.array([[sigma1**2, covariance * sigma1 * sigma2],
                            [covariance * sigma1 * sigma2, sigma2**2]])
        
        mu = np.array([mu1, mu2])
        
        # Generate a 2D Gaussian distribution
        rv = np.random.multivariate_normal(mu, cov_matrix, (size, size))
        g = np.exp(-0.5 * np.einsum('...k,kl,...l->...', pos - mu, np.linalg.inv(cov_matrix), pos - mu))
        
        return g / g.sum()

    @classmethod
    def get_distribution(cls, R_min = 0.05, R_max = 20, h_min = 0.05, h_max = 20, size = 20):
        # Creating Interval Mappings
        h_bins = np.linspace(h_min, h_max, size+1)
        R_bins = np.linspace(R_min, R_max, size+1)

        # random_weights_hR = np.random.dirichlet(alpha=[1] * n], size=1)[0]
        # gaussian_matrices_hR = sum(self.generate_gaussian_matrix(size) * random_weights_hR[i] for i in range(n))
        # gaussian_matrices_hR /= gaussian_matrices_hR.sum()

        # 预先计算并存储所有 hr_data 结果
        hr_data_dict = {}
        for i in tqdm(range(size)):
            for j in range(size):
                h = (h_bins[i] + h_bins[i+1]) / 2
                R = (R_bins[j] + R_bins[j+1]) / 2
                
                sample = cls.get_sample(h, R)
                simulation = cls.get_simulation(sample)
                result = simulation.simulate()
                hr_data = result.array()
                
                # 存储 hr_data 结果
                hr_data_dict[(i, j)] = hr_data
        
        return hr_data_dict



        
if __name__ == '__main__':
    sample = ellipsoid.get_sample(5,10)
    simulation = ellipsoid.get_simulation(sample)
    result = simulation.simulate()
    print(result)
    data = result.array()

    hr_data_dict = ellipsoid.get_distribution()