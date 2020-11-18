import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

W2, W1, W0 = 695.4154350661511, -45.231656699536124, 104.60941780103624
D2, D1 = 9.480956682443292, 27.154192555435458

class ChannelBase():
  def __init__(self):
    self.L = 20000
    self.W = 200
    self.D = 12

    # migration parameters
    self.pad = 0
    self.ds = 1000
    self.nit = 150
    self.Cf = 0.02
    self.crdist = 1.5*self.W
    self.kl = 60.0/(365*24*60*60.0)
    self.kv = 1.0E-11
    self.dt = 2*0.05*365*24*60*60.0
    self.density = 1000
    self.saved_ts = 15
    self.t1 = 50
    self.t2 = 70
    self.t3 = 100
    self.aggr_factor = 4.0

    self.x = np.linspace(0, self.L, int(self.L / self.ds) + 1)
    self.y = np.zeros(len(self.x))
    self.z = np.zeros(len(self.x))

  def compute_wd(self):
    dz = np.gradient(self.z) / self.ds

    self.w = W2 * np.exp(- W1 * dz) + W0
    self.d = D2 * np.exp(- D1 * dz)

  def plot(self):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('Superior View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.plot(self.x, self.y)

    ax2.set_title('Side View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.plot(self.x, self.z)

    fig.suptitle(self.__class__.__name__)

    return fig, (ax1, ax2)

class ChannelSine(ChannelBase):
  def __init__(self):
    super().__init__()

    self.y = 2 * self.W * np.sin(self.x / self.L * 2 * np.pi)

    self.saved_ts = 1
    self.nit = 1
    self.t1 = 2
    self.t2 = 2
    self.t3 = 2

    self.compute_wd()

class ChannelSineNoSlope(ChannelSine):
  def __init__(self):
    super().__init__()

    self.compute_wd()

class ChannelSineConstantSlope(ChannelSine):
  def __init__(self):
    super().__init__()

    self.z = np.tan(5.0 * np.pi / 180) * ( self.L - self.x )
    self.compute_wd()

class ChannelSineRampSlope(ChannelSine):
  def __init__(self):
    super().__init__()

    self.z = np.tan(5.0 * np.pi / 180) / (2 * self.L) * (self.x ** 2 + self.L * ( self.L - 2 * self.x ) )
    self.compute_wd()
class ChannelScatteredSine(ChannelBase):
  def __init__(self):
    super().__init__()
    self.ds = 100.0

    self.x = np.linspace(0, self.L, int(self.L / self.ds) + 1)
    self.y = 2 * self.W * np.exp(( 1.0 / self.L) * self.x) * np.sin((self.x / self.L) * 16 * np.pi)
    self.z = np.zeros(len(self.x))
    self.compute_wd()
    
    self.saved_ts = 1
    self.nit = 1
    self.t1 = 2
    self.t2 = 2
    self.t3 = 2

class ChannelScatteredSineNoSlope(ChannelScatteredSine):
  def __init__(self):
    super().__init__()

    self.z = np.zeros(len(self.x))
    self.compute_wd()

class ChannelScatteredSineConstantSlope(ChannelScatteredSine):
  def __init__(self):
    super().__init__()

    self.z = np.tan(5.0 * np.pi / 180) * ( self.L - self.x )
    self.compute_wd()

class ChannelScatteredSineRampSlope(ChannelScatteredSine):
  def __init__(self):
    super().__init__()

    self.z = np.tan(5.0 * np.pi / 180) / (2 * self.L) * (self.x ** 2 + self.L * ( self.L - 2 * self.x ) )
    self.compute_wd()

if __name__ == '__main__':
  #cases = [ChannelSineNoSlope(), ChannelSineConstantSlope(), ChannelSineRampSlope()]
  cases = [ChannelScatteredSineNoSlope(), ChannelScatteredSineConstantSlope(), ChannelScatteredSineRampSlope()]

  for case in cases:
    case.plot()
    plt.show()