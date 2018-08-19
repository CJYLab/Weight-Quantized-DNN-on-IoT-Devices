from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

figure = plt.figure(figsize = (4,4))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0, hspace=0)

fvpp = np.array([0.0, 0.0, 0.0, 0.0, 2.890196204185486, 1.0627451613545418, 1.0039216298609972, 0.9921569228172302, 6.517647445201874, 1.6431373357772827, 1.4862746000289917, 1.6784314513206482, 2.4666667580604553, 1.9490196704864502, 1.7647059559822083, 1.9176471829414368, 0.0, 0.0, 0.0, 0.0, 6.086274892091751, 5.843137577176094, 4.9333336353302, 2.9803923373110592, 4.968627721071243, 2.976470802910626, 2.564706027507782, 1.9529412984848022, 0.37254904210567474, 0.2196078598499298, 0.2196078598499298, 0.2196078598499298, 0.0, 0.0, 0.0, 0.0, 2.4000001698732376, 1.8901961743831635, 1.1450981050729752, 1.447058916091919, 5.568627774715424, 6.196078836917877, 4.776470899581909, 4.776470854878426, 0.0, 0.0, 0.0, 0.0, 2.5176472067832947, 1.9490197375416756, 1.6666667461395264, 1.878431499004364, 6.262745499610901, 3.337255120277405, 2.368627607822418, 3.81176495552063, 2.094117791391909, 1.3058824241161346, 1.027451042085886, 1.0901961624622345, 0.0, 0.0, 0.0, 0.0])
fvpp.shape
print(fvpp[8])
import matplotlib as mlp
from matplotlib import pyplot as plt
#the sequence of the FVPP: horizon, vertical, lefdig, rightdig
for i in range(16):
    ax1 = plt.subplot(gs1[i])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xticks([])
    plt.yticks([])   
    ax1.set_aspect('equal')
    ax1.plot([-fvpp[i*4]/2.0, fvpp[i*4]/2.0], [0, 0], 'k-', lw=2)
    ax1.plot([0, 0], [-fvpp[i*4+1]/2.0, fvpp[i*4+1]/2.0], 'k-', lw=2)
    ax1.plot([-(fvpp[i*4+2]/np.sqrt(2))/2.0, (fvpp[i*4+2]/np.sqrt(2))/2.0], [(fvpp[i*4+2]/np.sqrt(2))/2.0, -(fvpp[i*4+2]/np.sqrt(2))/2.0], 'k-', lw=2)
    ax1.plot([-(fvpp[i*4+3]/np.sqrt(2))/2.0, fvpp[i*4+3]/np.sqrt(2)], [-(fvpp[i*4+3]/np.sqrt(2))/2.0, (fvpp[i*4+3]/np.sqrt(2))/2.0], 'k-', lw=2)

figure.savefig('5.svg', format='svg', dpi=1200)
figure.savefig('5_jpg.jpg',dpi=1200)
figure.savefig('5_eps.eps', format='eps', dpi=1200)
