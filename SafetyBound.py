import numpy as np
import random
######################################### safety bound ############################################################
class Bound:
    def __init__(self, cfg, vwind0, alphaw0):
        self.vuav = cfg['vuav']
        self.a = cfg['a']
        self.tt = cfg['tt']
        self.ori = cfg['ori']
        self.des = cfg['des']
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0

    # sensor errors
    def winderror(self): # epistemic random variables
        self.evwind = np.random.normal(0, 0.05 * self.vwind0) #epsilon Vwind
        self.ealphaw = np.random.normal(0, 0.05 * self.alphaw0) # alpha wind
        return [self.evwind, self.ealphaw]

    def uaverror(self):
        self.epGPS = np.random.uniform(-1.5, 1.5) #epsilon GPS
        self.evuav = np.random.normal(0, 0.05 * self.vuav) # Vuav
        self.ea = np.random.normal(0, 0.05 * self.a) #a
        self.ett = np.random.normal(0, 0.05 * self.tt) # tupdate

    ######################### safety bound size #######################################################################
    def size(self):
        self.uaverror()
        self.winderror()
        ### angle between uav and wind
        self.vecwind = ((self.vwind0 + self.evwind) * np.cos(self.alphaw0 + self.ealphaw), (self.vwind0 +  self.evwind) * np.sin(self.alphaw0 + self.ealphaw)) #eq 9 'where' half , eq 10 'where' full
        self.uavdir = np.subtract(self.des, self.ori)
        self.uavvvec = self.vuav / np.linalg.norm(self.uavdir) * self.uavdir
        self.alphaw = np.arccos(np.dot(self.uavdir, self.vecwind)/(np.linalg.norm(self.uavdir)*np.linalg.norm(self.vecwind)))

        ### size
        self.valong = self.vuav + self.evuav + (self.vwind0 + self.evwind) * (np.cos(self.alphaw)) #along heading direction
        self.vper = (self.vwind0 + self.evwind) * (np.sin(self.alphaw0+ self.ealphaw)) #perpendicular to heading direction

        self.lh = abs(self.valong) * (self.tt + self.ett) + (self.valong) ** 2 / 2 / (self.a + self.ea)
        self.lp = abs(self.vper) * (self.tt + self.ett) + (self.vper) ** 2 / 2 / (self.a + self.ea) + self.epGPS
        self.r = self.lh + self.lp

cfg = {'vuav': 16, #configuration
        'a': 5,
        'tt': 1,
        'ori': (45, 0),  #origin
        'des': (0, 40)} #destination--- up: (0, 10) down (0, -10) left (-10, 0) right (10, 0)

np.random.seed(0)
random.seed(0)
vwind0 = 3
alphaw0 = (np.pi) / 4.0
N = 1000 #number of Monte Carlo iterations
lh = np.zeros(N)
lp = np.zeros(N)
b = Bound(cfg, vwind0, alphaw0)
# print(b)

for i in range(N):
    b.size()
    lh[i] = b.lh
    lp[i] = b.lp
print(np.sort(lh)[950], np.sort(lp)[950])
# print(np.sort(lh), np.sort(lp))
# print(len(lh), len(lp))
# print(vwind0, alphaw0)

"""
The file SafetyBound.py contains a class called Bound which has methods for calculating the size of a safety bound for a UAV given certain configuration parameters such as the UAV's velocity, acceleration, and update time.

The Bound class has the following methods:

__init__(self, cfg, vwind0, alphaw0): This is the constructor method for the Bound class. It initializes the object with the given configuration parameters such as the UAV's velocity (vuav), acceleration (a), update time (tt), origin (ori), and destination (des). It also takes in the wind velocity (vwind0) and the wind angle (alphaw0) as input.

winderror(self): This method adds error to the wind velocity and angle by sampling from normal distributions with mean 0 and standard deviation 5% of the wind velocity and angle, respectively. It returns the wind error terms as a list.

uaverror(self): This method adds error to the UAV velocity, acceleration, and update time by sampling from normal distributions with mean 0 and standard deviation 5% of the UAV velocity, acceleration, and update time, respectively. It also adds error to the GPS measurement by sampling from a uniform distribution between -1.5 and 1.5.

size(self): This method calculates the size of the safety bound by first finding the angle between the UAV heading and the wind direction. It then calculates the velocity of the UAV along the heading direction and the velocity of the UAV perpendicular to the heading direction. Using these velocities, it calculates the length of the safety bound along the heading direction (lh) and the length of the safety bound perpendicular to the heading direction (lp). The total size of the safety bound is the sum of these lengths (r).

The output of the Bound class is a tuple containing the values of lh, lp, and r. These values represent the size of the safety bound that should be maintained around the UAV as it travels from the origin to the destination. The UAV should not enter any areas that fall within this bound.
--------------------------------------------------------------------
what exactly is the size of the safety bound and why it is immportant?
In the Bound class, the size method calculates the size of the safety bound around the UAV's trajectory. The safety bound is a region around the UAV's trajectory in which the UAV is guaranteed to stay within, given certain assumptions about wind and UAV errors.

The size of the safety bound is important because it determines the level of safety that is guaranteed for the UAV's trajectory. A larger safety bound would allow for a greater level of uncertainty in wind and UAV errors, but it would also increase the amount of space the UAV needs to navigate through. On the other hand, a smaller safety bound would provide a higher level of safety, but it would also be more sensitive to wind and UAV errors.

In the size method, the size of the safety bound is calculated as the sum of two components: the length of the bound along the heading direction of the UAV (lh) and the length of the bound perpendicular to the heading direction (lp). The lh component takes into account the UAV's speed and acceleration, as well as the wind speed and direction. The lp component takes into account the wind speed and direction, as well as the error in the UAV's GPS position.

The size method also calculates the angle between the UAV's heading direction and the wind direction, which is used to determine the components of the wind speed that affect the size of the safety bound.
-------------------------------------------------------------------------
what is N = 1000 ?
In the provided code snippet, N is the number of Monte Carlo iterations. It is used to specify the number of times the size method of the Bound class should be called. The size method computes the size of the safety bound based on the specified configurations and wind and GPS errors.

The safety bound is a region around the UAV's intended path that represents a safe zone for the UAV to operate in. The size of the safety bound is important because it determines the space that the UAV needs to navigate through while avoiding obstacles and staying within the boundaries of the environment. The safety bound size is calculated by considering the UAV's speed, acceleration, and update time, as well as the wind speed and direction and GPS error. By running the size method multiple times and storing the results in the lh and lp arrays, the code is able to obtain statistical information about the size of the safety bound, such as its mean and standard deviation. This information can be used to help the UAV plan its path and make decisions about how to navigate through the environment safely.
-------------------------------------------------------------------------
how are the uav errors being calculated? on what basis and evidences?
In the Bound class, the uaverror function generates random errors for the UAV's GPS, velocity, and acceleration. These errors are modeled as uniform, normal, and normal distributions, respectively.

The uaverror function is called before the size function, which calculates the size of the safety bound. The size function uses the errors generated by the uaverror function to calculate the size of the bound, which is determined by the wind and UAV errors, as well as the UAV's velocity and acceleration.

The value of N in the code snippet you provided is 1000, which is the number of Monte Carlo iterations that are used to calculate the safety bound size. Monte Carlo methods involve generating a large number of random samples and using the statistical properties of the samples to estimate certain quantities. In this case, N is the number of samples used to estimate the size of the safety bound.
"""