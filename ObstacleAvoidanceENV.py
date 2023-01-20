import numpy as np
from gym.envs.toy_text import discrete
from geometryCheck import geometry
from SafetyBound import Bound
from my_env import MyEnv

UP = 3
RIGHT = 0
DOWN = 1
LEFT = 2
UP_RIGHT = 4
UP_LEFT = 5
DOWN_RIGHT = 6
DOWN_LEFT = 7
g=geometry()

polysize1 = [0,10,0,15]#[0,20,0,25] #[0, 20, 0, 30] 
polysize2 =[10,17,17,25] #[20,35,35,50]#[20, 35, 35, 50] # 
polysize3 = [2,7,25,32]#[36,50,4,25]#[4, 15, 50, 65] 
polysize4 = [19,30,5,15]#[40,55,45,60]#[38, 60, 10, 30]#
polysize5 = [12,17,32,47]#[4,15,55,70]#[25, 35, 65, 94] # 

def polyReal(polysize):
    return [(polysize[0], polysize[2]), (polysize[0], (polysize[3]+1)), ((polysize[1]+1), (polysize[3]+1)), ((polysize[1]+1), polysize[2]),((polysize[1]+1), polysize[2]+1)]

poly1 = polyReal(polysize1)
poly2 = polyReal(polysize2)
poly3 = polyReal(polysize3)
poly4 = polyReal(polysize4)
poly5 = polyReal(polysize5)

def polyFlag(g, poly1, poly2, poly3, poly4,poly5, boundp, c1, c2, r):  #c1- centre of circle, r= radii
    flag =0

    if (
            g.Flagrectc(poly1, c1, r) == 1 or g.Flagrectc(poly1, c2, r) == 1 or g.Flag2rect(poly1, boundp) == 1 or
            g.Flagrectc(poly2, c1, r) == 1 or g.Flagrectc(poly2, c2, r) == 1 or g.Flag2rect(poly2, boundp) == 1 or
            g.Flagrectc(poly3, c1, r) == 1 or g.Flagrectc(poly3, c2, r) == 1 or g.Flag2rect(poly3, boundp) == 1 or
            g.Flagrectc(poly4, c1, r) == 1 or g.Flagrectc(poly4, c2, r) == 1 or g.Flag2rect(poly4, boundp) == 1
            or
            g.Flagrectc(poly5, c1, r) == 1 or g.Flagrectc(poly5, c2, r) == 1 or g.Flag2rect(poly5, boundp) == 1
    ):
        flag = 1

    return flag

class UAVEnv(MyEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, origin, des, safetybound = False, safety_bound_size = None, vwind0=3, alphaw0=(np.pi) / 4.0, wind_direction = "N", wind_strength = 0):
        self.b = 0
        if safetybound == True:
            self.b = 1
        self.shape = (50, 50)
        self.des= des 
        nS = np.prod(self.shape)
        # nA = 4
        nA = 8
        self.origin = origin
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0
        self.safetybound = safetybound
        self.wind_direction = wind_direction
        self.wind_strength = wind_strength
        if self.safetybound:
            cfg = {'vuav': 16, #configuration
                         'a': 5,
                         'tt': 1,
                         'ori': (45,0),#(68, 0) ,   #origin
                         'des': (0,40),#(0, 65) ,
                         'safety_bound_size': safety_bound_size} #destination--- up: (0, 10) down (0, -10) left (-10, 0) right (10, 0)
            self.bound = Bound(cfg, vwind0, alphaw0 ) 
        # obstacle Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[polysize1[0]:polysize1[1]+1, polysize1[2]:polysize1[3]+1] = True
        self._cliff[polysize2[0]:polysize2[1]+1, polysize2[2]:polysize2[3]+1] = True
        self._cliff[polysize3[0]:polysize3[1]+1, polysize3[2]:polysize3[3]+1] = True
        self._cliff[polysize4[0]:polysize4[1]+1, polysize4[2]:polysize4[3]+1] = True
        self._cliff[polysize5[0]:polysize5[1]+1, polysize5[2]:polysize5[3]+1] = True

        # define transition model
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            # P[s][UP] = self._calculate_transition_prob(position, [5, 0])
            P[s][UP] = self._calculate_transition_prob(position, [0, -5] , wind_direction= "N", wind_strength = 6 )
            # P[s][RIGHT] = self._calculate_transition_prob(position, [0, 5])
            P[s][RIGHT] = self._calculate_transition_prob(position, [5, 0] , wind_direction = "E", wind_strength = 3)
            # P[s][DOWN] = self._calculate_transition_prob(position, [-4, 0])
            P[s][DOWN] = self._calculate_transition_prob(position, [0, -4] , wind_direction= "S", wind_strength = 1)
            # P[s][LEFT] = self._calculate_transition_prob(position, [0, -4])
            P[s][LEFT] = self._calculate_transition_prob(position, [-4, 0] , wind_direction = "W", wind_strength= 2)
            # P[s][UP_RIGHT] = self._calculate_transition_prob(position, [2,-1] )
            P[s][UP_RIGHT] = self._calculate_transition_prob(position, [2,-2] , wind_direction = "NE", wind_strength = 1)
            # P[s][UP_LEFT] = self._calculate_transition_prob(position, [-3, -3])
            P[s][UP_LEFT] = self._calculate_transition_prob(position, [-3, -3] , wind_direction = "NW", wind_strength= 0)
            # P[s][DOWN_RIGHT] = self._calculate_transition_prob(position, [-3, 3])
            P[s][DOWN_RIGHT] = self._calculate_transition_prob(position, [-3, 3] , wind_direction = "SE", wind_strength = 0)
            # P[s][DOWN_LEFT] = self._calculate_transition_prob(position, [5, -5])
            P[s][DOWN_LEFT] = self._calculate_transition_prob(position, [3, -3] , wind_direction = "SW", wind_strength = 0)

        isd = np.zeros(nS) #initial state distribution
        isd[np.ravel_multi_index(origin, self.shape)] = 1.0
        """
        isd (initial state distribution) is a numpy array of size nS (number of states) with all elements initialized to zero.
        It then uses the np.ravel_multi_index function to convert the two-dimensional origin position of the UAV into a one-dimensional index and set the corresponding element of the isd array to 1.0
        origin is the starting point of UAV, np.ravel_multi_index converts the 2D index to 1D index, this index is used as a starting point in the isd array and the value is set to 1.0 to indicate that UAV starts from this point. So, this line of code sets the probability of starting from the specified origin position to 1.0, and all other states will have a starting probability of 0.0
        """
        super(UAVEnv, self).__init__(nS, nA, P, isd, wind_direction, wind_strength) # super-line 44 which has init. Accesses methods of base class

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    """
    The function _limit_coordinates takes a single argument coord which is a list or array containing the x, y coordinates of a point. It then limits the x, y coordinates so that the point remains within the boundaries of the grid defined by the self.shape attribute.

    It first takes the minimum of the x-coordinate and the maximum x-coordinate of the grid and assigns it to the x-coordinate of the point, this makes sure that the x-coordinate of the point doesn't exceed the grid's x-coordinate.
    Then, it takes the minimum of the y-coordinate and the maximum y-coordinate of the grid and assigns it to the y-coordinate of the point, this makes sure that the y-coordinate of the point doesn't exceed the grid's y-coordinate.
    This way this function makes sure that the point is not exceeding the grid's boundary and returns the modified coordinates.
    """

    # def _calculate_transition_prob(self, current, delta):
    def _calculate_transition_prob(self, current, delta, wind_direction, wind_strength):
    
        """

        _calculate_transition_prob is a method in the UAVEnv class that is used to calculate the transition probabilities for taking an action in the environment. The transition probability is a measure of how likely it is that an action will result in a particular next state. In this case, it appears that _calculate_transition_prob is being used to define the possible outcomes of taking an action in the environment, including the reward received and whether the episode is terminated.
        It's worth noting that the _calculate_transition_prob method is not directly related to a transition model, which is a probabilistic model that describes the likelihood of transitioning between states in a system over time. In this code snippet, the transition probabilities are being calculated based on the rules of the environment and the actions taken by the agent, rather than being learned from data.

        """
        if wind_direction == 'N':
            wind_delta = [wind_strength, 0]
        elif wind_direction == 'S':
            wind_delta = [-wind_strength, 0]
        elif wind_direction == 'E':
            wind_delta = [0, wind_strength]
        elif wind_direction == 'W':
            wind_delta = [0, -wind_strength]
        elif wind_direction == 'NE':
            wind_delta = [wind_strength, wind_strength]
        elif wind_direction == 'NW':
            wind_delta = [wind_strength, -wind_strength]
        elif wind_direction == 'SE':
            wind_delta = [-wind_strength, wind_strength]
        elif wind_direction == 'SW':
            wind_delta = [-wind_strength, -wind_strength]
        else:
            wind_delta = [0, 0]
        # new_position = np.array(current) + np.array(delta)
        new_position = np.array(current) + np.array(delta) + np.array(wind_delta)
        # cur_pos = np.unravel_index(current, self.shape)
        # des_pos = np.unravel_index(self.des, self.shape)
        # distance = np.linalg.norm(np.subtract(cur_pos, des_pos))
        # new_position = np.unravel_index(next_s, self.shape)
        new_position = self._limit_coordinates(new_position).astype(int) #astype- convert to int
        
        new_s = np.ravel_multi_index(new_position, self.shape)
        reward = self._calculate_reward(current, delta, new_position, wind_direction, wind_strength)
        is_done = False
        return (1.0, new_s, reward, is_done)
        # next_distance = np.linalg.norm(np.substract(new_position, des_pos))
        # new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        # transition probability is 1 if the new state is within the grid boundaries, otherwise 0
        # if (new_position >= 0).all() and (new_position < self.shape).all():
        #     prob = 1.0
        # else:
        #     prob = 0.0
        # return [(new_state, prob, 0.0, {})]

    # def _calculate_reward(self, s, a, next_s, wind_direction,wind_strength):
    def _calculate_reward(self, current, delta, next_pos, wind_direction,wind_strength):
        # wind_direction, wind_strength = self._calculate_transition_prob(self.action_to_delta[a])
        # # Calculate the distance from the current position to the destination
        cur_pos = np.unravel_index(current, self.shape)
        # next_pos = np.unravel_index(next_s, self.shape)
        # cur_pos = np.array(current)
        des_pos = np.unravel_index(self.des, self.shape)
        distance = np.linalg.norm(np.subtract(next_pos, des_pos))

        penalty = 0
        if wind_direction == 'N':
            if delta != [-wind_strength, 0]:
                penalty = -wind_strength
        elif wind_direction == 'S':
            if delta != [wind_strength, 0]:
                penalty = -wind_strength
        elif wind_direction == 'E':
            if delta != [0, wind_strength]:
                penalty = -wind_strength
        elif wind_direction == 'W':
            if delta != [0, -wind_strength]:
                penalty = -wind_strength
        elif wind_direction == 'NE':
            if delta != [wind_strength, wind_strength]:
                penalty = -wind_strength
        elif wind_direction == 'NW':
            if delta != [wind_strength, -wind_strength]:
                penalty = -wind_strength
        elif wind_direction == 'SE':
            if delta != [-wind_strength, wind_strength]:
                penalty = -wind_strength
        elif wind_direction == 'SW':
            if delta != [-wind_strength, -wind_strength]:
                penalty = -wind_strength
        else:
            penalty = 0

        if np.array_equal(next_pos, des_pos):
            reward = 10 - distance
        else:
            reward = -0.5* distance* (1-penalty)
        # return reward

        # Calculate the distance from the next position to the destination
        # next_pos = np.unravel_index(next_s, self.shape)
        # next_distance = np.linalg.norm(np.subtract(next_pos, des_pos))

        # Check the wind direction and penalize the agent for moving against it
        # penalty = 0
        # if wind_direction == 'N':
        #     if a != UP:
        #         penalty = -wind_strength
        # elif wind_direction == 'S':
        #     if a != DOWN:
        #         penalty = -wind_strength
        # elif wind_direction == 'E':
        #     if a != RIGHT:
        #         penalty =-wind_strength
        # elif wind_direction == 'W':
        #     if a != LEFT:
        #         penalty = -wind_strength
        
        # elif wind_direction == 'NE':
        #     if a != UP_RIGHT:
        #         penalty = -wind_strength
        
        # elif wind_direction == 'NW':
        #     if a != UP_LEFT:
        #         penalty =-wind_strength
        
        # elif wind_direction == 'SE':
        #     if a != DOWN_RIGHT:
        #         penalty = -wind_strength

        # elif wind_direction == 'SW':
        #     if a != DOWN_LEFT:
        #         penalty = -wind_strength        
        # else:
        #     penalty = 0
    #     # if next_s == self.des:
    #     #     reward = 100 - distance
    #     # else:
    #     #     reward = -0.5*distance*(1-penalty)
    #     # return reward
        
        # delta = np.subtract(next_pos,cur_pos)
        # reward = -0.5*distance*(1-penalty)


        # Check if the agent is moving diagonally
        if (delta[0] != 0) and (delta[1] != 0):
            # Check if the agent is moving in the wind direction
            if np.any([wind_direction == 'N' and delta[0] == -wind_strength, wind_direction == 'S' and delta[0] == wind_strength, wind_direction == 'E' and delta[1] == wind_strength, wind_direction == 'W' and delta[1] == -wind_strength]):
                # If the agent is moving in the wind direction, adjust the reward based on the wind strength
                reward += wind_strength
            else:
                # If the agent is moving against the wind direction, adjust the reward based on the wind strength
                reward -= wind_strength

            # if next_s == self.des:
            #     reward = 10 - distance
            # else:
            #     reward = -0.5* distance* (1-penalty)    
# Check if the agent is going to move into a safety bound and if so, add a penalty to the reward

        # if self.safetybound:
        #     if self.b==1:
        #         #up
        #         if delta == [0, 5]:
        #             lh = 68.6
        #             r = 1.7
        #             c1 = (cur_pos[0]*4+2+wind_strength, cur_pos[1]*4 +2)
        #             c2 = (c1[0]-lh, c1[1])
        #             boundp = [(c1[0], c1[1]-r), (c2[0], c2[1]-r), (c2[0], c2[1]+r),(c1[0], c1[1]+r)]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

                #down
        #         if delta == [3, 0]:
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2+ wind_strength, cur_pos[1] * 4 + 2)
        #             c2 = (c1[0] + lh, c1[1])
        #             boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100 

                
        #         if delta == [0,5]:
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2, cur_pos[1] * 4 + 2 + wind_strength)
        #             c2 = (c1[0], c1[1] + lh)
        #             boundp = [(c1[0] - r, c1[1]), (c2[0] - r, c2[1]), (c2[0] + r, c2[1]), (c1[0] + r, c1[1])]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

                
        #         if delta == [0,-5]:
        #             lh = 68.6
        #             r = 1.7
        #             c1 = (cur_pos[0]*4 + 2, cur_pos[1]*4 + 2 - wind_strength)
        #             c2 = (c1[0], c1[1]-lh)
        #             boundp = [(c1[0]-r, c1[1]), (c2[0]-r, c2[1]), (c2[0]+r, c2[1]), (c1[0]+r, c1[1])]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

                
        #         if delta == [4, 4]:
        #             # Check if the agent is moving in the wind direction
        #             if wind_direction == 'NE':
        #                 reward += wind_strength
        #             else:
        #                 reward -= wind_strength
                    
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2+ wind_strength, cur_pos[1] * 4 + 2+ wind_strength)
        #             c2 = (c1[0] + lh, c1[1])
        #             boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]

        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

                
        #         if delta == [-5, -5]:
        #             # Check if the agent is moving in the wind direction
        #             if wind_direction == 'NW':
        #                 reward += wind_strength
        #             else:
        #                 reward -= wind_strength
                    
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2+ wind_strength, cur_pos[1] * 4 + 2- wind_strength)
        #             c2 = (c1[0] + lh, c1[1])
        #             boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]

        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

        #         #down_right
        #         if delta == [3, 5]:
        #             # Check if the agent is moving in the wind direction
        #             if wind_direction == 'SE':
        #                 reward += wind_strength
        #             else:
        #                 reward -= wind_strength
                    
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2- wind_strength, cur_pos[1] * 4 + 2+ wind_strength)
        #             c2 = (c1[0] + lh, c1[1])
        #             boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100

        #         #DOWN_LEFT
        #         if delta == [3, -5]:
        #         # Check if the agent is moving in the wind direction
        #             if wind_direction == 'SW':
        #                 reward += wind_strength
        #             else:
        #                 reward -= wind_strength
        #             lh = 31.9
        #             r = 1.7
        #             c1 = (cur_pos[0] * 4 + 2- wind_strength, cur_pos[1] * 4 + 2- wind_strength)
        #             c2 = (c1[0] + lh, c1[1])
        #             boundp = [(c1[0], c1[1]-r), (c2[0], c2[1]-r), (c2[0], c2[1]+r),(c1[0], c1[1]+r)]
        #             ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
        #             if ff == 1:
        #                 reward = reward - 100    



            # if next_s == self.des:
            #     reward = 10 - next_distance
            # return reward            
        # def _calculate_reward(self, s, a, next_s, vwind0, alphaw0):
        #     # Calculate the distance from the current position to the destination
        #     cur_pos = np.unravel_index(s, self.shape)
        #     des_pos = np.unravel_index(self.des, self.shape)
        #     distance = np.linalg.norm(np.subtract(cur_pos, des_pos))

        #     # Calculate the distance from the next position to the destination
        #     next_pos = np.unravel_index(next_s, self.shape)
        #     next_distance = np.linalg.norm(np.substract(next_pos, des_pos))
                        
        #     if next_distance < distance:
        #         return 10
            
        #     elif next_distance > distance:
        #         return -10
        #     # If the distance to the destination has not changed, return a small positive reward
        #     else:
        #         return 1
        
        # #updates state of the env 
        # def step(self,s, a):
        #     assert self.action_space.contains(a)
        #     position = np.unravel_index(self.s, self.shape)
        #     print(position)
        #     # wind and action taken
        #     vwind = self.vwind0 * np.cos(self.alphaw0)
        #     if a == RIGHT:
        #         self.s = np.ravel_multi_index((position[0] + 1, position[1] + int(vwind)), self.shape)
        #     elif a == LEFT:
        #         self.s = np.ravel_multi_index((position[0] + 1, position[1] - int(vwind)), self.shape)
        #     elif a == UP:
        #         self.s = np.ravel_multi_index((position[0] + 2, position[1]), self.shape)
        #     elif a == DOWN:
        #         self.s = np.ravel_multi_index((position[0], position[1]), self.shape)
        #     # if a == UP_RIGHT:
        #     #     self.s = np.ravel_multi_index((position[0] +1, position[1] + 1), self.shape)
        #     # elif a == UP_LEFT:
        #     #     self.s = np.ravel_multi_index((position[0] - 1, position[1] - int(vwind)), self.shape)
        #     # elif a == DOWN_RIGHT:
        #     #     self.s = np.ravel_multi_index((position[0] + 1, position[1] + int(vwind)), self.shape)
        #     # elif a == DOWN_LEFT:
        #     #     self.s = np.ravel_multi_index((position[0] + 1, position[1] - int(vwind)), self.shape)

        #     # current position
        #     x = position[0]
        #     y = position[1]
        #     boundp = [x + self.b, x + self.b, y + self.b, y + self.b]
        #     c1 = [x, y]
        #     c2 = [x + 1, y + 1]
        #     r = self.b

        #     # check if we collided with an obstacle or out of bounds

        #     if (x + self.b) > (self.shape[0] - 1) or (y + self.b) > (self.shape[1] - 1) or self._cliff[
        #         x + self.b, y + self.b] == 1 or polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r) == 1:
        #         self.s = np.ravel_multi_index(position, self.shape)
        #         return np.array(position), -100, True, {} # True = obstacle, false = empty space
        #     if (x - self.b) < 0 or (y - self.b) < 0 or self._cliff[x - self.b, y - self.b] == 1 or polyFlag(g, poly1, poly2,poly3, poly4,poly5, boundp, c1, c2, r) == 1:
        #         self.s = np.ravel_multi_index(position, self.shape)
        #         return np.array(position), -100, True, {}

        #     if self.safetybound:
        #         self.bound.size()
        #         self.lh = self.bound.lh
        #         self.lp = self.bound.lp

        #     # check if we reached the destination
        #     if self.s == self.nS - 1:
        #         return self.s, 100, True, {}

        #     reward = self._calculate_reward(s, a, self.s)
        #     return np.array(np.unravel_index(self.s, self.shape)), reward, False, {}

        # distance = np.sqrt((new_position[0] - self.des[0]) ** 2 + (new_position[1] - self.des[1]) ** 2) #distance between two points using Euclidean distance formula
        # distance = np.sqrt((next_pos[0] - self.des[0]) ** 2 + (next_pos[1] - self.des[1]) ** 2) #distance between two points using Euclidean distance formula
        # reward = -0.5 *distance
        # reward = -0.5*distance*(1-penalty) 
        
       
        # # If the distance to the destination has decreased, return a positive reward
        # if next_distance < distance:
        #     reward = 10
        # # If the distance to the destination has increased, return a negative reward
        # elif next_distance > distance:
        #     reward = -10
        # print("#", reward)
        # # vwind = self.vwind0 * np.cos(self.alphaw0)
        # # up
        if delta == [0, 5]: #delta variable represents the change in position that the agent will experience as a result of taking the action.The code checks which direction the agent is moving based on the value of delta, and applies a reward or penalty based on various conditions.
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 68.6 #66.3
                r = 1.7 # 1.6  r = Lh + Lp
            c1 = (current[0]*2+1, current[1]*2+1)
            # c1 = (s[0]+2, s[1]+2)
            c2 = (c1[0]-lh, c1[1])
            boundp = [(c1[0], c1[1]-r), (c2[0], c2[1]-r), (c2[0], c2[1]+r),(c1[0], c1[1]+r)]

            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("##", reward)
        # down
        if delta == [0, -5]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 31.9
                r = 1.7
            c1 = (current[0] * 2 + 1, current[1] * 2 + 1- wind_strength)
            # c1 = (s[0] * 4 + 2, s[1] * 4 + 2)
            c2 = (c1[0] + lh, c1[1])
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("###", reward)
        # right
        if delta == [4, 0]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 49.3
                r = 7.1
            c1 = (current[0] * 2 + 1 + wind_strength, current[1] * 2 + 1)
            # c1 = (s[0] * 4 + 2, s[1] * 4 + 2)
            c2 = (c1[0], c1[1] + lh)
            boundp = [(c1[0] - r, c1[1]), (c2[0] - r, c2[1]), (c2[0]+r, c2[1]), (c1[0]+r, c1[1])]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("####", reward)
        # left
        if delta == [-4, 0]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 49.0
                r = 7.1
            c1 = (current[0] * 2 + 1 - wind_strength, current[1] * 2 + 1)
            # c1 = (s[0] * 4 + 2, s[1] * 4 + 2)
            c2 = (c1[0], c1[1] - lh)
            boundp = [(c1[0] - r, c1[1]), (c2[0] - r, c2[1]), (c2[0] + r, c2[1]), (c1[0] + r, c1[1])]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("#####", reward)

        # up-right
        if delta == [4, 4]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 59.1
                r = 5.5
            c1 = (current[0] * 2 + 1+ wind_strength, current[1] * 2 + 1+ wind_strength)
            c2 = (c1[0] - lh, c1[1] + lh)
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("ur", reward)

        # up-left
        if delta == [-4, 4]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 59.1
                r = 5.5
            c1 = (current[0] * 2 + 1 - wind_strength, current[1] * 2 + 1+ wind_strength)
            c2 = (c1[0] - lh, c1[1] - lh)
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("ul", reward)      

        # down-right
        if delta == [4, -4]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 59.1
                r = 5.5
            c1 = (current[0] * 2 + 2+ wind_strength, current[1] * 2 + 2 - wind_strength)
            c2 = (c1[0] + lh, c1[1] + lh)
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("dr", reward)
        
        #down-left
        if delta == [-4, -4]:
            if self.b == 0:
                lh = 0
                r = 0
            else:
                lh = 59.1
                r = 5.5
            c1 = (current[0] * 2 + 1 - wind_strength, current[1] * 2 + 1 - wind_strength)
            c2 = (c1[0] + lh, c1[1] - lh)
            boundp = [(c1[0], c1[1] - r), (c2[0], c2[1] - r), (c2[0], c2[1] + r), (c1[0], c1[1] + r)]
            ff = polyFlag(g, poly1, poly2, poly3, poly4, poly5, boundp, c1, c2, r)
            reward = reward - 1000 * ff
            print("dl", reward)

        if self._cliff[tuple(next_pos)]:
            reward = reward -1000
            print("######", reward)

        if (next_pos == self.des).all():
            reward = 1
            print("#######", reward)

        is_done = False
        if tuple(next_pos) == self.des or np.sqrt((next_pos[0]- self.des[0])**2 + (next_pos[1]- self.des[1])**2) <4:
            is_done = True #If the agent has reached the destination (tuple(next_pos) == self.des), it is given a reward of 1

        return [(1.0, next_pos, reward, is_done)] #since the probability of the action being taken is always 1 (i.e., the action will always be taken), the list only contains a single tuple
        # return [(1.0, next_s, reward, is_done)] #since the probability of the action being taken is always 1 (i.e., the action will always be taken), the list only contains a single tuple


"""
step method of the UAVEnv class, which is responsible for simulating the movement of the agent in the environment and updating the state of the environment according to the action taken by the agent. The step method updates the state of the environment and determines if the agent has reached its destination or collided with an obstacle.It takes in the current state, s, and the action taken by the agent, a, as input and returns the new state, the reward received by the agent, a boolean value indicating whether the episode is complete, and a dictionary of additional information.


_calculate_reward method, which is a helper function used to compute the reward for the agent based on the current and previous states and the action taken. The _calculate_reward method is called by the step method as part of the process of updating the state of the environment. the _calculate_reward method computes the reward for the agent based on the current and previous states and the action taken. It takes in the current state, s, the action taken by the agent, a, and the new state, new_s, as input and returns the reward received by the agent as output.

The step method and the _calculate_reward method are related in that the step method uses the _calculate_reward method to determine the reward for the agent based on the current and previous states and the action taken, but they are not repeating the same task. 


"""

