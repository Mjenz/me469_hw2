import numpy as np
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt

def create_dataset(input_data, gnd_data):
    # load raw dataset
    input = np.loadtxt(input_data)
    gnd = np.loadtxt(gnd_data)

    # input = np.loadtxt("ds0/ds0_Control.dat")
    # gnd = np.loadtxt("ds0/ds0_Groundtruth.dat")
    
    # create function to interpolate ground truth locations
    
    f_x = interp1d(gnd[:,0], gnd[:,1], kind='linear', fill_value='extrapolate')
    f_y = interp1d(gnd[:,0], gnd[:,2], kind='linear', fill_value='extrapolate')
    f_theta = interp1d(gnd[:,0], gnd[:,3], kind='linear', fill_value='extrapolate')

    # get ground truth points that line up with input time
    times = input[:,0]
    gnd_x = f_x(times)
    gnd_y = f_y(times)
    gnd_theta = f_theta(times)

    # take difference to get change in position
    d_times = np.diff(times)
    d_x = np.diff(gnd_x)
    d_y = np.diff(gnd_y)
    d_theta = np.diff(gnd_theta)

    # wrap angles
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi

    # get input velocities
    v = input[:-1,1]
    w = input[:-1,2]

    # scale input velocities by timestep, and add dimensionality for x and y
    vel_scaled = np.column_stack([d_times, v, w, np.cos(gnd_theta[:-1]), np.sin(gnd_theta[:-1])])
    print(f'X ({np.shape(vel_scaled)}:\n{vel_scaled}\n')

    # save data, cut off last control input since we don't have data for its effect
    # np.savetxt('learning_dataset_gnd_truth.csv', np.array([input[:-1,0],d_x, d_y, d_theta]).T,header="time, dx, dy, dtheta")
    # np.savetxt('learning_dataset_controls.csv',  np.column_stack([input[:-1,0], vel_scaled]),header="time, vdt, omegadt")    
    np.save('learning_dataset_gnd_truth', np.array([input[:-1,0],d_x, d_y, d_theta]).T)
    np.save('learning_dataset_controls',  np.column_stack([input[:-1,0], vel_scaled]))

    print(f'input ({np.shape(input[:-1,0])}:\n{input[:-1,0]}\n')
    print(f'd_x ({np.shape(d_x)}:\n{d_x}\n')
    print(f'd_y ({np.shape(d_y)}:\n{d_y}\n')
    print(f'd_theta ({np.shape(d_theta)}:\n{d_theta}\n')

class LWL():
    """Locally Weighted Linear Regression class."""

    def __init__(self, control_data, gnd_truth_data,p=False):
        # extract from datafile
        # self.x = np.loadtxt(control_data)[:, 1:]
        # self.y = np.loadtxt(gnd_truth_data)[:, 1:]        
        self.x = np.load(control_data,allow_pickle=1)[:, 1:]
        self.y = np.load(gnd_truth_data,allow_pickle=1)[:, 1:]

        self.p = p

        # normalize inputs
        self.x_mean = self.x.mean(axis = 0)
        self.x_std = self.x.std(axis = 0)
        self.x = (self.x - self.x_mean) / self.x_std

        # normalize outputs
        self.y_mean = self.y.mean(axis = 0)
        self.y_std = self.y.std(axis = 0)
        self.y = (self.y - self.y_mean) / self.y_std

        # x_q = np.array([[1.0, 1.0, 1.0]])
        # y_q = self.predict(x_q)

        if self.p:
            print(f'x ({np.shape(self.x)}:\n{self.x}\n')
            print(f'y ({np.shape(self.y)}:\n{self.y}\n')
            # print(f'x_q ({np.shape(x_q)}:\n{x_q}\n')
            # print(f'y_q ({np.shape(y_q)}:\n{y_q}\n')

    def calc_weight_w(self, x_q):
        tau = 0.05
        self.w = np.exp(-(np.sum((self.x - x_q)**2, axis=1)) / (2 * tau ** 2))

        if self.p:
            print(f'w ({np.shape(self.w)}:\n{self.w}\n')

    def calc_weight_theta(self, x_q):
        # normalize input
        x_q = (x_q - self.x_mean) / self.x_std

        # calculate weights
        self.calc_weight_w(x_q)
        w_mat = np.diag(self.w)

        # add biasing term
        self.x_biased = np.hstack([np.ones((self.x.shape[0], 1)), self.x])  
        self.x_q_biased = np.hstack([1.0, x_q.ravel()])                     

        # calculate theta weights with normalization value to preserve stability
        lam = 1e-6
        self.theta = np.linalg.inv(self.x_biased.T @ w_mat @ self.x_biased + lam * np.eye(self.x_biased.shape[1])) @ self.x_biased.T @ w_mat @ self.y
        # self.theta = np.linalg.pinv(self.x_biased.T @ w_mat @ self.x_biased) @ self.x_biased.T @ w_mat @ self.y

        if self.p:
            print(f'w_mat ({np.shape(w_mat)}:\n{w_mat}\n')
            print(f'theta ({np.shape(self.theta)}:\n{self.theta}\n')

    def predict(self, x_q):
        self.calc_weight_theta(x_q)
        y_hat_normalized = self.x_q_biased @ self.theta
        return  y_hat_normalized * self.y_std + self.y_mean

class DiffDrive():
    """Simulated Diff Drive robot using dead reckoning or LWLR."""

    def __init__(self, control_data, gnd_data):
        self.lwl = LWL("learning_dataset_controls.npy","learning_dataset_gnd_truth.npy")
        self.control_data = np.loadtxt(control_data)
        self.gnd_truth_dataset = np.loadtxt(gnd_data)
        print(f'control_data ({np.shape(self.control_data)}:\n{self.control_data}\n')


    def simulate(self,x_init,y_init,theta_init,vis=False):
        """ simulate the state estimation system only usign the motion model          
        """
        dataset = self.control_data
        
        # initialize the history arrays
        self.DR_x_sim_hist = [x_init]
        self.DR_y_sim_hist = [y_init]
        self.DR_theta_sim_hist = [theta_init]

        self.ML_x_sim_hist = [x_init]
        self.ML_y_sim_hist = [y_init]
        self.ML_theta_sim_hist = [theta_init]

        # setup init conditions
        self.DR_prev_t_sim = dataset[0][0] - (dataset[1][0]-dataset[0][0])        # simulate similar time step to first one
        self.DR_x_sim = x_init
        self.DR_y_sim = y_init
        self.DR_theta_sim = theta_init

        self.ML_prev_t_sim = dataset[0][0] - (dataset[1][0]-dataset[0][0])        # simulate similar time step to first one
        self.ML_x_sim = x_init
        self.ML_y_sim = y_init
        self.ML_theta_sim = theta_init

        # iterate through control data, simulate system only using motion model
        for pt in dataset:
            self.dead_reckoning(pt)
            self.learned_motion_model(pt)
            

    def dead_reckoning(self ,pt):
        # get controls and calculate time step
        dt = pt[0] - self.DR_prev_t_sim        # change in time
        self.DR_prev_t_sim = pt[0]             # save previous time
        v = pt[1]                           # velocity
        w = pt[2]                           # angular velocity

        # model effect of controls
        self.DR_x_sim += v * dt * np.cos(self.DR_theta_sim)
        self.DR_y_sim += v * dt * np.sin(self.DR_theta_sim)
        self.DR_theta_sim += w * dt

        # wrap angles
        self.DR_theta_sim = (self.DR_theta_sim + np.pi) % (2 * np.pi) - np.pi

        # save values
        self.DR_x_sim_hist.append(self.DR_x_sim)
        self.DR_y_sim_hist.append(self.DR_y_sim)
        self.DR_theta_sim_hist.append(self.DR_theta_sim)

    def learned_motion_model(self, pt):
        dt = (pt[0]-self.ML_prev_t_sim)
        out = self.lwl.predict([dt, pt[1], pt[2], np.cos(self.ML_theta_sim), np.sin(self.ML_theta_sim)])
        self.ML_prev_t_sim = pt[0]             # save previous time

        self.ML_x_sim += out[0]  
        self.ML_y_sim += out[1]
        self.ML_theta_sim += out[2]

        # wrap angles
        self.ML_theta_sim = (self.ML_theta_sim + np.pi) % (2 * np.pi) - np.pi

        # save values
        self.ML_x_sim_hist.append(self.ML_x_sim)
        self.ML_y_sim_hist.append(self.ML_y_sim)
        self.ML_theta_sim_hist.append(self.ML_theta_sim)

    def visualize(self):
        # load gnd truth data
        gnd_trth_dat = self.gnd_truth_dataset

        # plot figure for code A part 2
        plt.figure()
        plt.plot(gnd_trth_dat[:,1],gnd_trth_dat[:,2],'g-')          # gnd truth path
        plt.plot(self.DR_x_sim_hist,self.DR_y_sim_hist,'-m')              # motion model only results
        plt.plot(self.ML_x_sim_hist,self.ML_y_sim_hist,'-b')              # motion model only results

        # # add arrows
        # decimation_factor = int(len(gnd_trth_dat)/1000)             # add arrows only for each one thousand points
        # for ii in range(0,len(gnd_trth_dat),decimation_factor):     # calculate arrow direction
        #     dx = .1 * np.cos(gnd_trth_dat[ii][3])
        #     dy = .1 * np.sin(gnd_trth_dat[ii][3])
        #     plt.arrow(gnd_trth_dat[ii][1], gnd_trth_dat[ii][2], dx, dy, head_width=0.07, head_length=0.09, fc='green', ec='black')

        # decimation_factor = int(len(self.DR_x_sim_hist)/1000)          # add arrows only for each one thousand points
        # for ii in range(0,len(self.DR_x_sim_hist),decimation_factor):  # calculate arrow direction
        #     dx = .1 * np.cos(self.DR_theta_sim_hist[ii])
        #     dy = .1 * np.sin(self.DR_theta_sim_hist[ii])
        #     plt.arrow(self.DR_x_sim_hist[ii], self.DR_y_sim_hist[ii], dx, dy, head_width=0.07, head_length=0.09, fc='magenta', ec='black')

        plt.legend(['Ground truth','Dead Reckoning', 'Locally Weighted Regression'])
        plt.title('Motion Model Only State Estimation')
        plt.xlabel('x position [m]')
        plt.ylabel('y position [m]')
        plt.show()

        


def main():

    create_dataset("ds0/ds0_Control_cropped.dat","ds0/ds0_Groundtruth_cropped.dat")
    ml = LWL("learning_dataset_controls.npy","learning_dataset_gnd_truth.npy")
    ddrive = DiffDrive("ds0/ds0_Control_cropped.dat","ds0/ds0_Groundtruth_cropped.dat")
    ddrive.simulate(1.29812900, 1.88315210, 2.82870000)
    ddrive.visualize()

    # create_dataset("ds1/ds1_Control_cropped.dat","ds1/ds1_Groundtruth_cropped.dat")
    # ml = LWL("learning_dataset_controls.npy","learning_dataset_gnd_truth.npy")
    # ddrive = DiffDrive("ds1/ds1_Control_cropped.dat","ds1/ds1_Groundtruth_cropped.dat")
    # ddrive.simulate(0.98038490, -4.99232180, 1.44849633)
    # ddrive.visualize()

if __name__ == "__main__":
    main()