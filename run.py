import numpy as np
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import time

def create_noisy_sin_data():
    # generate linspace
    space = np.linspace(0,2*np.pi,500)

    # initialize dataset
    X = np.array(space) + ((np.random.rand(len(space)) * 2) - 1) / 2

    # pass though sin function, add noise
    Y = np.sin(X) + ((np.random.rand(len(X)) * 2) - 1) / 2

    # save
    np.save('noisy_sin_data_in', np.array([X, X]).T)
    np.save('noisy_sin_data_out', np.array([Y, Y]).T)    

    plt.scatter(X,Y)
    plt.title("Locally Weighted Regression Noisy Sin Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def create_dataset(input_data, gnd_data, dataset, DR_input=False):
    # load raw dataset
    input = np.loadtxt(input_data)
    gnd = np.loadtxt(gnd_data)

    # create function to interpolate ground truth locations, this accounts for time recording difference
    f_x = interp1d(gnd[:,0], gnd[:,1], kind='linear', fill_value='extrapolate')
    f_y = interp1d(gnd[:,0], gnd[:,2], kind='linear', fill_value='extrapolate')
    f_theta = interp1d(gnd[:,0], gnd[:,3], kind='linear', fill_value='extrapolate')

    # get ground truth points that line up with input time
    times = input[:,0] # use the controller times
    gnd_x = f_x(times) # approximate positions at controller recorded times using linear interp function
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
    if not DR_input:
        vel_scaled = np.column_stack([d_times, v, w, np.cos(gnd_theta[:-1]), np.sin(gnd_theta[:-1])])
    else:
        vel_scaled = np.column_stack([d_times * v * np.cos(gnd_theta[:-1]),d_times * v * np.sin(gnd_theta[:-1]), d_times * w])

    # save data, cut off last control input since we don't have data for its effect, use npy for efficiency
    np.save('learning_dataset_gnd_truth_' + dataset, np.array([input[:-1,0],d_x, d_y, d_theta]).T)
    np.save('learning_dataset_controls_' + dataset,  np.column_stack([input[:-1,0], vel_scaled]))

class LWL():
    """Locally Weighted Linear Regression class."""

    def __init__(self, control_data, gnd_truth_data,lam=1e-6,tau=0.1,use_knn=False, k=200, p=False):
        # extract from datafile   
        self.x = np.load(control_data,allow_pickle=1)[:, 1:]
        self.y = np.load(gnd_truth_data,allow_pickle=1)[:, 1:]

        # print things?
        self.p = p

        # use knn?
        self.use_knn = use_knn

        # how many neighbors
        self.k = k

        # get regularization term
        self.lam = lam
        
        # get bandwidth parameter
        self.tau = tau

        # # normalize inputs
        # self.x_mean = self.x.mean(axis = 0)
        # self.x_std = self.x.std(axis = 0)
        # self.x = (self.x - self.x_mean) / self.x_std

        # # normalize outputs
        # self.y_mean = self.y.mean(axis = 0)
        # self.y_std = self.y.std(axis = 0)
        # self.y = (self.y - self.y_mean) / self.y_std

        # precalculate x biased
        self.x_biased = np.hstack([np.ones((self.x.shape[0], 1)), self.x])  

    def knn_indicies(self, x_q_normalized):

        d = np.sum((self.x - x_q_normalized)**2, axis=1)
        indexes = np.argpartition(d,self.k)[:self.k]
        return indexes

    def calc_weight_w(self, x, x_q):
        # weights, gaussian function
        self.w = np.exp(-(np.sum((x- x_q)**2, axis=1)) / (2 * self.tau ** 2))

    def calc_weight_theta_with_knn(self, x_q):
        # normalize input
        # x_q = (x_q - self.x_mean) / self.x_std

        idx = self.knn_indicies(x_q)
        x_local = self.x[idx]
        y_local = self.y[idx]
        x_biased_local = self.x_biased[idx]

        # calculate weights w
        self.calc_weight_w(x_local, x_q)

        # add biasing term
        # self.x_q_biased = np.hstack([1.0, x_q.ravel()])                     
        self.x_q_biased = np.hstack([1.0, x_q])                     

        # set up equation
        A = x_biased_local.T @ (x_biased_local * self.w[:, None]) 
        B = x_biased_local.T @ (y_local * self.w[:, None])

        # solve
        self.theta = np.linalg.solve(A + self.lam*np.eye(A.shape[0]), B)

    def calc_weight_theta(self, x_q):
        # normalize input
        # x_q = (x_q - self.x_mean) / self.x_std

        # calculate weights
        self.calc_weight_w(self.x, x_q)

        # add biasing term
        # self.x_q_biased = np.hstack([1.0, x_q.ravel()])                     
        self.x_q_biased = np.hstack([1.0, x_q])                     

        # set up equation
        A = self.x_biased.T @ (self.x_biased * self.w[:, None]) 
        B = self.x_biased.T @ (self.y * self.w[:, None])

        # solve
        self.theta = np.linalg.solve(A + self.lam*np.eye(A.shape[0]), B)

    def predict(self, x_q):
        if self.use_knn:
            self.calc_weight_theta_with_knn(x_q)
        else:
            self.calc_weight_theta(x_q)
        y_hat_normalized = self.x_q_biased @ self.theta
        return  y_hat_normalized #* self.y_std + self.y_mean

class DiffDrive():
    """Simulated Diff Drive robot using dead reckoning or LWLR."""

    def __init__(self, control_data, gnd_data,control_training_data, groundtruth_training_data, lam=1e-6, tau=0.1, DR_input=False, use_knn=False, k=200):
        self.lwl = LWL(control_training_data,groundtruth_training_data,lam, tau, use_knn,k)
        self.control_data = np.loadtxt(control_data)
        self.gnd_truth_dataset = np.loadtxt(gnd_data)
        self.DR_input = DR_input

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
        self.ML_prev_v_sim = 0.0
        self.ML_prev_w_sim = 0.0
        self.ML_prev_theta_sim = 0.0
        self.ML_prev_dt_sim = self.ML_prev_t_sim
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
        if not self.DR_input:
            out = self.lwl.predict([dt, pt[1], pt[2], np.cos(self.ML_theta_sim), np.sin(self.ML_theta_sim)])
        else:
            out = self.lwl.predict([dt * pt[1] * np.cos(self.ML_theta_sim),dt * pt[1] * np.sin(self.ML_theta_sim), pt[2] * dt])
        self.ML_prev_t_sim = pt[0]             # save previous time
        self.ML_prev_v_sim = pt[1]             # save previous v
        self.ML_prev_w_sim = pt[2]             # save previous w
        self.ML_prev_dt_sim = dt
        self.ML_prev_theta_sim = self.ML_theta_sim
        self.ML_x_sim += out[0]  
        self.ML_y_sim += out[1]
        self.ML_theta_sim += out[2]

        # wrap angles
        self.ML_theta_sim = (self.ML_theta_sim + np.pi) % (2 * np.pi) - np.pi

        # save values
        self.ML_x_sim_hist.append(self.ML_x_sim)
        self.ML_y_sim_hist.append(self.ML_y_sim)
        self.ML_theta_sim_hist.append(self.ML_theta_sim)

    def visualize(self, include_DR=False):
        # load gnd truth data
        gnd_trth_dat = self.gnd_truth_dataset

        # plot figure for code A part 2
        plt.figure()
        plt.plot(gnd_trth_dat[:,1],gnd_trth_dat[:,2],'g-',label="ground_truth")          # gnd truth path
        if include_DR:
            plt.plot(self.DR_x_sim_hist,self.DR_y_sim_hist,'-m',label="dead_reckoning")              # motion model only results
        plt.plot(self.ML_x_sim_hist,self.ML_y_sim_hist,'-b',label='LWL')              # motion model only results

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

        plt.legend()
        plt.title('Motion Model Only State Estimation')
        plt.xlabel('x position [m]')
        plt.ylabel('y position [m]')
        plt.show()

        
class CodeATest():
    def __init__(self, new_data=False):

        # create new dataset if desired
        if new_data:
            create_noisy_sin_data()
        pass
        
        # create instance of lwl
        self.lwl = LWL('noisy_sin_data_in.npy', 'noisy_sin_data_out.npy')

        # read ground truth
        self.X = np.load('noisy_sin_data_in.npy')
        self.Y = np.load('noisy_sin_data_out.npy')

        # # create test inputs
        self.x_test =  np.linspace(0, 3*np.pi,20)

    def test(self):
        self.y_hat = np.array([self.lwl.predict(x) for x in self.x_test])
        comp = np.linspace(0,3*np.pi)
        plt.plot(comp, np.sin(comp),'-k',label='true_sin')
        plt.scatter(self.X,self.Y, label='training_data')
        plt.scatter(self.x_test,self.y_hat.flatten(), label='test_data')
        plt.title("Locally Weighted Regression Noisy Sin Test")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

def main():
    # test on noisy sin data
    # codeA = CodeATest(1)
    # codeA.test()

    # ds1 part 1 data (uses back 20% as validation)

    # start = time.time()
    # use_dead_reckoning_input = False
    # create_dataset("ds1/ds1_Control_training.dat","ds1/ds1_Groundtruth_training.dat","ds1",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds1/ds1_Control_validation.dat",
    #             gnd_data="ds1/ds1_Groundtruth_validation.dat", 
    #             control_training_data="learning_dataset_controls_ds1.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds1.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input, 
    #             use_knn=0,
    #             k=200)
    #         ddrive.simulate(3.73849410,-1.75510540,1.90469633 )
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()
    # end = time.time()
    # print(f"Total runtime: {end-start} sec\n")

    # start = time.time()
    # use_dead_reckoning_input = True
    # create_dataset("ds1/ds1_Control_training.dat","ds1/ds1_Groundtruth_training.dat","ds1",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds1/ds1_Control_validation.dat",
    #             gnd_data="ds1/ds1_Groundtruth_validation.dat", 
    #             control_training_data="learning_dataset_controls_ds1.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds1.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input, 
    #             use_knn=0,
    #             k=200)
    #         ddrive.simulate(3.73849410,-1.75510540,1.90469633 )
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()
    # end = time.time()
    # print(f"Total runtime: {end-start} sec\n")

    # ds1 part 2 data (uses front 20% as validation)

    start = time.time()
    use_dead_reckoning_input = False
    create_dataset("ds1/ds1_Control_training2.dat","ds1/ds1_Groundtruth_training2.dat","ds1",use_dead_reckoning_input)
    t = [0.01, 0.05, 0.1, 0.5]
    l = [1e-6]
    for ii in range(len(t)):
        for jj in range(len(l)):
            ddrive = DiffDrive(
                control_data="ds1/ds1_Control_validation2.dat",
                gnd_data="ds1/ds1_Groundtruth_validation2.dat", 
                control_training_data="learning_dataset_controls_ds1.npy",
                groundtruth_training_data="learning_dataset_gnd_truth_ds1.npy",
                lam=l[jj],
                tau=t[ii],
                DR_input=use_dead_reckoning_input, 
                use_knn=0,
                k=25)
            ddrive.simulate(0.98038490, -4.99232180, 1.44849633)
            print(f'tau:{t[ii]}, lam: {l[jj]}')
            ddrive.visualize()
    end = time.time()
    print(f"Total runtime: {end-start} sec\n")

    start = time.time()
    use_dead_reckoning_input = True
    create_dataset("ds1/ds1_Control_training2.dat","ds1/ds1_Groundtruth_training2.dat","ds1",use_dead_reckoning_input)
    t = [0.01, 0.05, 0.1, 0.5]
    l = [1e-6]
    for ii in range(len(t)):
        for jj in range(len(l)):
            ddrive = DiffDrive(
                control_data="ds1/ds1_Control_validation2.dat",
                gnd_data="ds1/ds1_Groundtruth_validation2.dat", 
                control_training_data="learning_dataset_controls_ds1.npy",
                groundtruth_training_data="learning_dataset_gnd_truth_ds1.npy",
                lam=l[jj],
                tau=t[ii],
                DR_input=use_dead_reckoning_input, 
                use_knn=0,
                k=25)
            ddrive.simulate(0.98038490, -4.99232180, 1.44849633)
            print(f'tau:{t[ii]}, lam: {l[jj]}')
            ddrive.visualize()
    end = time.time()
    print(f"Total runtime: {end-start} sec\n")

    ####################### Dataset 2 testing (poor performance) ################################

    # use_dead_reckoning_input = False
    # create_dataset("ds0/ds0_Control_training.dat","ds0/ds0_Groundtruth_training.dat","ds0",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds0/ds0_Control_validation.dat",
    #             gnd_data="ds0/ds0_Groundtruth_validation.dat", 
    #             control_training_data="learning_dataset_controls_ds0.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds0.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input,
    #             use_knn=0,
    #             k=100)
    #         ddrive.simulate(3.09823900,0.55749340,-0.82330000)
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()

    # use_dead_reckoning_input = True
    # create_dataset("ds0/ds0_Control_training.dat","ds0/ds0_Groundtruth_training.dat","ds0",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds0/ds0_Control_validation.dat",
    #             gnd_data="ds0/ds0_Groundtruth_validation.dat", 
    #             control_training_data="learning_dataset_controls_ds0.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds0.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input,
    #             use_knn=0,
    #             k=10)
    #         ddrive.simulate(3.09823900,0.55749340,-0.82330000)
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()
    
    # start = time.time()
    # use_dead_reckoning_input = False
    # create_dataset("ds0/ds0_Control_training2.dat","ds0/ds0_Groundtruth_training2.dat","ds0",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds0/ds0_Control_validation2.dat",
    #             gnd_data="ds0/ds0_Groundtruth_validation2.dat", 
    #             control_training_data="learning_dataset_controls_ds0.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds0.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input,
    #             use_knn=0,
    #             k=100)
    #         ddrive.simulate(1.29812900, 	 1.88315210 ,	 2.82870000  )
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()
    # end = time.time()
    # print(f"Total runtime: {end-start} sec\n")

    # start = time.time()
    # use_dead_reckoning_input = True
    # create_dataset("ds0/ds0_Control_training2.dat","ds0/ds0_Groundtruth_training2.dat","ds0",use_dead_reckoning_input)
    # t = [0.01]
    # l = [1e-6]
    # for ii in range(len(t)):
    #     for jj in range(len(l)):
    #         ddrive = DiffDrive(
    #             control_data="ds0/ds0_Control_validation2.dat",
    #             gnd_data="ds0/ds0_Groundtruth_validation2.dat", 
    #             control_training_data="learning_dataset_controls_ds0.npy",
    #             groundtruth_training_data="learning_dataset_gnd_truth_ds0.npy",
    #             lam=l[jj],
    #             tau=t[ii],
    #             DR_input=use_dead_reckoning_input,
    #             use_knn=0,
    #             k=25)
    #         ddrive.simulate(1.29812900, 	 1.88315210, 	 2.82870000  )
    #         print(f'tau:{t[ii]}, lam: {l[jj]}')
    #         ddrive.visualize()
    # end = time.time()
    # print(f"Total runtime: {end-start} sec\n")

    

if __name__ == "__main__":
    main()