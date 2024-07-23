import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk

class RBF:
    # initialization
    def __init__(self, M, learning_rate=0.01, max_epoch=500, threshold=1e-3):
        self.M = M
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.threshold = threshold
    
    # load file data
    def load_data(self, file_path):
        data = np.loadtxt(file_path) #(N,2)
        self.X_train = data[:, 0]  #(N,)
        self.y_train = data[:, 1]  #(N,)

    # load model parameters from files
    def load_params(self, M, folder_name):
        self.centers = np.loadtxt(f'./{folder_name}/rbfnet_c{M}.txt')
        self.W = np.loadtxt(f'./{folder_name}/rbfnet_w{M}.txt')
        self.maxd = max(self.centers) - min(self.centers)
        self.sigma2 = 0.05 * self.maxd**2 / (2 * self.M)
    
    def gaus(self, X, C):
        return np.exp(-((X[:, None] - C[None, :])**2) / (2 * self.sigma2))
    
    def train(self, file_path):
        self.load_data(file_path)
        self.centers = np.linspace(self.X_train.min(), self.X_train.max(), self.M)
        self.maxd = max(self.centers) - min(self.centers)
        self.sigma2 = 0.05 * self.maxd**2 / (2 * self.M)
        self.cost_ = []
        np.random.seed(42)
        self.W = np.random.randn(self.M)

        for epoch in range(self.max_epoch):
            phi = self.gaus(self.X_train, self.centers)
            y_pred = phi.dot(self.W)
            error = self.y_train - y_pred
            cost = (error**2).sum()/2  # loss function is squared loss
            self.cost_.append(cost)
            self.W += self.learning_rate * phi.T.dot(error)
            if cost < self.threshold:
                print(f'Converged at epoch {epoch}')
                break

        # plot cost function
        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"M={self.M} | Cost={cost:.3f}")
        plt.show()

    def params_to_file(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.savetxt(f'./{folder_name}/rbfnet_c{self.M}.txt', self.centers)
        np.savetxt(f'./{folder_name}/rbfnet_w{self.M}.txt', self.W)

    def test_and_display(self, M, file_path, folder_name):
        self.load_data(file_path)
        self.load_params(M,folder_name)
        test_data = np.loadtxt("test.txt")
        sorted_test_data = test_data[test_data[:, 0].argsort()]

        X_test = sorted_test_data[:,0]
        y_true = sorted_test_data[:,1]

        phi = self.gaus(X_test, self.centers)
        y_pred = phi.dot(self.W)

        test_error = np.mean((y_true - y_pred)**2)
        plt.figure(figsize=(10, 6))
        plt.plot(self.X_train, self.y_train, 'bo', label='Training Data')
        plt.plot(X_test, y_true, color="g", label='True Function')
        plt.plot(X_test, y_pred, color='r', label='RBF Network Output')
        plt.title(f'M={M}__Test Error: {test_error:.4f}')
        plt.legend()
        plt.show()


def show_gui(file_path,folder_name):
    ###GUI
    root = tk.Tk()
    root.title("menu")
    # set window size
    root.geometry("400x300")

    # add label
    label1 = tk.Label(root, text="基底数M",width=10,height=4)
    # display
    label1.grid(column = 0,row = 0)
    # add entry
    EditBox1 = tk.Entry()
    EditBox1.grid(column = 1,row = 0)

    # add label
    label2 = tk.Label(root, text="学習率",width=10,height=4)
    # display
    label2.grid(column =0,row = 5)
    EditBox2 = tk.Entry()
    EditBox2.grid(column = 1,row = 5)

    # add label
    label3 = tk.Label(root, text="回数",width=10,height=4)
    # display
    label3.grid(column = 0,row = 10)
    EditBox3 = tk.Entry()
    EditBox3.grid(column = 1,row = 10)

    # add button
    def goto_train():
        M = int(EditBox1.get())
        learning_rate = float(EditBox2.get()) if EditBox2.get() else 0.01
        max_epoch = int(EditBox3.get()) if EditBox3.get() else 500
        rbf = RBF(M, learning_rate, max_epoch)
        rbf.load_data(file_path)
        rbf.train(file_path)
        rbf.params_to_file(folder_name)
        
    def goto_result():
        M = int(EditBox1.get())
        rbf = RBF(M)
        rbf.test_and_display(M,file_path,folder_name)

    button = tk.Button(root, text="学習", command=goto_train,width=20,height = 3)
    button.grid(column = 0,row = 50,columnspan=3, rowspan = 3)
    button2 = tk.Button(root, text="結果表示", command=goto_result,width=20,height = 3)
    button2.grid(column = 10,row = 50,columnspan=3,rowspan = 3)
    root.mainloop()


class RBF_C(RBF):
    def train(self, file_path):
        self.load_data(file_path)
        self.centers = np.linspace(self.X_train.min(), self.X_train.max(), self.M)
        self.maxd = max(self.centers) - min(self.centers)
        self.sigma2 = 0.05 * self.maxd**2 / (2 * self.M)
        self.cost_ = []
        np.random.seed(42)
        self.W = np.random.randn(self.M)
        for epoch in range(self.max_epoch):
            phi = self.gaus(self.X_train, self.centers) #(N, M)
            dphi_dc = np.linalg.norm(self.X_train[:,None] - self.centers[None,:], axis=1)/self.sigma2 #(N, M)

            y_pred = phi.dot(self.W)
            error = self.y_train - y_pred
            cost = (error**2).sum()/2  # loss function is squared loss
            self.cost_.append(cost)
            self.W += self.learning_rate * phi.T.dot(error)
            self.centers += 0.01*self.learning_rate/self.X_train.shape[0] * np.dot(phi.T*self.W[:,None]*dphi_dc, error)
            if cost < self.threshold:
                print(f'Converged at epoch {epoch}')
                break

        # plot cost function
        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"M={self.M} | Cost={cost:.3f}")
        plt.show()

    def test_and_display(self, M, file_path, folder_name):
        self.load_data(file_path)
        self.load_params(M,folder_name)
        test_data = np.loadtxt("test.txt")
        sorted_test_data = test_data[test_data[:, 0].argsort()]

        X_test = sorted_test_data[:,0]
        y_true = sorted_test_data[:,1]

        phi = self.gaus(X_test, self.centers)
        y_pred = phi.dot(self.W)

        test_error = np.mean((y_true - y_pred)**2)
        plt.figure(figsize=(10, 6))
        plt.plot(self.X_train, self.y_train, 'bo', label='Training Data')
        plt.plot(X_test, y_true, color="g", label='True Function')
        plt.plot(X_test, y_pred, color='r', label='RBF Network Output')
        plt.title(f'M={M}__Test Error: {test_error:.4f}')
        plt.legend()
        plt.show()

# RBF network for 3D data
class RBF_3d:
    def __init__(self, M,  learning_rate=0.1, max_epoch=1000, threshold=1e-6):
        self.M = M
        self.M2 = M**2
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.threshold = threshold

    def _initialize_centers_and_sigma(self):
        x_c = np.linspace(self.X_train[:,0].min(), self.X_train[:,0].max(), self.M)  # (M,)
        y_c = np.linspace(self.X_train[:,1].min(), self.X_train[:,1].max(), self.M)  # (M,)
        xx_c, yy_c = np.meshgrid(x_c, y_c)  # (M,M) (M,M)
        centers = np.concatenate((xx_c.reshape(-1,1), yy_c.reshape(-1, 1)), axis=1)  # (M**2,2)
        maxd = np.linalg.norm(centers[0] - centers[-1])
        sigma2 = 0.05 * maxd**2 / (2 * self.M)
        return centers, sigma2

    def _kernel(self, point, cen):  # phi, which in this case, is gaussian function
        euc_dis_squared = np.linalg.norm(point - cen, axis=1)
        return np.exp(-euc_dis_squared / (2 * self.sigma2))  # (M**2,)

    def predict(self, X):
        y = []
        for feature in X:
            phi_i = self._kernel(feature, self.centers)
            y.append(sum(phi_i * self.W))
        return np.array(y)

    def train(self,file_path):
        self.data = np.loadtxt(file_path)
        self.X_train = self.data[:, :-1]  # (N,2)
        self.y_train = self.data[:, -1]  # (N,)
        self.centers, self.sigma2 = self._initialize_centers_and_sigma() # (M**2,2), 
        np.random.seed(42)
        self.W = np.random.randn(self.M2)  # (M**2,)
        cost_ = []
        # Pre-compute phi for all training examples
        phi = np.array([self._kernel(feature, self.centers) for feature in self.X_train])  # (N, M**2)
        
        for epoch in range(self.max_epoch):
            # Vectorized prediction
            y_pred = np.dot(phi, self.W)  # (N,)
            error = self.y_train - y_pred  # (N,)
            cost = np.sum(np.square(error)) / (2 * len(self.y_train))
            cost_.append(cost)
            # Vectorized weight update
            self.W += self.learning_rate / self.X_train.shape[0]**0.5 * np.dot(phi.T, error)  # (M**2,)
            if cost < self.threshold:
                print(f'Converged at epoch {epoch}')
                break
        
        # plot cost function
        plt.plot(range(1, len(cost_) + 1), cost_, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"M={self.M} | Cost={cost:.3f}")
        plt.show()
        return self.W

    def plot3d(self):
        N = int(np.sqrt(self.X_train.shape[0]))
        xx, yy = self.X_train[:,0].reshape(N, N), self.X_train[:,1].reshape(N, N)
        zz = self.predict(self.X_train).reshape(N, N)
        
        test_data = np.loadtxt("./kadai8_data/test_data_8.txt")
        N_test = int(np.sqrt(test_data.shape[0]))
        x_real = test_data[:,0]
        y_real = test_data[:,1]
        z_real = test_data[:,2]

        train_data = np.loadtxt("./kadai8_data/train_data_8.txt")
        N_train = int(np.sqrt(train_data.shape[0]))
        x_train = train_data[:,0]
        y_train = train_data[:,1]
        z_train = train_data[:,2]

        # Create the figure and 3D subplots
        fig = plt.figure(figsize=(10, 5))

        # First subplot: Real Plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(x_real.reshape(N_test,N_test), y_real.reshape(N_test, N_test), z_real.reshape(N_test, N_test), cmap='viridis')
        ax1.set_title('Real Plot')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Second subplot: Train Plot
        ax1 = fig.add_subplot(132, projection='3d')
        ax1.scatter(x_train, y_train, z_train, c=z_train, cmap='viridis')
        ax1.set_title('Train Plot')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Third subplot: RBF Plot
        ax2 = fig.add_subplot(133, projection='3d')
        ax2.scatter3D(xx, yy, zz, c=zz, cmap='viridis')
        ax2.set_title('RBF Plot')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        plt.show()

# RBF network for 3D data with center update
# This class takes too long to train 
class RBF_3d_C:
    def __init__(self, file_path, M=50, max_epoch=1000, eta=0.1, threshold=1e-3):
        self.file_path = file_path
        self.M = M
        self.M2 = M**2
        self.max_epoch = max_epoch
        self.eta = eta
        self.threshold = threshold
        self.data = np.loadtxt(file_path)
        self.X_train = self.data[:, :-1]  # (N, 2)
        self.y_train = self.data[:, -1]  # (N,)
        self.x_c = np.linspace(self.X_train[:, 0].min(), self.X_train[:, 0].max(), M)  # (M,)
        self.y_c = np.linspace(self.X_train[:, 1].min(), self.X_train[:, 1].max(), M)  # (M,)
        self.xx_c, self.yy_c = np.meshgrid(self.x_c, self.y_c)  # (M, M), (M, M)
        self.centers = np.concatenate((self.xx_c.reshape(-1, 1), self.yy_c.reshape(-1, 1)), axis=1)  # (M**2, 2)
        self.maxd = np.linalg.norm(self.centers[0] - self.centers[-1])
        self.sigma2 = 0.05 * self.maxd**2 / (2 * M)

    def kernel(self, X, C):
        X_expanded = X[:, np.newaxis, :]
        C_expanded = C[np.newaxis, :, :]
        distances = np.sum((X_expanded - C_expanded) ** 2, axis=2)
        return np.exp(-distances / (2 * self.sigma2))

    def predict(self, X, C, W):
        phi = self.kernel(X, C)
        return np.dot(phi, W)

    def train(self):
        cost_ = []
        np.random.seed(42)
        W = np.random.randn(self.M2)
        C = self.centers
        N = self.X_train.shape[0]
        phi = self.kernel(self.X_train, C)  # (N, M**2)

        for epoch in range(self.max_epoch):
            # phi = self.kernel(self.X_train, C)  # (N, M**2)
            y_pred = np.dot(phi, W)
            error = self.y_train - y_pred
            cost = np.sum(np.square(error)) / (2 * np.sqrt(N))
            cost_.append(cost)
            if cost < self.threshold:
                print(f'Converged at epoch {epoch}')
                break

            # Vectorized weight update
            W += (self.eta / np.sqrt(N)) * np.dot(phi.T, error)

            # Update centers
            X_expanded = self.X_train[:, np.newaxis, :]  # (N, 1, 2)
            C_expanded = C[np.newaxis, :, :]  # (1, M**2, 2)
            diff = X_expanded - C_expanded  # (N, M**2, 2)
            dphi_dc = diff / self.sigma2 * phi[:, :, np.newaxis]  # (N, M**2, 2)
            C += (0.1 * self.eta / np.sqrt(N)) * np.sum(dphi_dc * error[:, np.newaxis, np.newaxis], axis=0)

        # plot cost function
        plt.plot(range(1, len(cost_) + 1), cost_, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"M={self.M} | Cost={cost:.3f}")
        plt.show()
        self.W = W
        self.C = C

    def plot3D(self):
        # Reshape data for plotting
        N = int(np.sqrt(self.X_train.shape[0]))
        xx, yy = self.X_train[:,0].reshape(N, N), self.X_train[:,1].reshape(N, N)
        zz = self.predict(self.X_train, self.C, self.W).reshape(N, N)
        z_real = np.sin(xx) + np.cos(yy)

        # Create the figure and 3D subplots
        fig = plt.figure(figsize=(10, 5))

        # First subplot: Real Plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter3D(xx, yy, z_real, c=z_real, cmap='viridis')
        ax1.set_title('Real Plot')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Second subplot: RBF Plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter3D(xx, yy, zz, c=zz, cmap='viridis')
        ax2.set_title('RBF Plot')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        plt.show()