import tensorflow as tf
import numpy as np
from pyDOE import lhs
import time
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_b_train, u_train, xt, x1_train, x2_train, layerM, layers1, layers2):
        #边界点及其数据
        self.x_b = X_b_train[:, 0:1]
        self.y_b = X_b_train[:, 1:2]
        self.z_b = X_b_train[:, 2:3]
        self.u_b = u_train
        #界面内部配置点
        self.x_f_1 = x1_train[:, 0:1]
        self.y_f_1 = x1_train[:, 1:2]
        self.z_f_1 = x1_train[:, 2:3]
        # 界面外部配置点
        self.x_f_2 = x2_train[:, 0:1]
        self.y_f_2 = x2_train[:, 1:2]
        self.z_f_2 = x2_train[:, 2:3]
        # 界面配置点
        self.x_t = xt[:, 0:1]
        self.y_t = xt[:, 1:2]
        self.z_t = xt[:, 2:3]
        #网络
        self.layers_M = layerM
        self.layers_1 = layers1
        self.layers_2 = layers2

        # Initialize NNs
        self.weights_M, self.biases_M = self.initialize_NN(layerM)
        self.weights_1, self.biases_1 = self.initialize_NN(layers1)
        self.weights_2, self.biases_2 = self.initialize_NN(layers2)


        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.xb = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]])
        self.yb = tf.placeholder(tf.float32, shape=[None, self.y_b.shape[1]])
        self.zb = tf.placeholder(tf.float32, shape=[None, self.z_b.shape[1]])
        self.ub = tf.placeholder(tf.float32, shape=[None, self.u_b.shape[1]])
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.x_f_1.shape[1]])
        self.y1 = tf.placeholder(tf.float32, shape=[None, self.y_f_1.shape[1]])
        self.z1 = tf.placeholder(tf.float32, shape=[None, self.z_f_1.shape[1]])
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.x_f_2.shape[1]])
        self.y2 = tf.placeholder(tf.float32, shape=[None, self.y_f_2.shape[1]])
        self.z2 = tf.placeholder(tf.float32, shape=[None, self.z_f_2.shape[1]])
        self.xt = tf.placeholder(tf.float32, shape=[None, self.x_t.shape[1]])
        self.yt = tf.placeholder(tf.float32, shape=[None, self.y_t.shape[1]])
        self.zt = tf.placeholder(tf.float32, shape=[None, self.z_t.shape[1]])


        self.pred = self.u_pred(self.xt, self.yt, self.zt)
        self.u_b2 = self.net_u(self.xb, self.yb, self.zb) + self.net_u_2(self.xb, self.yb, self.zb)
        self.f_1, self.f_2 = self.net_f(self.x1, self.y1, self.z1, self.x2, self.y2, self.z2)
        self.u_t, self.u_dt = self.net_dt(self.xt, self.yt, self.zt)


        self.lossb_2 = tf.reduce_mean(tf.square(self.u_b2 - self.ub))
        self.lossv = tf.reduce_mean(tf.square(self.f_1)) + tf.reduce_mean(tf.square(self.f_2))
        self.loss_t = tf.reduce_mean(tf.square(self.u_t))
        self.loss_dt = tf.reduce_mean(tf.square(self.u_dt))
        self.loss = self.lossb_2 + self.lossv + self.loss_t + self.loss_dt

        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net_1(self, x, y, z, weights, biases):
        X = tf.concat([x, y, z], axis=1)
        H = X
        R = tf.add(tf.matmul(H, weights[2]), biases[2])

        H1 = 2*x**2 + 3*y**2 + 6*z**2 - 1.69
        H1 = tf.nn.softmax(tf.add(tf.matmul(H1, weights[0]), biases[0]))

        Q = tf.add(tf.matmul(H, weights[2]), biases[2])
        W = tf.add(tf.matmul(H, weights[4]), biases[4])
        V = tf.add(tf.matmul(H, weights[6]), biases[6])
        A = tf.multiply(tf.cos(tf.multiply(Q, W)), V)
        H = tf.cos(tf.add(tf.matmul(A, weights[7]), biases[7]))
        H = tf.multiply((1-H), H1) + tf.multiply(H, R)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_2(self, x, y, z, weights, biases):
        X = tf.concat([x, y, z], axis=1)
        H = X
        R = tf.add(tf.matmul(H, weights[2]), biases[2])  # The linear expansion layer is used for dimension matching.

        H1 = 2*x**2 + 3*y**2 + 6*z**2 - 1.69
        H1 = tf.nn.softmax(tf.add(tf.matmul(H1, weights[0]), biases[0]))

        Q = tf.add(tf.matmul(H, weights[2]), biases[2])
        W = tf.add(tf.matmul(H, weights[4]), biases[4])
        V = tf.add(tf.matmul(H, weights[6]), biases[6])
        A = tf.multiply(tf.sin(tf.multiply(Q, W)), V)
        H = tf.sin(tf.add(tf.matmul(A, weights[7]), biases[7]))

        H = tf.multiply((1 - H), H1) + tf.multiply(H, R)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y, z):
        u = self.neural_net(tf.concat([x, y, z], 1), self.weights_M, self.biases_M)
        return u

    def net_u_1(self, x, y, z):
        u = self.neural_net_1(x, y, z, self.weights_1, self.biases_1)
        return u

    def net_u_2(self, x, y, z):
        u = self.neural_net_2(x, y, z, self.weights_2, self.biases_2)
        return u

    def df1(self, x, y, z):
        u = tf.sin(2 * x) * tf.cos(2 * y) * tf.exp(z)
        u_y = tf.gradients(u, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_z = tf.gradients(u, z)[0]

        u_y = u_y * self.b(x, y, z)
        u_x = u_x * self.b(x, y, z)
        u_z = u_z * self.b(x, y, z)

        u_yy = tf.gradients(u_y, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_zz = tf.gradients(u_z, z)[0]
        return u_yy + u_xx + u_zz

    def df2(self, x, y, z):
        u = (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * ((y - x) / 3)) * tf.log(x + y + 3) * tf.cos(z)
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_z = tf.gradients(u, z)[0]
        u_zz = tf.gradients(u_z, z)[0]
        return u_yy + u_xx + u_zz

    def b(self,x ,y, z):
        return 10 * (1 + 0.2 * tf.cos(2*np.pi*(x+y))*tf.sin(2*np.pi * (x-y))*tf.cos(z))

    def net_f(self, x, y, z, x2, y2, z2):
        #界面内部
        u_1 = self.net_u(x, y, z) + self.net_u_1(x, y, z)
        u_x = tf.gradients(u_1, x)[0]
        u_y = tf.gradients(u_1, y)[0]
        u_z = tf.gradients(u_1, z)[0]

        u_x = u_x * self.b(x, y, z)
        u_y = u_y * self.b(x, y, z)
        u_z = u_z * self.b(x, y, z)

        u_yy = tf.gradients(u_y, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_zz = tf.gradients(u_z, z)[0]
        f_1 = u_yy + u_xx + u_zz - self.df1(x, y, z)

        #界面外部
        u_2 = self.net_u(x2, y2, z2) + self.net_u_2(x2, y2, z2)
        u_x2 = tf.gradients(u_2, x2)[0]
        u_y2 = tf.gradients(u_2, y2)[0]
        u_z2 = tf.gradients(u_2, z2)[0]

        u_yy2 = tf.gradients(u_y2, y2)[0]
        u_xx2 = tf.gradients(u_x2, x2)[0]
        u_zz2 = tf.gradients(u_z2, z2)[0]
        f_2 = u_yy2 + u_xx2 + u_zz2 - self.df2(x2, y2, z2)
        return f_1, f_2

    def net_dt(self, x, y, z):
        u_1 = tf.sin(2 * x) * tf.cos(2 * y) * tf.exp(z)
        u_2 = (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * ((y - x) / 3)) * tf.log(x + y + 3) * tf.cos(z)
        u1 = self.net_u_1(x, y, z)
        u2 = self.net_u_2(x, y, z)
        ut = (u1 - u2) - (u_1 - u_2)

        uy1 = tf.gradients(u_1, y)[0] * self.b(x, y, z)
        ux1 = tf.gradients(u_1, x)[0] * self.b(x, y, z)
        uz1 = tf.gradients(u_1, z)[0] * self.b(x, y, z)
        uy2 = tf.gradients(u_2, y)[0]
        ux2 = tf.gradients(u_2, x)[0]
        uz2 = tf.gradients(u_2, z)[0]

        u_y1 = tf.gradients(u1, y)[0] * self.b(x, y, z)
        u_x1 = tf.gradients(u1, x)[0] * self.b(x, y, z)
        u_z1 = tf.gradients(u1, z)[0] * self.b(x, y, z)
        u_y2 = tf.gradients(u2, y)[0]
        u_x2 = tf.gradients(u2, x)[0]
        u_z2 = tf.gradients(u2, z)[0]

        r = 2 * x ** 2 + 3 * y ** 2 + 6 * z ** 2

        nx = tf.gradients(r, x)[0]  # x方向法向量分量
        ny = tf.gradients(r, y)[0]  # y方向法向量分量
        nz = tf.gradients(r, z)[0]  # z方向法向量分量

        dx = nx / tf.sqrt(nx ** 2 + ny ** 2 + nz ** 2)  # x方向法向量分量
        dy = ny / tf.sqrt(nx ** 2 + ny ** 2 + nz ** 2)  # y方向法向量分量
        dz = nz / tf.sqrt(nx ** 2 + ny ** 2 + nz ** 2)  # z方向法向量分量

        u_dt = ((u_x1 * dx + u_y1 * dy + u_z1 * dz) - (u_x2 * dx + u_y2 * dy + u_z2 * dz)) - \
               ((ux1 * dx + uy1 * dy + uz1 * dz) - (ux2 * dx + uy2 * dy + uz2 * dz))

        return ut, u_dt

    def u_pred(self, x, y, z):
        u1 = self.net_u(x, y, z) + self.net_u_1(x, y, z)
        u2 = self.net_u(x, y, z) + self.net_u_2(x, y, z)
        u = tf.where(2 * x**2 + 3 * y**2 + 6 * z**2 - 1.69 < 0, u1, u2)
        return u

    def callback(self, loss):
        print('Loss:', loss)

    def train(self,nIter, tresh):

        tf_dict = {self.xb: self.x_b, self.yb: self.y_b, self.zb: self.z_b, self.ub: self.u_b, self.x1: self.x_f_1, self.y1: self.y_f_1, self.z1: self.z_f_1,
                   self.x2: self.x_f_2, self.y2: self.y_f_2, self.z2: self.z_f_2, self.xt: self.x_t, self.yt: self.y_t, self.zt: self.z_t}

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                if loss_value < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    break

            if it % 100 == 0:
                print(f"Step {it}, loss_value: {loss_value}")
                print()

    def predict(self, X1):
        u1 = self.sess.run(self.pred, {self.xt: X1[:, 0:1], self.yt: X1[:, 1:2], self.zt: X1[:, 2:3]})
        return u1


if __name__ == "__main__":
    LR = 0.001
    Opt_Niter = 50000 + 1
    Opt_tresh = 2e-32
    N = 32
    layerM = [3] + [20] * 3 + [1]
    layers1 = [1] + [N] + [3] + [N] + [3] + [N] + [3] + [N] + [N] * 2 + [1]
    layers2 = [1] + [N] + [3] + [N] + [3] + [N] + [3] + [N] + [N] * 2 + [1]

    def u_ext(x, y, z):
        u1 = np.sin(2 * x) * np.cos(2 * y) * np.exp(z)
        u2 = (16 * ((y - x)/3) ** 5 - 20 * ((y - x)/3) ** 3 + 5 * ((y - x)/3)) * np.log(x + y + 3) * np.cos(z)
        t = 2 * x ** 2 + 3 * y ** 2 + 6 * z ** 2 - 1.69
        u = np.where(t < 0, u1, u2)
        return u

#界面内外区域配置点
    def z(n: int):
        m = 2 * lhs(3, n) - 1
        x = m[:, 0][:, None]
        y = m[:, 1][:, None]
        z = m[:, 2][:, None]
        # 花瓣内配置点
        x1_x = []
        x1_y = []
        x1_z = []
        # 花瓣外配置点
        x2_x = []
        x2_y = []
        x2_z = []
        for i in range(n):
            if 2*x[i]**2 + 3*y[i]**2 + 6*z[i]**2 - 1.69 < 0:
                x1_x.append(x[i])
                x1_y.append(y[i])
                x1_z.append(z[i])
            else:
                x2_x.append(x[i])
                x2_y.append(y[i])
                x2_z.append(z[i])
        x1_x = np.asarray(x1_x)
        x1_y = np.asarray(x1_y)
        x1_z = np.asarray(x1_z)
        x2_x = np.asarray(x2_x)
        x2_y = np.asarray(x2_y)
        x2_z = np.asarray(x2_z)
        x1_t = np.concatenate([np.concatenate([x1_x, x1_y], axis=1), x1_z], axis=1)
        x2_t = np.concatenate([np.concatenate([x2_x, x2_y], axis=1), x2_z], axis=1)
        return x1_t, x2_t

    #界面
    def r_xy(v, u):
        a = np.sqrt(1.69 / 2)  # x轴半长
        b = np.sqrt(1.69 / 3)  # y轴半长
        c = np.sqrt(1.69 / 6)  # z轴半长
        x = a * np.sin(v) * np.cos(u)
        y = b * np.sin(v) * np.sin(u)
        z = c * np.cos(v)
        return x, y, z

    # 外边界配置点
    Nf1 = 100
    X_u_train1 = 2 * lhs(2, Nf1)-1
    X_u_train1 = np.asarray(X_u_train1)
    t1 = np.ones([X_u_train1.shape[0]])[:, None]
    X_u_traine1 = np.concatenate([t1, X_u_train1[:, 0][:, None], X_u_train1[:, 1][:, None]], axis=1)
    X_u_traine2 = np.concatenate([-t1, X_u_train1[:, 0][:, None], X_u_train1[:, 1][:, None]], axis=1)
    X_u_traine3 = np.concatenate([X_u_train1[:, 0][:, None], t1, X_u_train1[:, 1][:, None]], axis=1)
    X_u_traine4 = np.concatenate([X_u_train1[:, 0][:, None], -t1, X_u_train1[:, 1][:, None]], axis=1)
    X_u_traine5 = np.concatenate([X_u_train1[:, 0][:, None], X_u_train1[:, 1][:, None], t1], axis=1)
    X_u_traine6 = np.concatenate([X_u_train1[:, 0][:, None], X_u_train1[:, 1][:, None], -t1], axis=1)
    X_u_train = np.concatenate([X_u_traine1, X_u_traine2, X_u_traine3, X_u_traine4, X_u_traine5, X_u_traine6], axis=0)

    u = np.linspace(0, 2 * np.pi, 20)  # 角度 u
    v = np.linspace(0, np.pi, 20)
    u, v = np.meshgrid(u, v)
    xt, yt, zt = r_xy(v, u)
    xt, yt, zt = np.reshape(xt, [400, 1]), np.reshape(yt, [400, 1]), np.reshape(zt, [400, 1])
    xs = np.concatenate([xt, yt, zt], axis=1)

    # 外边界精确值
    u_train = u_ext(X_u_train[:, 0][:, None], X_u_train[:, 1][:, None], X_u_train[:, 2][:, None])

    #测试集
    delta_test = 0.05
    xbtest = np.arange(-1, 1 + delta_test, delta_test)
    # 将抽取到的测试样本点由精确解计算，再组成样本值对，挨个放在一起
    data_temp = np.asarray(
        [[xbtest[j], xbtest[i], xbtest[k]] for k in range(len(xbtest)) for i in range(len(xbtest)) for j in
         range(len(xbtest))])
    # 将X和U打平并赋值
    XB1_test = data_temp.flatten()[0::3]  # flatten(n)在第N维度展开,
    XB2_test = data_temp.flatten()[1::3]
    XB3_test = data_temp.flatten()[2::3]
    X1_test = XB2_test[:, None]
    X2_test = XB3_test[:, None]
    X3_test = XB1_test[:, None]
    u_test = u_ext(X1_test, X2_test, X3_test)
    X_test = np.concatenate([X1_test, X2_test, X3_test], axis=1)
    # 训练点
    x1_train, x2_train = z(500)

    model = PhysicsInformedNN(X_u_train, u_train, xs, x1_train, x2_train, layerM, layers1, layers2)
    start_time = time.time()
    model.train(Opt_Niter, Opt_tresh)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    u_pred = model.predict(X_test)
    RE = np.linalg.norm(u_test - u_pred) / (np.linalg.norm(u_test))
    print('Relative error:', RE)
    # np.savetxt("3D_ER", ER)


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

