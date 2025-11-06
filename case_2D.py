import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_b_train, u_train, x1_train, x2_train, xt, layerM, layers1, layers2, X_test, u_test):
        #边界点及其数据
        self.x_b = X_b_train[:, 0:1]
        self.y_b = X_b_train[:, 1:2]
        self.u_b = u_train
        #界面内部配置点
        self.x_f_1 = x1_train[:, 0:1]
        self.y_f_1 = x1_train[:, 1:2]
        # 界面外部配置点
        self.x_f_2 = x2_train[:, 0:1]
        self.y_f_2 = x2_train[:, 1:2]
        # 界面配置点
        self.x_t = xt[:, 0:1]
        self.y_t = xt[:, 1:2]
        #网络
        self.layers_M = layerM
        self.layers_1 = layers1
        self.layers_2 = layers2

        self.x_te = X_test[:, 0:1]
        self.y_te = X_test[:, 1:2]
        self.u_te = u_test

        # Initialize NNs
        self.weights_M, self.biases_M = self.initialize_NN(layerM)
        self.weights_1, self.biases_1 = self.initialize_NN(layers1)
        self.weights_2, self.biases_2 = self.initialize_NN(layers2)


        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.xb = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]])
        self.yb = tf.placeholder(tf.float32, shape=[None, self.y_b.shape[1]])
        self.ub = tf.placeholder(tf.float32, shape=[None, self.u_b.shape[1]])
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.x_f_1.shape[1]])
        self.y1 = tf.placeholder(tf.float32, shape=[None, self.y_f_1.shape[1]])
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.x_f_2.shape[1]])
        self.y2 = tf.placeholder(tf.float32, shape=[None, self.y_f_2.shape[1]])
        self.xt = tf.placeholder(tf.float32, shape=[None, self.x_t.shape[1]])
        self.yt = tf.placeholder(tf.float32, shape=[None, self.y_t.shape[1]])
        self.xte = tf.placeholder(tf.float32, shape=[None, self.x_te.shape[1]])
        self.yte = tf.placeholder(tf.float32, shape=[None, self.y_te.shape[1]])
        self.ute = tf.placeholder(tf.float32, shape=[None, self.u_te.shape[1]])

        self.ET = tf.linalg.norm(
            (self.u_pred(self.xte, self.yte)) - self.ute) / tf.linalg.norm(
            tf.abs(self.ute))

        self.pred = self.u_pred(self.xt, self.yt)
        self.u_b1 = self.net_u(self.xt, self.yt) + self.net_u_1(self.xt, self.yt)
        self.u_b2 = self.net_u(self.xb, self.yb) + self.net_u_2(self.xb, self.yb)
        self.f_1, self.f_2 = self.net_f(self.x1, self.y1, self.x2, self.y2)
        self.u_t, self.u_dt = self.net_dt(self.xt, self.yt)

        self.lossb = tf.reduce_mean(tf.square(self.u_b2 - self.ub))
        self.lossv = tf.reduce_mean(tf.square(self.f_1)) + tf.reduce_mean(tf.square(self.f_2))
        self.loss = self.lossb + self.lossv + tf.reduce_mean(tf.square(self.u_t)) + tf.reduce_mean(tf.square(self.u_dt))

        self.LR = LR
        self.global_step = tf.Variable(0, trainable=False)

        # 或者使用余弦衰减（取消注释使用）
        self.learning_rate = tf.train.cosine_decay(
            learning_rate=self.LR,
            global_step=self.global_step,
            decay_steps=Opt_Niter  # 总训练步数
        )

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)

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


    def neural_net_1(self, x, y, weights, biases):
        X = tf.concat([x, y], axis=1)
        H = X
        H1 = tf.sqrt(x**2 + y**2) - 0.3
        R = tf.add(tf.matmul(H, weights[2]), biases[2]) # The linear expansion layer is used for dimension matching.
        H1 = tf.cos(tf.add(tf.matmul(H1, weights[0]), biases[0]))

        Q = tf.add(tf.matmul(H, weights[2]), biases[2])
        W = tf.add(tf.matmul(H, weights[4]), biases[4])
        V = tf.add(tf.matmul(H, weights[6]), biases[6])
        A = tf.multiply(tf.cos(tf.multiply(Q, W)), V)
        H = tf.cos(tf.add(tf.matmul(A, weights[7]), biases[7]))
        H2 = tf.multiply((1 - H), H1) + tf.multiply(H, R)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H2, W), b)
        return Y

    def neural_net_2(self, x, y, weights, biases):
        X = tf.concat([x, y], axis=1)
        H = X
        R = tf.add(tf.matmul(H, weights[2]), biases[2]) # The linear expansion layer is used for dimension matching.
        H1 = tf.sqrt(x**2 + y**2) - 0.3
        H1 = tf.sin(tf.add(tf.matmul(H1, weights[0]), biases[0]))

        Q = tf.add(tf.matmul(H, weights[2]), biases[2])
        W = tf.add(tf.matmul(H, weights[4]), biases[4])
        V = tf.add(tf.matmul(H, weights[6]), biases[6])
        A = tf.multiply(tf.sin(tf.multiply(Q, W)), V)
        H = tf.sin(tf.add(tf.matmul(A, weights[7]), biases[7]))
        H2 = tf.multiply((1 - H), H1) + tf.multiply(H, R)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H2, W), b)
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

    def net_u(self, x, y):
        u = self.neural_net(tf.concat([x, y], 1), self.weights_M, self.biases_M)
        return u

    def net_u_1(self, x, y):
        u = self.neural_net_1(x, y, self.weights_1, self.biases_1)
        return u

    def net_u_2(self, x, y):
        u = self.neural_net_2(x, y, self.weights_2, self.biases_2)
        return u

    def df(self, x, y):
        u = 2 * x**2 + 3 * y**2
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return u_yy + u_xx

    def df2(self, x, y):
        u = tf.sin(x+y) + tf.cos(x+y) + 1
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return u_yy + u_xx

    def net_f(self, x, y, x2, y2):
        #界面内部
        u_1 = self.net_u(x, y) + self.net_u_1(x, y)
        u_y = tf.gradients(u_1, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_x = tf.gradients(u_1, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_1 = (u_yy + u_xx) - self.df(x, y)
        #界面外部
        u_2 = self.net_u(x2, y2) + self.net_u_2(x2, y2)
        u_y2 = tf.gradients(u_2, y2)[0]
        u_yy2 = tf.gradients(u_y2, y2)[0]
        u_x2 = tf.gradients(u_2, x2)[0]
        u_xx2 = tf.gradients(u_x2, x2)[0]
        f_2 = (u_yy2 + u_xx2) - self.df2(x2, y2)
        return f_1, f_2

    def net_dt(self, x, y):
        e1 = 2*x**2 + 3*y**2
        e2 = tf.sin(x +y) + tf.cos(x+y) + 1
        r = tf.sqrt(x**2 + y**2) - 0.3

        nx = tf.gradients(r, x)[0]  # x方向法向量分量
        ny = tf.gradients(r, y)[0]  # y方向法向量分量

        dx = nx / tf.sqrt(nx ** 2 + ny ** 2)  # x方向法向量分量
        dy = ny / tf.sqrt(nx ** 2 + ny ** 2)  # y方向法向量分量

        u1 = self.net_u_1(x, y)
        u2 = self.net_u_2(x, y)
        ut = (u1 - u2) - (e1 - e2)

        u_y1 = tf.gradients(u1, y)[0]
        u_x1 = tf.gradients(u1, x)[0]
        u_y2 = tf.gradients(u2, y)[0]
        u_x2 = tf.gradients(u2, x)[0]

        e_y1 = tf.gradients(e1, y)[0]
        e_x1 = tf.gradients(e1, x)[0]
        e_y2 = tf.gradients(e2, y)[0]
        e_x2 = tf.gradients(e2, x)[0]

        u_dt = ((u_x1 * dx + u_y1 * dy) - (u_x2 * dx + u_y2 * dy)) - ((e_x1 * dx + e_y1 * dy) - (e_x2 * dx + e_y2 * dy))

        return ut, u_dt
    def u_pred(self, x, y):
        u1 = self.net_u(x, y) + self.net_u_1(x, y)
        u2 = self.net_u(x, y) + self.net_u_2(x, y)
        u = tf.where(tf.sqrt(x ** 2 + y ** 2) < 0.3, u1, u2)
        return u

    def u_1(self, x, y):
        u = self.net_u_1(x, y)
        return u

    def u_2(self, x, y):
        u = self.net_u_2(x, y)
        return u

    def u_m(self, x, y):
        u = self.net_u(x, y)
        return u

    def callback(self, loss):
        print('Loss:', loss)

    def train(self,nIter, tresh):

        tf_dict = {self.xb: self.x_b, self.yb: self.y_b, self.ub: self.u_b, self.x1: self.x_f_1, self.y1: self.y_f_1,
                   self.x2: self.x_f_2, self.y2: self.y_f_2, self.xt: self.x_t, self.yt: self.y_t, self.xte: self.x_te, self.yte: self.y_te, self.ute: self.u_te}

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                UT = self.sess.run(self.ET, tf_dict)
                if UT < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    break

            if it % 100 == 0:
                print(f"Step {it}, Relative error: {UT}, loss_value: {loss_value}")
                print()

    def predict(self, X1):
        u1 = self.sess.run(self.pred, {self.xt: X1[:, 0:1], self.yt: X1[:, 1:2]})
        return u1

if __name__ == "__main__":
    LR = 0.001
    Opt_Niter = 50000 + 1
    Opt_tresh = 1e-5
    N = 32
    layerM = [2] + [20] * 3 + [1]
    layers1 = [1] + [N] + [2] + [N] + [2] + [N] + [2] + [N] + [N] * 2 + [1]
    layers2 = [1] + [N] + [2] + [N] + [2] + [N] + [2] + [N] + [N] * 2 + [1]

    def u_ext(x, y):
        u1 = 2 * x ** 2 + 3 * y ** 2
        u2 = np.sin(x+y) + np.cos(x+y) + 1
        utemp = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, u1, u2)
        return utemp

#界面内外区域配置点
    def z(n: int):
        m = 2 * lhs(2, n) - 1
        x = m[:, 0][:, None]
        y = m[:, 1][:, None]
        # 花瓣内配置点
        x1_x = []
        x1_y = []
        # 花瓣外配置点
        x2_x = []
        x2_y = []
        for i in range(n):
            if np.sqrt(x[i] ** 2 + y[i] ** 2) < 0.3:
                x1_x.append(x[i])
                x1_y.append(y[i])
            else:
                x2_x.append(x[i])
                x2_y.append(y[i])
        x1_x = np.asarray(x1_x)
        x1_y = np.asarray(x1_y)
        x2_x = np.asarray(x2_x)
        x2_y = np.asarray(x2_y)
        x1_t = np.concatenate([x1_x, x1_y], axis=1)
        x2_t = np.concatenate([x2_x, x2_y], axis=1)
        return x1_t, x2_t

#界面曲线函数
    def x(t):
        return 0.3 * np.cos(t)

    def y(t):
        return 0.3 * np.sin(t)

    Nf1 = 100
    X_u_train1 = lhs(1, Nf1)
    X_u_train1 = np.asarray(X_u_train1)
    t1 = np.ones([X_u_train1.shape[0]])[:, None]
    X_u_traine1 = np.concatenate([2 * X_u_train1 - 1, t1], axis=1)
    X_u_traine2 = np.concatenate([2 * X_u_train1 - 1, -1*t1], axis=1)
    X_u_traine3 = np.concatenate([t1, 2 * X_u_train1 - 1], axis=1)
    X_u_traine4 = np.concatenate([-1*t1, 2 * X_u_train1 - 1], axis=1)
    u_b1 = np.concatenate([X_u_traine1, X_u_traine2], axis=0)
    u_b2 = np.concatenate([X_u_traine3, X_u_traine4], axis=0)
    X_b_train = np.concatenate([u_b1, u_b2], axis=0)
    u_train = u_ext(X_b_train[:, 0][:, None], X_b_train[:, 1][:, None])

    delta_test = 0.007
    xbtest = np.arange(-1, 1 + delta_test, delta_test)
    data_temp = np.asarray([[xbtest[j], xbtest[i]] for i in range(len(xbtest)) for j in range(len(xbtest))])
    XB1_test = data_temp.flatten()[0::2]
    XB2_test = data_temp.flatten()[1::2]
    XB2_test = np.flipud(XB2_test)
    X1_test = XB1_test[:, None]
    X2_test = XB2_test[:, None]
    u_test = u_ext(X1_test, X2_test)
    X_test = np.concatenate([X1_test, X2_test], axis=1)

    x1_train, x2_train = z(500)
    delta_test = 0.001
    xbtest = np.arange(-np.pi, np.pi + delta_test, delta_test)
    x = x(xbtest)[:, None]
    y = y(xbtest)[:, None]
    xt = np.concatenate([x, y], axis=1)

    model = PhysicsInformedNN(X_b_train, u_train, x1_train, x2_train, xt, layerM, layers1, layers2, X_test, u_test)
    start_time = time.time()
    model.train(Opt_Niter, Opt_tresh)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    u_pred = model.predict(X_test)
    print(np.linalg.norm(u_test - u_pred) / (np.linalg.norm(u_test)))
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    fig, ax = plt.subplots()
    data = np.reshape(np.abs(u_test - u_pred), (287, 287))
    plt.xticks(np.arange(0, 287, 70), np.arange(-1, 1 + delta_test, 0.5))
    plt.xlabel('$x$', fontsize=15)
    # plt.axvline(x=100, linewidth=1,color='w', ls='-')
    t = np.arange(-1, 1 + delta_test, 0.5)
    t = np.flipud(t)
    plt.yticks(np.arange(0, 287, 70), t)
    plt.ylabel('$t$', fontsize=15)
    plt.imshow(data, cmap=plt.cm.jet)
    plt.colorbar()  # 显示颜色条
    plt.title('IG-PINNs', fontsize=25, color='#0033cc')
    # plt.savefig('2DResults/error.pdf')
    plt.show()

