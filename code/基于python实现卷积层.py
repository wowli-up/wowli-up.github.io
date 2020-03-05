import numpy as np


def im2col2(input_data, fh, fw, stride=1, pad=1):
    '''
     input_data--输入数据，shape为(batch_size,Channel,Height,Width)
     fh -- 滤波器的height 3
     fw --滤波器的width 3
     stride -- 步幅 1
     pad -- 填充 1
     Returns :
     col -- 输入数据根据滤波器、步幅等展开的二维数组，每一行代表一条卷积数据
    '''
    N, C, H, W = input_data.shape
    "[20,1,28,28]"

    out_h = (H + 2 * pad - fh) // stride + 1

    "[28]"
    out_w = (W + 2 * pad - fw) // stride + 1
    "[28]"
    print("out_h={0},out_w{1}".format(out_h,out_w))
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    print(img.shape)
    "[30*30*1]"

    col = np.zeros((N, out_h, out_w, fh * fw * C))

    "fh * fw * C 负责存储每次参与卷积的参数"
    print(col.shape)
    # 将所有维度上需要卷积的值展开成一行（列）,卷积次数为out_h*out_w*c,每次卷积内含参数量为（fh*fw*c）
    for y in range(out_h):
        y_start = y * stride
        y_end = y_start + fh
        for x in range(out_w):
            x_start = x * stride
            x_end = x_start + fw
            col[:, y, x] = img[:, :, y_start:y_end, x_start:x_end].reshape(N, -1)

    col = col.reshape(N * out_h * out_w, -1)
    return col

def col2im2(col, out_shape, fh, fw, stride=1, pad=0):
    '''

     col: 二维数组
     out_shape-- 输出的shape，shape为(Number of example,Channel,Height,Width)
     fh -- 滤波器的height
     fw --滤波器的width
     stride -- 步幅
     pad -- 填充

     Returns :
     img -- 将col转换成的img ，shape为out_shape
    '''
    N, C, H, W = out_shape

    col_m, col_n = col.shape

    out_h = (H + 2 * pad - fh) // stride + 1

    out_w = (W + 2 * pad - fw) // stride + 1

    print("out_h,out_w",out_h,out_w)
    img = np.zeros((N, C, H, W))
    #img = np.pad(img,[(0,0),(0,0),(pad,pad),(pad,pad)],"constant")

    # 将col转换成一个filter
    for c in range(C):
        for y in range(out_h):
            for x in range(out_w):
                col_index = (c * out_h * out_w) + y * out_w + x
                ih = y * stride
                iw = x * stride
                img[:, c, ih:ih + fh, iw:iw + fw] = col[col_index].reshape((fh, fw))
    return img
class Convolution:
    def __init__(self, W, fb, stride=1, pad=1):
        """
        W-- 滤波器权重，shape为(FN,NC,FH,FW),FN 为滤波器的个数
        fb -- 滤波器的偏置，shape 为(1,FN)
        stride -- 步长
        pad -- 填充个数
        """
        self.W = W
        self.fb = fb
        self.stride = stride
        self.pad = pad

        self.col_X = None
        self.X = None
        self.col_W = None

        self.dW = None
        self.db = None
        self.out_shape = None

    #    self.out = None

    def forward(self, input_X):
        """
        input_X-- shape为(m,nc,height,width)
        """
        self.X = input_X
        FN, NC, FH, FW = self.W.shape

        m, input_nc, input_h, input_w = self.X.shape


       # 将输入数据展开成二维数组，shape为（m*out_h*out_w,FH*FW*C)
        self.col_X = col_X = im2col2(self.X, FH, FW, self.stride, self.pad)
        print("self.col_X.shape",self.col_X .shape)
        # 将滤波器一个个按列展开(FH*FW*C,FN)       col_W.shape 15680,9 col_w 9,20  输出 15680，20
        self.col_W = col_W = self.W.reshape(FN, -1).T
        out = np.dot(col_X, col_W) + self.fb

        out = out.T #     20，15680

        out = out.reshape(m, FN, input_h, input_w)
        self.out_shape = out.shape
        print("out.shape", out.shape)
        return out #(20, 20, 28, 28)

    def backward(self, dz, learning_rate):
        print("==== Conv backbward ==== ")
        assert (dz.shape == self.out_shape)

        FN, NC, FH, FW = self.W.shape #[20,1,28,28]
        o_FN, o_NC, o_FH, o_FW = self.out_shape #[20,20,28,28]

        print("o_FN = {0}, o_NC = {1}, o_FH = {2}, o_FW = {3} ".format(o_FN,o_NC,o_FH,o_FW))

        col_dz = dz.reshape(o_NC, -1)  #col_dz  [20,15680]   dz[20, 20, 28, 28]

        col_dz = col_dz.T

        print("self.col_X.T,col_dz",self.col_X.shape,col_dz.shape)
        self.dW = np.dot(self.col_X.T, col_dz)  # [15680,9]  [15680,20]


        self.db = np.sum(col_dz, axis=0, keepdims=True)

        self.dW = self.dW.T.reshape(self.W.shape)
        self.db = self.db.reshape(self.fb.shape)
        print("dw.shape = {0},db.shape = {1} ,self.col_W={2}".format(self.dW.shape, self.db.shape,self.col_W.shape))
        d_col_x = np.dot(col_dz, self.col_W.T)  # shape is (m*out_h*out_w,FH,FW*C)
        print("d_col_x.shape= ", d_col_x.shape)
        dx = col2im2(d_col_x, self.X.shape, FH, FW, stride=1)
        print("dx.shape= ",dx.shape)
        assert (dx.shape == self.X.shape)

        # 更新W和b
        self.W = self.W - learning_rate * self.dW
        self.fb = self.fb - learning_rate * self.db

        return dx

w=np.random.uniform(0,1,(20,1,3,3))
b=x=np.random.uniform(0,1,(1,20))
x=np.random.uniform(0,255,(20,1,28,28))
dz=np.random.uniform(0,255,(20,20,28,28))

test = Convolution(w,b)
test.forward(x)
test.backward(dz,0.01)