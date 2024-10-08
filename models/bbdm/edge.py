import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 用于生成高斯核的函数
def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    x = np.linspace(-sigma, sigma, kernel_size)
    y = np.linspace(-sigma, sigma, kernel_size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return torch.from_numpy(kernel).float()

# 用于计算图像梯度的函数
def compute_gradients(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = torch.atan2(grad_y, grad_x)
    return gradient_magnitude, gradient_direction

# 非最大值抑制函数
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    grad_magnitude = gradient_magnitude.squeeze(0).squeeze(0).numpy()
    grad_direction = gradient_direction.squeeze(0).squeeze(0).numpy()
    grad_direction = grad_direction * 180.0 / np.pi
    grad_direction[grad_direction < 0] += 180

    suppressed_img = np.zeros(grad_magnitude.shape)
    for i in range(1, grad_magnitude.shape[0] - 1):
        for j in range(1, grad_magnitude.shape[1] - 1):
            q = 255
            r = 255
            angle = grad_direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = grad_magnitude[i, j + 1]
                r = grad_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = grad_magnitude[i + 1, j - 1]
                r = grad_magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = grad_magnitude[i + 1, j]
                r = grad_magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = grad_magnitude[i - 1, j - 1]
                r = grad_magnitude[i + 1, j + 1]

            if grad_magnitude[i, j] >= q and grad_magnitude[i, j] >= r:
                suppressed_img[i, j] = grad_magnitude[i, j]
            else:
                suppressed_img[i, j] = 0

    return torch.from_numpy(suppressed_img).unsqueeze(0).unsqueeze(0)

# 双阈值和边缘连接
def threshold_and_hysteresis(suppressed_img, low_threshold, high_threshold):
    strong_edges = (suppressed_img > high_threshold).float()
    weak_edges = ((suppressed_img >= low_threshold) & (suppressed_img <= high_threshold)).float()

    final_edges = strong_edges.clone()
    M, N = final_edges.shape[2], final_edges.shape[3]
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak_edges[0, 0, i, j] == 1:
                if (strong_edges[0, 0, i+1, j-1:j+2].sum() > 0 or
                    strong_edges[0, 0, i-1, j-1:j+2].sum() > 0 or
                    strong_edges[0, 0, i, [j-1, j+1]].sum() > 0):
                    final_edges[0, 0, i, j] = 1
                else:
                    final_edges[0, 0, i, j] = 0
    return final_edges

# Canny边缘检测主函数
def canny_edge_detection(img, low_threshold=0.1, high_threshold=0.2, kernel_size=5, sigma=1.0):
    img = img.unsqueeze(0).unsqueeze(0).float() / 255.0
    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    blurred_img = F.conv2d(img, gaussian_kernel, padding=kernel_size//2)
    gradient_magnitude, gradient_direction = compute_gradients(blurred_img)
    suppressed_img = non_maximum_suppression(gradient_magnitude, gradient_direction)
    edges = threshold_and_hysteresis(suppressed_img, low_threshold, high_threshold)
    return edges


class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)


if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('/data/data/AIGC/02/02_preprocessed/input/test_png/20231007002761__5__t2_tse_tra_384_p2 DNE__9.png', cv2.IMREAD_GRAYSCALE)
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    grad_layer = GradLayer()
    edge = grad_layer(img_tensor)
    edge = edge.numpy()[0, 0]
    plt.imshow(edge, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 应用Canny边缘检测
    # edges = canny_edge_detection(img_tensor, low_threshold=0.05, high_threshold=0.2)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges.squeeze().numpy(), cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.show()
