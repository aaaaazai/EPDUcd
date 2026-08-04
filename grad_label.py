# import os
# import numpy as np
# from PIL import Image, ImageFilter
# from tqdm import tqdm
#
#
# def generate_edge_maps():
#     # 基础路径配置
#     base_dir = './datasets/TempLevircd_FC'
#     modes = ['test', 'train', 'val']
#
#     for mode in modes:
#         label_dir = os.path.join(base_dir, mode, 'label')
#         output_dir = os.path.join(base_dir, mode, 'grad_label')
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 获取当前分组的标签文件
#         label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]
#
#         for filename in tqdm(label_files, desc=f'处理 {mode} 数据集'):
#             img_path = os.path.join(label_dir, filename)
#             label = Image.open(img_path).convert('L')
#
#             # 保持原有的边缘检测处理
#             sobel_x = label.filter(ImageFilter.FIND_EDGES)
#             sobel_y = label.filter(ImageFilter.Kernel((3, 3), (-1, 0, 1, -2, 0, 2, -1, 0, 1), 1, 0))
#             edge_map = Image.blend(sobel_x, sobel_y, alpha=0.5)
#
#             output_path = os.path.join(output_dir, filename)
#             edge_map.save(output_path)
#
#
# if __name__ == '__main__':
#     generate_edge_maps()
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_edge_maps():
    # 基础路径配置
    base_dir = './datasets/LEVIRCD_FC'
    modes = ['test', 'train', 'val']

    for mode in modes:
        label_dir = os.path.join(base_dir, mode, 'label')
        output_dir = os.path.join(base_dir, mode, 'grad_label')
        os.makedirs(output_dir, exist_ok=True)

        # 获取当前分组的标签文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]

        for filename in tqdm(label_files, desc=f'处理 {mode} 数据集'):
            img_path = os.path.join(label_dir, filename)

            try:
                # 使用OpenCV读取图像（灰度模式）
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告：无法读取图像 {img_path}，跳过该文件")
                    continue

                # ========== 核心修改：替换为拉普拉斯边缘检测 ==========
                # 1. 先进行高斯模糊降噪（避免拉普拉斯放大噪声）
                blurred = cv2.GaussianBlur(img, (3, 3), 0)
                # 2. 应用拉普拉斯算子（ksize=3表示3x3核）
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
                # 3. 转换为8位无符号整数（拉普拉斯会输出正负值，需要归一化）
                laplacian_8u = cv2.convertScaleAbs(laplacian)
                # 4. 可选：二值化增强边缘（根据需求调整阈值，这里用127作为分界）
                _, edge_map = cv2.threshold(laplacian_8u, 127, 255, cv2.THRESH_BINARY)

                # 转换回PIL Image以便保存
                edge_pil = Image.fromarray(edge_map)

                output_path = os.path.join(output_dir, filename)
                edge_pil.save(output_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错：{str(e)}")
                continue

if __name__ == '__main__':
    generate_edge_maps()

'''
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def generate_edge_maps():
    # 基础路径配置
    base_dir = './datasets/LEVIRCD_FC'
    modes = ['test', 'train', 'val']

    for mode in modes:
        label_dir = os.path.join(base_dir, mode, 'label')
        output_dir = os.path.join(base_dir, mode, 'grad_label')
        os.makedirs(output_dir, exist_ok=True)

        # 获取当前分组的标签文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]

        for filename in tqdm(label_files, desc=f'处理 {mode} 数据集'):
            img_path = os.path.join(label_dir, filename)

            # 使用OpenCV读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 应用Canny边缘检测
            # 调整阈值参数以获得最佳效果
            edge_map = cv2.Canny(img, 50, 150)

            # 转换回PIL Image以便保存
            edge_pil = Image.fromarray(edge_map)

            output_path = os.path.join(output_dir, filename)
            edge_pil.save(output_path)


if __name__ == '__main__':
    generate_edge_maps()
'''

'''
import os
import numpy as np
import cv2
from PIL import Image


def check_label_and_grad_values():
    # 1. 配置路径（根据实际数据集路径调整）
    base_dir = './datasets/CLCD_256_1440'  # 数据集根目录
    mode = 'train'  # 可切换为 'test' 或 'val'
    label_dir = os.path.join(base_dir, mode, 'label')  # 标签目录（0/1二值标签）
    grad_label_dir = os.path.join(base_dir, mode, 'grad_label')  # 边缘标签目录（Canny检测结果）

    # 2. 获取样本文件名（取第一个文件作为示例）
    label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]
    if not label_files:
        print("未找到标签文件！")
        return
    sample_filename = label_files[0]  # 取第一个文件

    # 3. 读取label和grad_label图像
    label_path = os.path.join(label_dir, sample_filename)
    grad_label_path = os.path.join(grad_label_dir, sample_filename)

    # 读取图像（转为numpy数组查看数值）
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 标签图像（0/1）
    grad_label_img = cv2.imread(grad_label_path, cv2.IMREAD_GRAYSCALE)  # 边缘图像（0/255）

    # 4. 打印数值统计信息
    print(f"=== 样本文件: {sample_filename} ===")

    # 标签图像（label）统计
    print("\n【label图像数值信息】")
    print(f"形状: {label_img.shape}")
    print(f"数据类型: {label_img.dtype}")
    print(f"最小值: {np.min(label_img)}")
    print(f"最大值: {np.max(label_img)}")
    print(f"唯一值: {np.unique(label_img)}")  # 二值标签应输出 [0 1]

    # 边缘图像（grad_label）统计
    print("\n【grad_label图像数值信息】")
    print(f"形状: {grad_label_img.shape}")
    print(f"数据类型: {grad_label_img.dtype}")
    print(f"最小值: {np.min(grad_label_img)}")
    print(f"最大值: {np.max(grad_label_img)}")
    print(f"唯一值: {np.unique(grad_label_img)}")  # Canny边缘检测结果通常为 [0 255]

    # 5. 可选：可视化图像（如需查看图像可取消注释）
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121), plt.imshow(label_img, cmap='gray'), plt.title('label (0/1)')
    # plt.subplot(122), plt.imshow(grad_label_img, cmap='gray'), plt.title('grad_label (edges)')
    # plt.show()


if __name__ == '__main__':
    check_label_and_grad_values()
'''