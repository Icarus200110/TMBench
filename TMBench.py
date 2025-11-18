import argparse
import os

from re import I
import time
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

import cv2

import matplotlib.pyplot as plt

from pandas.compat.pyarrow import pa

import TMBspatial_cell

from Wangshj import generate_simdata

from merge_fastq import merge_fastq_files



parser  = argparse.ArgumentParser(description='Generate filled matrix with different shapes or from an image.')

parser.add_argument('--rows', type=int, required=False, help='Number of rows in the 2D coordinate system.',default=50)

parser.add_argument('--cols', type=int, required=False, help='Number of columns in the 2D coordinate system.',default=50)

parser.add_argument('--shape', type=str, choices=['Circle','Square', 'image'], required=False,

                    help='Organization shape',default='image')

parser.add_argument('--image_path', type=str, help='Path to the image.',default='./data/tissue_lowres_image.png')

parser.add_argument('--out_path', type=str, help='Path to simulating data.',default=None)

# 添加barcode文件参数

parser.add_argument('--barcode_file', type=str, help='Path to the spatial barcode file.',default='spatial_barcodes_new.txt')

# 添加基因表达数据相关参数
parser.add_argument('--expression_matrix', type=str, default='./data/data.csv', help='Path to the gene expression matrix CSV file (format must match data/data.csv)')
parser.add_argument('--gene_annotation', type=str, default='./data/gene_annotation.txt', help='Path to the gene annotation file (format: gene_name,chromosome,start,end)')


parser.add_argument('--TMBspatial_snv', help='the snv, example:-snv onesnv,chr12,25398284,C,T,0.001', required=False, type=str)
parser.add_argument('--snv_mutation_mode', help='SNV mutation mode (boundary/gradient/nested)', required=False, type=str,choices=['boundary','gradient','nested'])
parser.add_argument('--snv_gradient_direction', help='SNV gradient direction (horizontal/vertical/radial)', required=False, type=str,choices=['horizontal','vertical','radial'])

parser.add_argument('--TMBspatial_cnv', help='the cnv', required=False, type=str)
parser.add_argument('--cnv_mutation_mode', help='CNV mutation mode (boundary/gradient/nested)', required=False, type=str,choices=['boundary','gradient','nested'])
parser.add_argument('--cnv_gradient_direction', help='CNV gradient direction (horizontal/vertical/radial)', required=False, type=str,choices=['horizontal','vertical','radial'])

parser.add_argument('--TMBspatial_bed', help='the bed file', required=False, default="./mybed.bed", type=str)

parser.add_argument('--noise_level', type=float, default=0, help='Noise level (0 for no noise)')
parser.add_argument('--noise_dispersion', type=float, default=1, help='Noise dispersion (larger values mean less noise)')
parser.add_argument('--spatial_corr', type=float, default=0, help='Spatial correlation strength (0 for no spatial correlation)')

# 兼容旧参数
parser.add_argument('--TMBspatial_mutation_mode', help='[Deprecated] Global mutation mode (use --snv_mutation_mode and --cnv_mutation_mode instead)', required=False, type=str,choices=['boundary','gradient','nested'])
parser.add_argument('--gradient_direction', help='[Deprecated] Global gradient direction (use --snv_gradient_direction and --cnv_gradient_direction instead)', required=False, type=str,choices=['horizontal','vertical','radial'])




args, unknown = parser.parse_known_args()





def generate_circle(rows, cols, shrink_ratio=0):

    """

    :param rows: 矩阵的行数

    :param cols: 矩阵的列数

    :param shrink_ratio: 缩小比例，默认为 0.15

    :return: 填充圆形矩阵

    """

    center_row = rows // 2

    center_col = cols // 2

    radius = min(center_row, center_col) * (1 - shrink_ratio)

    filled_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):

        for j in range(cols):

            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)

            if distance <= radius:

                filled_matrix[i, j] = 1

    return filled_matrix



def generate_square(rows, cols, shrink_ratio=0):

    """

    :param rows: 矩阵的行数

    :param cols: 矩阵的列数

    :param shrink_ratio: 缩小比例，默认为 0.15

    :return: 填充正方形矩阵

    """

    center_row = rows // 2

    center_col = cols // 2

    side_length = min(rows, cols) // 2 * (1 - shrink_ratio)

    filled_matrix = np.zeros((rows, cols), dtype=int)

    top = int(center_row - side_length)

    bottom = int(center_row + side_length)

    left = int(center_col - side_length)

    right = int(center_col + side_length)

    filled_matrix[top:bottom + 1, left:right + 1] = 1

    return filled_matrix



def generate_from_image(rows, cols, image_path, shrink_ratio=0):

    """

    :param rows: 矩阵的行数

    :param cols: 矩阵的列数

    :param image_path: 图片的路径

    :param shrink_ratio: 缩小比例，默认为 0.15

    :return: 填充后的矩阵

    """

    try:

        # 读取图片

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:

            raise ValueError("无法读取图片，请检查图片路径。")

        # 调整图片大小

        resized_image = cv2.resize(image, (cols, rows))

        # 使用Canny边缘检测

        edges = cv2.Canny(resized_image, 50, 150)


        # 进行形态学膨胀操作，连接断开的边缘

        kernel = np.ones((3, 3), np.uint8)

        dilated_edges = cv2.dilate(edges, kernel, iterations=1)


        # 查找轮廓

        contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 计算缩放后的轮廓

        M = cv2.moments(contours[0])

        if M["m00"] != 0:

            cX = int(M["m10"] / M["m00"])

            cY = int(M["m01"] / M["m00"])
        else:

            cX, cY = 0, 0


        scaled_contours = []

        # 确保contours不为空
        if len(contours) > 0:
            for contour in contours:
                # 创建新的轮廓点列表
                scaled_contour = []
                for point in contour:
                    x = point[0][0]
                    y = point[0][1]
                    # 计算缩放后的坐标
                    new_x = int(cX + (x - cX) * (1 - shrink_ratio))
                    new_y = int(cY + (y - cY) * (1 - shrink_ratio))
                    # 添加到新轮廓（保持OpenCV的格式）
                    scaled_contour.append([[new_x, new_y]])
                # 转换为numpy数组并指定int32类型，确保与OpenCV兼容
                scaled_contours.append(np.array(scaled_contour, dtype=np.int32))


        # 创建与OpenCV兼容的矩阵数据类型
        filled_matrix = np.zeros((rows, cols), dtype=np.uint8)
        # 确保数组是连续的
        filled_matrix = np.ascontiguousarray(filled_matrix)
        
        # 只有当有有效的轮廓时才调用drawContours
        if scaled_contours:
            try:
                cv2.drawContours(filled_matrix, scaled_contours, -1, 1, thickness=cv2.FILLED)
            except Exception as e:
                print(f"绘制轮廓时出错: {e}")
                # 出错时仍返回空矩阵，确保函数继续执行
                return filled_matrix
        
        return filled_matrix

    except Exception as e:

        print(f"读取图片时出错: {e}")

        return np.zeros((rows, cols), dtype=int)


def get_all_coordinates(matrix):

    """

    读取矩阵中值为 1 的所有坐标

    :param matrix: 输入的矩阵

    :return: 包含值为 1 的坐标的列表，每个坐标以 (行索引, 列索引) 的形式表示

    """

    coordinates = []

    rows, cols = matrix.shape

    for i in range(rows):

        for j in range(cols):

            if matrix[i, j] == 1:

                coordinates.append((i+1, j+1))
    return coordinates


def visualize_matrix(matrix, output_path=None, vmax=None, image_type='snv'):  # image_type: 'snv' or 'cnv'
    # 默认归一化
    norm = plt.Normalize()

    """

    可视化矩阵

    :param matrix: 矩阵数据

    :param output_path: 图像保存路径（None表示直接显示）

    :param vmax: 颜色映射的最大值（None表示使用矩阵最大值）

    """

    plt.figure(figsize=(10, 8))
    # 将矩阵转换为float类型numpy数组并处理非数值值
    # 确保矩阵是numpy数组并转换为float类型
    if not isinstance(matrix, np.ndarray):
        # 尝试将列表矩阵转换为数值数组
        matrix = np.array(matrix, dtype=np.float64)
    # 处理可能的非数值类型
    if matrix.dtype == object:
        # 强制转换对象类型数组为float
        matrix = np.array(matrix, dtype=np.float64)
    matrix_np = matrix.astype(np.float64)
    # 将-1值替换为NaN表示无数据区域
    matrix_np[matrix_np == -1] = np.nan
    # 设置vmax参数
    if vmax is None:
        # 使用nanmax忽略NaN值（无数据区域）
        vmax = np.nanmax(matrix_np)
        # 处理空矩阵的情况
        if np.isneginf(vmax) or np.isnan(vmax):
            vmax = 1.0
    if not isinstance(matrix, np.ndarray):
        # 尝试将列表矩阵转换为数值数组
        matrix = np.array(matrix, dtype=np.float64)
    # 处理可能的非数值类型
    if matrix.dtype == object:
        # 强制转换对象类型数组为float
        matrix = np.array(matrix, dtype=np.float64)
    matrix_np = matrix.astype(np.float64)
    # 创建掩码数组，将-1值标记为无数据区域（不修改原始数据）
    mask = (matrix_np == -1)
    matrix_masked = np.ma.masked_array(matrix_np, mask=mask)
    # 计算vmax时排除-1值
    if vmax is None:
        valid_values = matrix_np[~mask]
        vmax = np.max(valid_values) if valid_values.size > 0 else 1.0
    
    # 使用np.atleast_2d确保至少二维
    matrix_np = np.atleast_2d(matrix_np)
    # 如果超过二维，取前两维
    if matrix_np.ndim > 2:
        matrix_np = matrix_np[0, :, :]
    
    # 确保矩阵非空且维度有效
    if matrix_np.size == 0:
        matrix_np = np.zeros((1, 1))
        warnings.warn("Empty matrix detected, using default 1x1 matrix.")
    # 确保至少1x1的有效形状
    matrix_np = matrix_np.reshape(max(1, matrix_np.shape[0]), max(1, matrix_np.shape[1]))
    # 最终形状验证，确保为二维数组
    if matrix_np.ndim != 2:
        matrix_np = np.zeros((1, 1))
        warnings.warn(f"Matrix has invalid shape {matrix_np.shape}, using default 1x1 matrix.")
    # 可视化前最终形状检查
    if matrix_np.ndim != 2:
        matrix_np = np.zeros((1, 1), dtype=np.float64)
        warnings.warn(f"Visualization matrix has invalid shape {matrix_np.shape}, using default 1x1 matrix.")
    # 根据图像类型设置颜色映射
    if image_type == 'cnv':
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        min_val = 0
        max_val = vmax if vmax is not None else np.max(matrix_np)
        midpoint = 2.0
        
        # 确保vmin <= vcenter <= vmax
        vmin = min(min_val, midpoint)
        vmax = max(max_val, midpoint)
        
        # 创建蓝色到白色的渐变（小于2的值）
        blue_colors = plt.cm.Blues_r(np.linspace(0, 1, 128))
        # 创建白色到红色的渐变（大于2的值）
        red_colors = plt.cm.Reds(np.linspace(0, 1, 128))
        
        # 合并颜色并创建自定义渐变颜色映射
        cmap_colors = np.vstack((blue_colors, red_colors))
        cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        cmap.set_bad('gray')  # 设置无数据区域为灰色
        
        # 使用TwoSlopeNorm确保中间点(2)为白色
        norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
        
        # 使用掩码数组进行可视化，保留原始数据
        plt.imshow(matrix_masked, cmap=cmap, norm=norm, interpolation='nearest')
    else:  # snv默认蓝色映射
        # 创建掩码数组，将-1值标记为无数据区域
        mask = (matrix_np == -1)
        matrix_masked = np.ma.masked_array(matrix_np, mask=mask)
        plt.imshow(matrix_masked, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)
        plt.cm.Blues.set_bad('gray')

    #plt.imshow(matrix, cmap='Blues', interpolation='bicubic', vmin=0, vmax=np.max(matrix))

    if output_path is not None:

        name = output_path.split('/')[-1]

    # 根据图像类型设置颜色条标签
    if image_type == 'cnv':
        plt.colorbar(label='Copy Number')
    else:
        plt.colorbar(label='SNP Frequency')

    plt.xlabel('x')

    plt.ylabel('y')

    plt.axis('on')
    

    if output_path is not None:

        plt.savefig(output_path)

        print(f"图像已保存至: {output_path}")
    else:

        plt.show()


def generate_matrix():

    # 获取用户输入的二维坐标系大小

    rows = args.rows

    cols = args.cols

    matrix = None
    

    if args.shape =='circle':

        matrix = generate_circle(rows, cols)

    elif args.shape == 'square':

        matrix = generate_square(rows, cols)

    elif args.shape == 'image':

        image_path = args.image_path

        matrix = generate_from_image(rows, cols, image_path)
    

    if matrix is None:

        # 如果matrix未被赋值，返回空矩阵

        # 将无数据区域设置为-1而非0
        matrix = np.full((rows, cols), -1, dtype=int)
        

    # 将矩阵中的0值替换为-1以表示无数据区域
    if matrix is not None:
        matrix = np.where(matrix == 0, -1, matrix)
    return matrix


def boundary_mutation(matrix, boundary_type='diagonal', normal_copy=2, variant_copy=4):

    """

    分界变异模式

    :param matrix: 原始细胞矩阵

    :param boundary_type: 分界类型 ('diagonal'-对角线, 'horizontal'-水平, 'vertical'-垂直)

    :param normal_copy: 正常区域的拷贝数 (默认为2)

    :param variant_copy: 变异区域的拷贝数 (默认为4)

    :return: 拷贝数变异矩阵

    """

    rows, cols = matrix.shape

    mutation_matrix = np.ones_like(matrix, dtype=float) * normal_copy
    

    for i in range(rows):

        for j in range(cols):

            if boundary_type == 'diagonal':

                if i > j:  # 对角线分界

                    mutation_matrix[i,j] = variant_copy

            elif boundary_type == 'horizontal':

                if i > rows//2:  # 水平分界

                    mutation_matrix[i,j] = variant_copy

            elif boundary_type == 'vertical':

                if j > cols//2:  # 垂直分界

                    mutation_matrix[i,j] = variant_copy
    

    # 添加调试信息检查矩阵值范围
    print(f"nested_mutation stats - min: {mutation_matrix.min()}, max: {mutation_matrix.max()}, non-zero: {np.count_nonzero(mutation_matrix)}")
    # 将非变异区域设为正常拷贝数2（白色），确保颜色映射正确
    return np.where(matrix == 1, mutation_matrix, -1)


def gradient_mutation(matrix, gradient_direction='horizontal', max_copy=9, center_row=None, center_col=None, min_copy=0):

    """

    渐变变异模式

    :param matrix: 原始细胞矩阵

    :param gradient_direction: 渐变方向 ('horizontal'-水平, 'vertical'-垂直, 'radial'-圆心扩散)

    :param max_copy: 最大拷贝数 (默认为9)

    :param center_row: 中心行坐标 (None表示自动计算中心)

    :param center_col: 中心列坐标 (None表示自动计算中心)

    :param min_copy: 最小拷贝数 (默认为0)

    :return: 拷贝数变异矩阵 (值大于min_copy且保留一位小数)

    """

    max_copy = float(max_copy)

    rows, cols = matrix.shape

    mutation_matrix = np.ones_like(matrix, dtype=float) * min_copy
    

    # 设置中心点

    if center_row is None:

        center_row = rows // 2

    if center_col is None:

        center_col = cols // 2
        

    max_radius = min(center_row, center_col, rows-center_row, cols-center_col)
    

    for i in range(rows):

        for j in range(cols):

            if gradient_direction == 'horizontal':

                # 水平渐变 (从左到右，拷贝数从min_copy增加到max_copy)

                copy_num = min_copy + (max_copy - min_copy) * j/cols

                mutation_matrix[i,j] = round(max(min_copy + 0.1, copy_num), 1)

            elif gradient_direction == 'vertical':

                # 垂直渐变 (从上到下，拷贝数从min_copy增加到max_copy)

                copy_num = min_copy + (max_copy - min_copy) * i/rows

                mutation_matrix[i,j] = round(max(min_copy + 0.1, copy_num), 1)

            elif gradient_direction == 'radial':
                # 圆心扩散 (从中心向外，拷贝数从max_copy减少到min_copy) - 使用非线性变换更符合生物学梯度
                distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                # 避免除零错误
                if max_radius == 0:
                    normalized_distance = 0
                else:
                    normalized_distance = distance / max_radius
                    
                # 使用指数衰减函数模拟更自然的生物学梯度
                # 参数2.0控制衰减速度，可以根据需要调整
                k = 1.7
                copy_num = max_copy * np.exp(-k * normalized_distance) + min_copy * (1 - np.exp(-k * normalized_distance))
                
                # 确保值在有效范围内
                mutation_matrix[i,j] = round(max(min_copy + 0.1, min(max_copy, copy_num)), 1)
    

    return np.where(matrix == 1, mutation_matrix, -1)


def nested_mutation(matrix, inner_copy=4, middle_copy=1, outer_copy=2, center_row=None, center_col=None):

    """

    嵌套变异模式

    :param matrix: 原始细胞矩阵

    :param inner_copy: 核心区域拷贝数 (默认为4)

    :param middle_copy: 中间区域拷贝数 (默认为1)

    :param outer_copy: 外围区域拷贝数 (默认为2)

    :param center_row: 中心行坐标 (None表示自动计算中心)

    :param center_col: 中心列坐标 (None表示自动计算中心)

    :return: 拷贝数变异矩阵

    """

    rows, cols = matrix.shape

    mutation_matrix = np.ones_like(matrix, dtype=float) * outer_copy
    

    # 设置中心点

    if center_row is None:

        center_row = rows // 2

    if center_col is None:

        center_col = cols // 2
        

    max_radius = min(center_row, center_col, rows-center_row, cols-center_col)
    

    for i in range(rows):

        for j in range(cols):

            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)

            # 根据距离中心的远近设置不同的拷贝数

            # 扩大核心和中间区域范围以确保可视化效果
            if distance < max_radius * 0.3:
                mutation_matrix[i,j] = inner_copy  # 核心区域拷贝数
            elif distance < max_radius * 0.6:
                mutation_matrix[i,j] = middle_copy  # 中间区域拷贝数
    

    return np.where(matrix == 1, mutation_matrix, -1)


def cnv_matrix(matrix, mode='boundary', boundary_type='diagonal', gradient_direction='radial', max_copy=6, min_copy=0):
    """
    生成CNV变异矩阵 (入口函数)

    :param matrix: 原始细胞矩阵
    :param mode: 变异模式 ('boundary'-分界, 'gradient'-渐变, 'nested'-嵌套)
    :param boundary_type: 分界类型 ('diagonal'-对角线, 'horizontal'-水平, 'vertical'-垂直)
    :param gradient_direction: 渐变方向 ('horizontal'-水平, 'vertical'-垂直, 'radial'-圆心扩散)
    :param max_copy: 最大拷贝数 (默认为6)
    :param min_copy: 最小拷贝数 (默认为1)
    :return: CNV拷贝数变异矩阵
    """
    if mode == 'boundary':
        return boundary_mutation(matrix, boundary_type, normal_copy=2, variant_copy=max_copy)
    elif mode == 'gradient':
        return gradient_mutation(matrix, gradient_direction, max_copy=max_copy, min_copy=min_copy)
    elif mode == 'nested':
        # 设置中间拷贝数为内外拷贝数的中间值，确保正确的视觉层次
        outer_copy = 1
        middle_copy = (outer_copy + max_copy) // 2
        return nested_mutation(matrix, inner_copy=max_copy, middle_copy=middle_copy, outer_copy=outer_copy)
    else:
        return np.where(matrix == 1, 2, 0)  # 默认返回正常拷贝数

def mutation_matrix(matrix, mode='boundary', boundary_type='diagonal', gradient_direction='radial', max_copy=6):

    """

    生成变异矩阵 (入口函数)

    :param matrix: 原始细胞矩阵

    :param mode: 变异模式 ('boundary'-分界, 'gradient'-渐变, 'nested'-嵌套)

    :param boundary_type: 分界类型 ('diagonal'-对角线, 'horizontal'-水平, 'vertical'-垂直)

    :param gradient_direction: 渐变方向 ('horizontal'-水平, 'vertical'-垂直, 'radial'-圆心扩散)

    :param max_copy: 最大拷贝数 (默认为4)

    :return: 拷贝数变异矩阵 (2表示正常拷贝数)

    """

    if mode == 'boundary':

        return boundary_mutation(matrix, boundary_type)

    elif mode == 'gradient':

        return gradient_mutation(matrix, gradient_direction, max_copy)

    elif mode == 'nested':
        # 将max_copy作为inner_copy传递，确保用户指定的最大拷贝数生效
        return nested_mutation(matrix, inner_copy=max_copy)
    else:

        return np.where(matrix == 1, 2, 0)  # 默认返回正常拷贝数



def snv_matrix(matrix, mode='boundary', boundary_type='diagonal', gradient_direction='radial', max_copy=1.0):

    """

    生成SNV变异概率矩阵 (入口函数)

    :param matrix: 原始细胞矩阵

    :param mode: 变异模式 ('boundary'-分界, 'gradient'-渐变, 'nested'-嵌套)

    :param boundary_type: 分界类型 ('diagonal'-对角线, 'horizontal'-水平, 'vertical'-垂直)

    :param gradient_direction: 渐变方向 ('horizontal'-水平, 'vertical'-垂直, 'radial'-圆心扩散)

    :param max_copy: 最大变异概率 (默认为1.0)

    :return: SNV变异概率矩阵 (值范围0-1)

    """

    # 使用mutation_matrix生成基础矩阵

    prob_matrix = mutation_matrix(matrix, mode=mode, boundary_type=boundary_type, 
                               gradient_direction=gradient_direction, max_copy=max_copy)
    

    # 将拷贝数转换为变异概率 (归一化到0-1范围)

    # 对于boundary模式，将4转换为max_copy，将2转换为0

    if mode == 'boundary':

        for i in range(prob_matrix.shape[0]):

            for j in range(prob_matrix.shape[1]):

                if prob_matrix[i,j] == 4:  # 变异区域

                    prob_matrix[i,j] = max_copy

                elif prob_matrix[i,j] == 2:  # 正常区域

                    prob_matrix[i,j] = 0

    # 对于nested模式，将middle_copy(1)转换为0（正常区域），将inner_copy(4)和outer_copy(2)转换为变异概率

    elif mode == 'nested':

        for i in range(prob_matrix.shape[0]):

            for j in range(prob_matrix.shape[1]):

                if prob_matrix[i,j] == 1:  # 中间区域（正常区域）

                    prob_matrix[i,j] = 0

                elif prob_matrix[i,j] == 4:  # 内部区域（变异区域）

                    prob_matrix[i,j] = max_copy

                elif prob_matrix[i,j] == 2:  # 外部区域（变异区域）

                    prob_matrix[i,j] = max_copy * 0.5  # 外部区域变异概率设为最大值的一半
    

    return prob_matrix


import numpy as np

def adjust_gene_expression_by_cnv(cell, cell_cnv, gene_annotations, noise_level=0, noise_dispersion=1, spatial_corr=0):
    """
    根据CNV信息调整细胞的基因表达量
    
    参数:
    cell: Cell对象 - 包含基因表达数据的细胞对象
    cell_cnv: str - 细胞的CNV信息字符串
    gene_annotations: dict - 基因注释字典，包含基因的染色体位置信息
    noise_level: float - 噪声水平，0表示无噪声，值越大噪声越强 (默认: 0)
    noise_dispersion: float - 噪声分散度，值越大噪声越小 (默认: 1)
    spatial_corr: float - 空间相关性强度，0表示无空间相关性 (默认: 0)
    """
    # 参数验证
    noise_level = max(0, noise_level)  # 确保噪声水平非负
    noise_dispersion = max(0.1, noise_dispersion)  # 确保分散度为正
    spatial_corr = np.clip(spatial_corr, -1, 1)  # 空间相关性限制在[-1, 1]

    if cell_cnv and gene_annotations and cell.gene_expression:
        # 解析CNV信息
        cnv_regions = []
        for cnv_part in cell_cnv.split(':'):
              # 清理并分割CNV信息，过滤空字符串
              cnv_info = [part.strip() for part in cnv_part.split(',') if part.strip()]
              # 添加调试日志

              # 验证是否有足够元素
              if len(cnv_info) >= 5:
                  try:
                      # 重新调整索引位置，匹配实际CNV数据格式
                      cnv_type = cnv_info[0]
                      chrom = cnv_info[1]
                      start = int(cnv_info[2])
                      end = int(cnv_info[3])
                      copy_num = float(cnv_info[4])
                      cnv_regions.append((chrom, start, end, copy_num))
                  except ValueError as e:
                      print(f"解析CNV条目失败: {cnv_part}, 错误: {str(e)}, 字段值: {cnv_info}")
              else:
                  print(f"跳过无效的CNV条目: {cnv_part}, 元素数量不足: {len(cnv_info)} (预期至少5个)")

        # 调整基因表达量
        for gene, expr in cell.gene_expression.items():
            if gene in gene_annotations:
                gene_chrom, gene_start, gene_end = gene_annotations[gene]
                # 检查基因是否在CNV区域内
                for (cnv_chrom, cnv_start, cnv_end, copy_num) in cnv_regions:
                    if (gene_chrom == cnv_chrom and 
                        gene_start >= cnv_start and 
                        gene_end <= cnv_end):
                        # 基础拷贝数为2，按比例调整表达量
                        adjustment_ratio = copy_num / 2
                        adjusted_expr = round(expr * adjustment_ratio)
                        cell.gene_expression[gene] = adjusted_expr

                        # 添加负二项分布噪声及空间相关性
                        if noise_level > 0 and hasattr(cell, 'coordinate'):
                            # 获取细胞坐标
                            x, y = cell.coordinate
                            # 生成空间相关噪声（简化模型）
                            spatial_factor = 1 + spatial_corr * (np.sin(x/5) * np.cos(y/5))
                            # 负二项分布参数
                            mu = adjusted_expr * spatial_factor
                            if mu <= 0:  # 确保均值为正
                                mu = 1
                            # 分散参数=1/noise_dispersion，值越小噪声越大
                            n = 1 / noise_dispersion
                            p = n / (n + mu)
                            # 生成噪声表达量
                            noisy_expr = np.random.negative_binomial(n, p)
                            cell.gene_expression[gene] = noisy_expr

def generate_organization(matrix):

    """

    根据矩阵生成组织，可选包含拷贝数变异信息

    :param matrix: 原始细胞矩阵

    :param mutation_matrix:

    :return: Organization对象

    """
    print("generate_organization\n")
    organization = TMBspatial_cell.Organization(matrix.shape[0], matrix.shape[1])

    # 读取表达矩阵和基因注释文件
    expr_df = None
    gene_annotations = {}
    
    # 读取表达矩阵
    if args.expression_matrix and os.path.exists(args.expression_matrix):
        try:
            import pandas as pd
            expr_df = pd.read_csv(args.expression_matrix, index_col=0)
            # 存储基因顺序
            organization.gene_order = expr_df.columns.tolist()
            print(f"成功读取表达矩阵，包含{len(organization.gene_order)}个基因")
        except Exception as e:
            print(f"读取表达矩阵失败: {e}")
    
    # 读取基因注释文件
    if args.gene_annotation and os.path.exists(args.gene_annotation):
        try:
            import pandas as pd
            # 基因注释文件为制表符分隔，包含表头gene_name	chromosome	start_position	end_position
            annot_df = pd.read_csv(args.gene_annotation, sep='\t')
            for _, row in annot_df.iterrows():
                gene_annotations[row['gene_name']] = (row['chromosome'], row['start_position'], row['end_position'])
            print(f"成功读取基因注释，包含{len(gene_annotations)}个基因区域")
        except Exception as e:
            print(f"读取基因注释文件失败: {e}")
    
    all_coordinates = get_all_coordinates(matrix)
    i = 0
    a_snv_matrix = None
    cnv = args.TMBspatial_cnv
    snv = args.TMBspatial_snv


    if snv:

        snv_arr = snv.split(",")

        # 确定SNV变异模式和方向，优先使用专用参数
        snv_mode = args.snv_mutation_mode if args.snv_mutation_mode else args.TMBspatial_mutation_mode
        snv_dir = args.snv_gradient_direction if args.snv_gradient_direction else args.gradient_direction
        a_snv_matrix = snv_matrix(matrix, mode=snv_mode, gradient_direction=snv_dir, max_copy=float(snv_arr[5]) if len(snv_arr) > 5 else 1.0)
    


    if cnv:
        import numpy as np
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cnv_arr = cnv.split(":")

        cnv_matrix_arr = []

        for cnv_one in cnv_arr:

            cnv_one_arr = cnv_one.split(",")

            # 确定CNV变异模式和方向，优先使用专用参数
            cnv_mode = args.cnv_mutation_mode if args.cnv_mutation_mode else args.TMBspatial_mutation_mode
            cnv_dir = args.cnv_gradient_direction if args.cnv_gradient_direction else args.gradient_direction
            cnv_mat = cnv_matrix(matrix, mode=cnv_mode, gradient_direction=cnv_dir, max_copy=float(cnv_one_arr[4]))

            cnv_matrix_arr.append(cnv_mat)
    else:

        cnv_matrix_arr = None
    
    

    for coordinate in all_coordinates:

        id = str(i).zfill(8)


        # 创建细胞

        cell = TMBspatial_cell.Cell(id, coordinate)

        # 设置基因表达数据
        if expr_df is not None:
            coord_key = f"{coordinate[0]}x{coordinate[1]}"
            if coord_key in expr_df.index:
                # 获取该坐标的所有基因表达值
                gene_expr = expr_df.loc[coord_key].to_dict()
                cell.gene_expression = gene_expr
            else:
                print(f"警告：表达矩阵中未找到坐标{coord_key}的数据")

        #snv变异

        if a_snv_matrix is not None:

            snv_arr[5] = str(a_snv_matrix[coordinate[0]-1, coordinate[1]-1])

            asnv=",".join(snv_arr)

            cell.set_snv(asnv)


        # 使用set_cnv方法设置拷贝数

        if cnv_matrix_arr is not None:

            cell_cnv = ''

            for (cnv_mat_element, acnv) in zip(cnv_matrix_arr, cnv_arr):

                # 获取拷贝数，默认为2

                copy_number = 2

                # 注意坐标从1开始，而矩阵索引从0开始

                copy_number = cnv_mat_element[coordinate[0]-1, coordinate[1]-1]

                cnv_one_arr = acnv.split(",")

                cnv_one_arr[4] = str(copy_number)

                cell_cnv += ",".join(cnv_one_arr)+':'

            cell_cnv = cell_cnv[:-1]

            cell.set_cnv(cell_cnv)

            # 根据CNV调整基因表达量
            # 调用函数调整基因表达量
            adjust_gene_expression_by_cnv(cell, cell_cnv, gene_annotations, noise_level=args.noise_level, noise_dispersion=args.noise_dispersion, spatial_corr=args.spatial_corr)

        organization.add({id:cell})

        i += 1

    organization.set_matrix(matrix)
    print("organization.matrix:",organization.matrix)
    print("matrix:",matrix)
    return organization


def read_barcodes(barcode_file):

    """

    读取barcode文件

    :param barcode_file: barcode文件路径

    :return: 坐标到barcode的映射字典

    """

    barcodes = {}

    if barcode_file and os.path.exists(barcode_file):

        try:

            with open(barcode_file, 'r') as f:

                for line in f:

                    parts = line.strip().split('\t')

                    if len(parts) >= 3:

                        barcode = parts[0]

                        row = int(parts[1])

                        col = int(parts[2])

                        barcodes[(row, col)] = barcode

            print(f"成功读取{len(barcodes)}个barcode")

        except Exception as e:

            print(f"读取barcode文件出错: {e}")

    return barcodes


def insert_barcode_to_make2(out_dir, cell_coord, barcodes):

    """

    在make2.fq文件中插入barcode

    :param out_dir: 输出目录

    :param cell_coord: 细胞坐标

    :param barcodes: 坐标到barcode的映射字典

    """

    make2_path = os.path.join(out_dir, 'make2.fq')

    if not os.path.exists(make2_path):

        print(f"make2.fq文件不存在，跳过barcode插入: {make2_path}")
        return
    

    # 如果没有对应的barcode，直接返回

    if cell_coord not in barcodes:

        print(f"坐标 {cell_coord} 没有对应的barcode，跳过插入")
        return
    

    barcode = barcodes[cell_coord]

    if len(barcode) < 16:

        print(f"barcode长度不足16位，跳过插入: {barcode}")
        return
    

    # 读取文件内容

    with open(make2_path, 'r') as f:

        lines = f.readlines()
    

    # 只修改第一个序列，每4行为一组（标题行、序列行、加号行、质量行）

    if len(lines) >= 4:

        header = lines[0]

        sequence = lines[1].strip()

        plus = lines[2]

        quality = lines[3].strip()
        

        # 插入barcode

        barcode1 = barcode[8:16]  # 取barcode的9-16位作为Barcode1

        barcode2 = barcode[0:8]   # 取barcode的1-8位作为Barcode2
        

        # 确保序列长度足够

        if len(sequence) >= 40:

            # 在33-40位置插入Barcode2

            sequence = sequence[:32] + barcode2 + sequence[40:]
            

            # 调整质量值

            #quality = quality[:32] + 'I' * 8 + quality[40:]
            

            # 如果序列长度足够，在71-78位置插入Barcode1

            if len(sequence) >= 78:

                sequence = sequence[:70] + barcode1 + sequence[78:]

                #quality = quality[:70] + 'I' * 8 + quality[78:]
        

        # 更新文件内容

        lines[1] = sequence + '\n'

        #lines[3] = quality + '\n'
        

        # 写回文件

        with open(make2_path, 'w') as f:

            f.writelines(lines)
        

        print(f"barcode插入完成: {make2_path}")
    else:

        print(f"文件格式不正确，跳过barcode插入: {make2_path}")


def out_TMBspatial_data(organization):

    out = args.out_path

    #用户为指定数据输出路径则使用当前时间做文件夹名保存文件

    if out is None:

        out = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    if not os.path.exists(out):
        os.makedirs(out)


    # 导出调整后的基因表达矩阵
    if hasattr(organization, 'gene_order') and organization.gene_order:
        try:
            import pandas as pd
            import numpy as np

            # 准备表达矩阵数据
            expr_data = {}
            coord_index = []

            # 收集所有细胞的表达数据
            for cell_id, cell in organization.cell_dict.items():
                coord = cell.coordinate
                coord_key = f"{coord[0]}x{coord[1]}"
                coord_index.append(coord_key)

                # 按基因顺序提取表达值
                expr_values = []
                for gene in organization.gene_order:
                    expr_values.append(cell.gene_expression.get(gene, 0.0))

                expr_data[cell_id] = expr_values

            # 创建DataFrame
            expr_matrix = pd.DataFrame.from_dict(
                expr_data, 
                orient='index', 
                columns=organization.gene_order
            )
            expr_matrix.index = coord_index

            # 保存为CSV文件
            output_path = os.path.join(out, 'expression_matrix_adjusted.csv')
            expr_matrix.to_csv(output_path)
            print(f"调整后的基因表达矩阵已保存至: {output_path}")

        except Exception as e:
            print(f"导出基因表达矩阵失败: {e}")

    organization.save_to_csv(out)

    matrix = organization.matrix
    print
    # 保存原始组织形态的可视化

    visualize_matrix(matrix, out+'/original_matrix.png')

    a_snv_matrix = None

    cnv = args.TMBspatial_cnv

    snv = args.TMBspatial_snv
    bed = args.TMBspatial_bed


    if snv:

        snv_arr = snv.split(",")

        a_snv_matrix = snv_matrix(matrix, mode=args.snv_mutation_mode, gradient_direction=args.snv_gradient_direction, max_copy=float(snv_arr[5]) if len(snv_arr) > 5 else 1.0)

        visualize_matrix(a_snv_matrix, output_path=out+'/snv_matrix.png', vmax=1, image_type='snv')
    

    if cnv:
        import numpy as np
        cnv_arr = cnv.split(":")

        i = 0

        for cnv_one in cnv_arr:

            cnv_one_arr = cnv_one.split(",")

            cnv_matrix = mutation_matrix(matrix, mode=args.cnv_mutation_mode,gradient_direction=args.cnv_gradient_direction,max_copy=cnv_one_arr[4])

            
            # 自定义颜色映射：小于2为蓝色，等于2为白色，大于2为红色
            visualize_matrix(cnv_matrix, output_path=out+f'/cnv_matrix{i}.png', vmax=float(cnv_one_arr[4]), image_type='cnv')

            i += 1

    
    

    #读取barcode文件

    barcodes = read_barcodes(args.barcode_file)

    # 为每个细胞生成数据并插入barcode

    for v in organization.cell_dict.values():

        # 生成模拟数据

        generate_simdata(v.id, out, TMBspatial_snv=v.snv, TMBspatial_cnv=v.cnv, TMBspatial_bed=bed)
    

        # 如果有barcode文件，则插入barcode

        if args.barcode_file:

            insert_barcode_to_make2(out+'/'+v.id, v.coordinate, barcodes)
    

    merge_fastq_files(out,'merge')
    
    return




def main():

    #生成组织矩阵

    #获得细胞坐标

    matrix = generate_matrix()
    organization = generate_organization(matrix)
    out_TMBspatial_data(organization)



if __name__ == "__main__":
    main()
