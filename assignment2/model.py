import numpy as np

# 读取.ply文件
with open('./model/bun_zipper_res2.ply', 'r') as f:
    lines = f.readlines()

# 获取点的数量和起始行号
for i, line in enumerate(lines):
    if line.startswith('element vertex'):
        num_points = int(line.split()[2])
        start_line = i + 9
        break

# 读取点的坐标
points = []
for i in range(start_line, start_line + num_points):
    x, y, z = map(float, lines[i].split()[:3])
    points.append([x, y, z])

# 将点的坐标绕X轴顺时针旋转90度
points = np.array(points)
points[:, 1], points[:, 2] = points[:, 2], -points[:, 1]

# 将点的坐标放大
points = points * 500 // 21.2

# 将点沿X轴正向平移50
points[:, 0] += 60 // 21.2
# 将点沿Y轴正向平移70
points[:, 1] += 70 // 21.2
# 将点沿Z轴正向平移10
points[:, 2] += 10 // 21.2

# 输出坐标变换后的.ply文件
with open('./model.ply', 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment zipper output\n')
    f.write('element vertex %d\n' % num_points)
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('end_header\n')
    for i in range(num_points):
        f.write('%f %f %f\n' % tuple(points[i]))