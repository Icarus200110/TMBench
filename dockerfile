# 使用 Miniconda 基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制 environment.yml
COPY environment.yml .

# 创建 Conda 环境
RUN conda env create -f environment.yml && conda clean -af

# 激活环境的简便方法：将环境 bin 加入 PATH
ENV PATH /opt/conda/envs/tmbench/bin:$PATH

# 复制项目代码
COPY . .

# 设置默认命令（确保使用正确的 Python 解释器）
CMD ["python", "-m", "tmbench"]
