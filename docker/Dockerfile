FROM osrf/ros:noetic-desktop-full

# 直接添加
RUN echo "\
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse \
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse \
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse \
deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse \
" > /etc/apt/sources.list

# 安装ceres，gtsam，eigen等依赖库
RUN apt-get update
RUN apt-get install -y libopencv-dev libgoogle-glog-dev libeigen3-dev libsuitesparse-dev libpcl-dev libyaml-cpp-dev libbtbb-dev libgmock-dev unzip python3-tk
RUN apt-get install -y cmake libgflags-dev libatlas-base-dev libceres-dev libboost-all-dev libtbb-dev libfmt-dev

COPY ./thirdparty/ /tmp/

# 安装gtsam
RUN apt-get install -y cmake libboost-all-dev libtbb-dev python3-dev python3-numpy doxygen git
RUN cd /tmp/gtsam \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install

# 安装pangolin
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-dev \
    libglew-dev \
    libpython3-dev \
    libopencv-dev
RUN cd /tmp/Pangolin \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install
# 安装GLEW库（上面已经隐式的安装了）
# libglew-dev

# 安装sophus库
RUN cd /tmp/Sophus \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

RUN ldconfig
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /var/tmp/*