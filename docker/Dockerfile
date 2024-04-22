FROM osrf/ros:noetic-desktop-full

# 更新源
RUN echo "\
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse \
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse \
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse \
deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse \
" > /etc/apt/sources.list

# 安装ceres，gtsam，eigen等依赖库
RUN apt-get update
RUN apt-get install -y libgoogle-glog-dev libeigen3-dev libsuitesparse-dev libyaml-cpp-dev libbtbb-dev libgmock-dev \
    cmake libgflags-dev libatlas-base-dev libfmt-dev git

# 拷贝第三方库
COPY ./thirdparty/ /tmp/

# 安装gtsam
RUN apt-get install -y cmake libboost-all-dev libtbb-dev python3-dev python3-numpy doxygen
RUN cd /tmp/gtsam \
    && mkdir build \
    && cd build \
    && cmake -DGTSAM_USE_SYSTEM_EIGEN=ON - GTSAM_BUILD_WITH_MARCH_NATIVE=Off ..\
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

# 安装sophus库
RUN cd /tmp/Sophus \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

RUN ldconfig
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /var/tmp/*