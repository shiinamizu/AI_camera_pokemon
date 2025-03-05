FROM python:3.11

# 必要なパッケージをインストールする
WORKDIR /app

#COPY requirements.txt ./ 
COPY . .
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN apt-get install build-essential cmake pkg-config -y

RUN pip install --upgrade pip
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

#RUN apt-get install libusb-1.0-0-dev
RUN pip install --no-cache-dir -r requirements.txt