FROM python:3.10.13-slim

# Set thư mục làm việc
WORKDIR /app

# Copy và cài đặt các thư viện yêu cầu
COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r /app/requirements.txt

# Tạo và chuyển quyền cho user không phải root
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/app && \
    chown -R user:user /home/user

# Thiết lập quyền cho user
USER user
WORKDIR /home/user/app

# Copy mã nguồn ứng dụng vào container
COPY . /home/user/app

EXPOSE 80

# Lệnh để khởi chạy Flask app
CMD ["python", "setup.py"]
