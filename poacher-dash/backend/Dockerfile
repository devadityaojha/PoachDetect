
FROM python:3.8

RUN mkdir /app
WORKDIR /app

RUN apt update && \
    apt install -y postgresql-client \
    python3-pip \
    pip3 install --upgrade pip \
    pip3 install -f sounddevice \
    pip3 install -f scipy

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
