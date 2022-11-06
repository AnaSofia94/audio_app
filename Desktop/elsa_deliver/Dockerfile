FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR .

COPY requirements.txt /app/requirements.txt


COPY requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1


RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY app /app