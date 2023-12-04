FROM python:3.9-slim

WORKDIR /app

ENV INPUT_DIR="/app/docs"

VOLUME ${INPUT_DIR}

RUN apt-get update 
    # && apt-get install -y \
    # build-essential \
    # curl \
    # software-properties-common \
    # git \
    # && rm -rf /var/lib/apt/lists/*

COPY src/* .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

# RUN echo "$(ls -la /app/docs)"

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py", "--input-dir", "${INPUT_DIR}"]