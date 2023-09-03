FROM rpozzi/blink:latest AS blink

FROM python:3.8

# Upgrade PIP
RUN python3 -m pip install --upgrade pip

# Install utils
RUN apt-get update && \
    apt-get -y install --no-install-recommends gcc

WORKDIR /home/app

COPY . .
COPY --from=blink /home/app/blink blink

RUN pip install --no-cache-dir --upgrade -r /home/app/requirements.txt

# RUN pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK

ENV PYTHONPATH=/home/app

CMD python main.py --host 0.0.0.0 --port 80 --index $INDEXER_INDEX --postgres postgres://postgres:$POSTGRES_PASSWORD@postgres:5432/postgres --vector-size $INDEXER_VECTOR_SIZE --language $INDEXER_LANGUAGE

