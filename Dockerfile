FROM python:3.6-slim-stretch
ENV GCSFUSE_REPO gcsfuse-stretch

RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
  && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
RUN mkdir /mnt/bucket

RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip python-virtualenv && \
rm -rf /var/lib/apt/lists/*

RUN \
  apt-get update && \
  apt-get install -y unzip wget && \
rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./*.py ./
COPY ./microesc /usr/src/app/microesc
COPY ./experiments /usr/src/app/experiments

CMD ["sleep", "3600"]
