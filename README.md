
# Environmental Sound Classification on microcontrollers using Convolutional Neural Networks

## Status
**Work-in-progress**

## Keywords

    Wireless Sensor Networks, Embedded Systems
    Edge Computing, Edge Machine Learning
    Noise classification, ESC, Urbansounds
    Tensorflow, Keras, librosa

## See also

* [Machine Learning on Embedded Systems](https://github.com/jonnor/datascience-master/tree/master/embeddedml) notes.
* [emlearn](https://github.com/jonnor/emlearn) - Machine Learning inference engine for Microcontrollers and Embedded Systems


## Run experiments locally

Install dependencies

    pip install -r requirements.txt

Preprocess audio files into features

    python3 preprocess.py

Train the models

    python3 train.py

Evaluate the resulting models

    python3 test.py


## Run experiments using Docker, Kubernetes and Google Cloud

Create project in Google Cloud

Install locally

    Docker
    google-cloud-sdk
    kubectl

Create Kubernetes cluster

    gcloud container clusters create cluster --scopes storage-full --machine-type n1-highcpu-2 --num-nodes 10 \
        --create-subnetwork name=my-subnet-0 \
        --enable-ip-alias \
        --enable-private-nodes \
        --master-ipv4-cidr 172.16.0.0/28 \
        --no-enable-basic-auth \
        --no-issue-client-certificate \
        --no-enable-master-authorized-networks

    gcloud container clusters get-credentials cluster
    kubectl get nodes

Build Docker images and push to GKE

    export PROJECT_ID="$(gcloud config get-value project -q)"
    docker build -t gcr.io/${PROJECT_ID}/base:15 -f Dockerfile .
    docker push gcr.io/${PROJECT_ID}/base

Generate Kubernetes jobs and start them

    python3 microesc/jobs.py experiments/sbcnn16k30.yaml
    kubectl create -f data/jobs/

Delete jobs

    kubectl delete jobs `kubectl get jobs -o custom-columns=:.metadata.name`
