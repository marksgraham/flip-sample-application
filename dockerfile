#FROM python:3.8.10
FROM nvcr.io/nvidia/pytorch:21.05-py3

RUN apt-get update && apt-get install -y dos2unix

RUN pip install -U pip
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

RUN python3 -c "import torchvision;torchvision.datasets.CIFAR10(root='/root/data/', download=True)"

WORKDIR /nvflare/
RUN yes | poc -n 2 \
  && mkdir -p poc/admin/transfer

COPY ./apps/ /nvflare/poc/admin/transfer/
COPY ./utils/test.py /nvflare/poc/admin/

COPY ./utils/start_nvflare_components.sh /nvflare/

# Fix the Docker issue with line endings from .sh files when running it on Windows
# from https://github.com/docker/for-win/issues/1340
RUN dos2unix /nvflare/start_nvflare_components.sh
RUN chmod 777 /nvflare/start_nvflare_components.sh

ENV PATH="${PATH}:/nvflare/poc/admin/startup"
CMD ["./start_nvflare_components.sh"]
