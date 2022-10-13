FROM python:3.8.10

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools

COPY ./requirements.txt ./

RUN pip install -r ./requirements.txt
RUN apt-get update && apt-get install -y dos2unix
RUN apt-get install -y wget

WORKDIR /nvflare/
RUN git clone https://github.com/NVIDIA/NVFlare.git

WORKDIR /nvflare/NVFlare
RUN git checkout 2.0

WORKDIR /nvflare/
COPY ./utils/start_nvflare_components.sh start_nvflare_components.sh

RUN dos2unix ./start_nvflare_components.sh

RUN yes | poc -n 2

WORKDIR /nvflare/poc/admin/
COPY apps transfer/
COPY ./utils/test.py ./

ENV PATH="${PATH}:/nvflare/poc/admin/startup"

WORKDIR /nvflare/

RUN chmod 777 ./start_nvflare_components.sh
CMD ["./start_nvflare_components.sh"]