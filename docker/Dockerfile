# FROM ubuntu:22.04
FROM python:3.9.17
# copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt

WORKDIR /digits

CMD ["pytest"]


