# FROM ubuntu:22.04
FROM python:3.9.17
WORKDIR /digits
# copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /digits/requirements.txt


#create volume to mount it on host
# VOLUME /digits/models

ENV FLASK_APP=api/app
# CMD ["flask", "run", "--host=0.0.0.0"]
CMD ["python", "api/app.py"]
#run python script to train model
# ENTRYPOINT ["python","exp.py"]
# run exp.py to train svm model
# CMD ["python", "exp.py", "--total_run", "1", "--dev_size", "0.2", "--test_size", "0.2", "--model_type", "svm"]
