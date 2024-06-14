
FROM python:3.8-slim-buster

COPY testdata ./ 
COPY templates ./ 
COPY pretrained_model ./  
COPY classification.py ./


WORKDIR /home/ujlee/Templates/Speech_disorder/brain

ADD . /home/ujlee/Templates/Speech_disorder/brain

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt


EXPOSE 80


CMD ["python", "app.py"]