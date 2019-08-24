FROM python:3.6-slim

MAINTAINER Bruno Casali

ADD requirements.txt /app/requirements.txt

WORKDIR /app

RUN python3.6 -m pip install -r requirements.txt

ADD . /app

RUN mkdir logs

RUN python3.6 -m nltk.downloader wordnet pros_cons reuters stopwords rslp punkt

EXPOSE 5000

CMD 'bin/bash'
