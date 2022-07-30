FROM python:3.8.10

ENV PYTHONBUFFERED 1

WORKDIR /modelvalidation

ADD . /modelvalidation

COPY ./requirements.txt /modelvalidation/requirements.txt

RUN pip install -r requirements.txt

COPY . /modelvalidation