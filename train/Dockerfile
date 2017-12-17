#FROM tensorflow/tensorflow
FROM python

WORKDIR /train

RUN pip install keras \
                ujson \
                sklearn

COPY *.py /train/

