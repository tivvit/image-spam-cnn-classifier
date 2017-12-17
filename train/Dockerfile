FROM tensorflow/tensorflow

WORKDIR /train

RUN pip install keras \
                ujson \
                sklearn \
                h5py

COPY *.py /train/

