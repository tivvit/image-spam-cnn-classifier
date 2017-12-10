FROM ubuntu:xenial

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

RUN apt-get update && apt-get install -y curl gnupg2

RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -


RUN apt-get update && \
    apt-get install -y \
    tensorflow-model-server \
    && rm -rf /var/lib/apt/lists/*

#RUN mkdir /tfs/
#COPY sample.conf /www/email/szn-tensorflow-serving/conf/models.autoconf
#COPY sample /tfs/sample
COPY model.conf /

CMD ["tensorflow_model_server", "--port=2233", "--model_config_file=/model.conf"]
