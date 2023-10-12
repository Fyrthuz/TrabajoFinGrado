FROM ubuntu

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libxcb-xinerama0
RUN apt-get -y install python3-pyqt5

ENV DISPLAY=host.docker.internal:0.0
COPY . /app


ENTRYPOINT [ "python3","GUI.py" ]

