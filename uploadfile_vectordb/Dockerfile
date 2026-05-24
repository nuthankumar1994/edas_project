FROM python:3.9

RUN mkdir -p /root/nltk_data

COPY ./chinese_tmi_uploadfile/nltk_data/ /root/nltk_data

WORKDIR /rootDir

COPY ./chinese_tmi_uploadfile/requirements.txt /rootDir/requirements.txt

# ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install --no-cache-dir --default-timeout=100 -r /rootDir/requirements.txt

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

RUN python -m nltk.downloader averaged_perceptron_tagger

RUN apt-get install poppler-utils -y

RUN apt-get install tesseract-ocr -y

RUN git init

# RUN git clone --recurse-submodules https://github.com/tesseract-ocr/tessdata_fast.git 2> /dev/null || (cd tessdata_fast; git pull)

# RUN git clone --recurse-submodules https://github.com/tesseract-ocr/tessdata_fast.git || (cd tessdata_fast && git pull origin main)

RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/

COPY ./chinese_tmi_uploadfile/ /rootDir/

RUN cp kor.traineddata /usr/share/tesseract-ocr/5/tessdata/

RUN cp eng.traineddata /usr/share/tesseract-ocr/5/tessdata/

RUN cp chi_tra.traineddata /usr/share/tesseract-ocr/5/tessdata/

RUN cp chi_sim.traineddata /usr/share/tesseract-ocr/5/tessdata/

EXPOSE 5004

# chinese_tmi_demo/app/main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5004", "--workers", "4"]
