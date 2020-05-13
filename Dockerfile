FROM python:3.8.2
WORKDIR /working
RUN git clone -b production https://github.com/sw6-aau/LSTnet-demo.git
WORKDIR /working/LSTnet-demo/
RUN mkdir log/ save/ data/
COPY . /working/LSTnet-demo/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]