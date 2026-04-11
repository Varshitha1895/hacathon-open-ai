FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# Port 7860 use cheyalani chepthunnam
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
