FROM python:3.9

# Meta requirements ki port expose cheyali
EXPOSE 8000

WORKDIR /code

# Requirements install chestunnam
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Files anni copy chestunnam
COPY . .

# Meta system kavalasina port 8000 lo app ni run chestunnam
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
