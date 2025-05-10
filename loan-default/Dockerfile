
FROM python:3.10-slim

ADD requirements/ requirements/

RUN pip install -r requirements/api_requirements.txt

ADD *.py .

EXPOSE 7860
EXPOSE 8000

CMD ["python", "app2.py"]


# docker build . -t loan_defaulter_pred

# docker run -it -d -p 8000:8000 --name myapp loan_defaulter_pred
