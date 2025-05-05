FROM python:3.9-slim

RUN pip install metaflow[service]

EXPOSE 8080

CMD ["python", "-m", "metaflow.metadata_service.main"]