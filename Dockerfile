FROM python:3.12
WORKDIR /rent_price_pred
COPY /app /rent_price_pred/app
RUN pip install -r /rent_price_pred/app/requirements_app.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]