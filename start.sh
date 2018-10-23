#! /bin/sh

cd chatbot;
docker-compose build;
docker-compose up -d;
docker-compose exec iky_backend python manage.py init