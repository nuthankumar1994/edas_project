sudo docker run -d \
    --name supabase_db \
    -p 5019:5432 \
    -e POSTGRES_DB=vecs_db \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_USER=postgres \
    supabase/postgres:15.1.0.74


uvicorn app.main:app --reload