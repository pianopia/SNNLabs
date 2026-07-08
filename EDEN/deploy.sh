# プロジェクトIDを直書き
PROJECT_ID="pianopia"

gcloud config set project $PROJECT_ID

gcloud artifacts repositories create eden-web \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Frontend Docker images"

# ビルド
docker buildx build -t asia-northeast1-docker.pkg.dev/$PROJECT_ID/eden-web/eden-web:latest --platform linux/amd64 .

# プッシュ
docker push asia-northeast1-docker.pkg.dev/$PROJECT_ID/eden-web/eden-web:latest

gcloud run deploy eden-web --image asia-northeast1-docker.pkg.dev/$PROJECT_ID/eden-web/eden-web:latest --platform managed --set-env-vars NODE_ENV=production --region=asia-northeast1

gcloud run services update-traffic eden-web --to-latest --region=asia-northeast1