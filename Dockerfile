# Cloud Run: Nim Crown app (listens on PORT, HOST 0.0.0.0)
FROM nimlang/nim:2.2.8

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    libpcre3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV NIMBLE_DIR=/app/.nimble_lp
ENV PATH="/app/.nimble_lp/bin:${PATH}"

COPY elfentier_lp.nimble crown.json ./
COPY scripts/bootstrap.sh ./scripts/
RUN bash scripts/bootstrap.sh
COPY src ./src/
COPY public ./public/

RUN crown build

ENV HOST=0.0.0.0
EXPOSE 8080

CMD [".crown/main"]
