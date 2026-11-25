# inbed

a text embedding server in Rust with zero crate dependencies. runs on Metal, CUDA, or the CPU and speaks the OpenAI `/v1/embeddings` API.

```
Qwen3-Embedding-0.6B, RTX 4090:  1.9 ms per query, 112k tok/s batched
```

## install

```bash
nix run github:drbh/inbed -- serve --model qwen/qwen3-embedding-0p6b    # or .#cuda on linux
cargo install --git https://github.com/drbh/inbed.git
```

## usage

```bash
hf download Qwen/Qwen3-Embedding-0.6B
inbed serve --model qwen/qwen3-embedding-0p6b

curl -s 127.0.0.1:8070/v1/embeddings -d '{
  "model": "qwen/qwen3-embedding-0p6b",
  "input": ["red fox", "blue bird"],
  "dimensions": 512
}'
```

`inbed --help` lists the flags. any BERT or Qwen3 embedding checkpoint in the Hugging Face cache works.

## develop

```bash
cargo test --release
cargo bench
```
