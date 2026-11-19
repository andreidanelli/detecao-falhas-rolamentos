import json
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/y2px5tg92h/"
VERSAO = 5
ROLAMENTOS = [1, 2, 6, 7, 11, 12, 16, 17]
PASTA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def abrir(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120)


def ler_json(url):
    with abrir(url) as r:
        return json.load(r)


def listar_csvs():
    pastas = ler_json(API + f"folders/{VERSAO}")
    nomes = {p["id"]: p["name"] for p in pastas}
    arquivos = []
    for p in pastas:
        pai = p.get("parent_id")
        if pai in nomes and nomes[pai].startswith("1_CSV"):
            for f in ler_json(API + f"files?folder_id={p['id']}&version={VERSAO}"):
                arquivos.append((f["filename"], f["size"], f["content_details"]["download_url"]))
    return arquivos


def main():
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    arquivos = [a for a in listar_csvs() if int(a[0].split("_")[1]) in ROLAMENTOS]

    for i, (nome, tamanho, url) in enumerate(arquivos, 1):
        destino = PASTA_RAW / nome
        if destino.exists() and destino.stat().st_size == tamanho:
            continue
        print(f"[{i}/{len(arquivos)}] baixando {nome}")
        with abrir(url) as r, open(destino, "wb") as f:
            while bloco := r.read(1 << 20):
                f.write(bloco)
        if destino.stat().st_size != tamanho:
            raise SystemExit(f"download incompleto: {nome}")

    print(f"{len(arquivos)} arquivos em {PASTA_RAW}")


if __name__ == "__main__":
    main()
