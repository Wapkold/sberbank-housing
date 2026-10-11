"""Скачивает данные Sberbank Russian Housing Market в папку data/ и распаковывает train.csv.

Оригинал лежит на Kaggle (https://www.kaggle.com/c/sberbank-russian-housing-market)
и требует аккаунта и принятых правил соревнования. Этот скрипт берёт те же файлы
из публичной копии на GitHub (catherinehata/Sberbank-Russian-Housing-Market);
размеры совпадают с оригиналом: train 30 471 × 292, test 7 662 × 291, macro 2 484 × 100.

Запуск: python download_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://raw.githubusercontent.com/catherinehata/Sberbank-Russian-Housing-Market/master/"
FILES = ["train.csv.zip", "test.csv", "macro.csv"]
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA / name
        if target.exists():
            print(f"{name}: уже есть, пропускаю")
            continue
        print(f"{name}: скачиваю...", end=" ", flush=True)
        # Сначала во временный файл: оборванная загрузка не должна выглядеть как готовая
        part = target.with_name(name + ".part")
        urllib.request.urlretrieve(BASE + name, part)
        part.replace(target)
        print(f"{target.stat().st_size / 1e6:.1f} МБ")

    # В архиве кроме train.csv лежит служебная папка __MACOSX, поэтому
    # pandas не читает его напрямую: распаковываем только нужный файл
    if not (DATA / "train.csv").exists():
        with zipfile.ZipFile(DATA / "train.csv.zip") as z:
            z.extract("train.csv", DATA)
        print("train.csv: распакован")


if __name__ == "__main__":
    sys.exit(main())
