import sys
from tqdm import tqdm

bar = tqdm(total=6, desc="[人物识别]", file=sys.stdout, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", leave=True, mininterval=0.3, ascii=True)
bar.total = 0 # 会不会是谁修改了total属性？
bar.refresh()
bar.close()
