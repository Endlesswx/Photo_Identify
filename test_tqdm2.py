import sys
from tqdm import tqdm

bar = tqdm(total=0, desc="[人物识别]", file=sys.stdout, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", leave=True, mininterval=0.3, ascii=True)
bar.refresh()
bar.close()

bar2 = tqdm(total=None, desc="[人物识别]", file=sys.stdout, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", leave=True, mininterval=0.3, ascii=True)
bar2.refresh()
bar2.close()
