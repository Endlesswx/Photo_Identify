import sys
from tqdm import tqdm
import time

bar1 = tqdm(total=6, desc="[信息扫描]", file=sys.stdout, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", leave=True, mininterval=0.3, ascii=True)
bar2 = tqdm(total=6, desc="[人物识别]", file=sys.stdout, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", leave=True, mininterval=0.3, ascii=True)

bar1.update(2)
bar2.set_description_str("[人物识别] (0张有人脸) IMG_123.jpg", refresh=False)
bar2.refresh()
time.sleep(1)
bar1.close()
bar2.close()
