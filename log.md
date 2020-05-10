## 关于proposal文件格式的解读

``` 
# 5  % 一个简单的video编号
/data/DataSets/THUMOS14/frames/val/video_validation_0000154 # 文件地址
1573 # duration帧数
1 # fps都是1 
3 # gt个数
4 161 347 #  label start_frame end_frame
4 389 589
4 640 876
38 # proposal个数
4 0.3031 0.3031 108 888 # label, overlap, overlap_self, start_frame, end_frame
4 0.2699 0.2699 96 972
4 0.2984 0.2984 174 846
4 0.3375 0.3375 216 810
4 0.2403 0.2403 0 984
4 0.4925 0.4925 402 882
4 0.4021 0.4021 384 972
0 0.0000 0.0000 0 66
# 2
```