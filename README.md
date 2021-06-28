# Frame Skip Algorithm
#### 모든 frame에서 YOLO를 사용하는 것이 아닌 yolo_frame에서만 YOLO를 사용하고 skip_frame에서는 yolo_frame에서 얻은 BoundingBOX를 사용하는 알고리즘입니다.
#### 비디오 영상에서는 이웃한 frame의 이미지 차이가 일반적으로 크지 않다는 점에서 생각하였습니다.  
#### 아래 저장소에서 좀 더 구체적으로 설명하였습니다.
https://github.com/kacel33/yolo_skip_frame
# 
### YOLO와 tensorrt최적화는 아래 저장소를 참고하였습니다.
https://github.com/jkjung-avt/tensorrt_demos

여기서 제가 한 환경설정은 밑에 사이트에 적어두었습니다.   
https://ddo-code.tistory.com/20

### 테스트 해 보실 때 https://github.com/jkjung-avt/tensorrt_demos(Demo #5)를 실행하시고 tensorrt_demos폴더에 custom_yolo.py를 넣고 실행하시면 됩니다.
``` 
python custom_yolo.py --video {videofile} --m {YOLO file} -n {fram skip number}
```

### Example
<pre><code> python custom_yolo.py --video shorts1.mp4 -m yolov4-416 -n 3</code></pre>

#  
이전에 yolo_frame_skip이라는 저장소에서 frame_skip알고리즘을 제안하였습니다.    
거기에서는 opencv를 이용하여 진행하였는데  
## 이 저장소에서는 YOLOv4-416 pytorch버전을 Tensorrt로 변환 후에 실험하였습니다.

RTX 3080에서 실험한 결과입니다.

|원본 동영상 시간|YOLOv4-416(tensorrt)|skip_frame=1|skip_frame=3|
|------|---|---|--|
|37초|7.46초|4.54초|4.19초|
|41초|8.61초|6.11초|5.초|
|83초|13.58초|10.96초|8.99초|
|32초|7.94초|4.98초|4.58초|
|396초|121초|83.57초|70.78초|

frame_skip을 한번하면 속도가 2배는 아니지만 크게 향상된다는 것을 보실 수 있습니다.


# License

I refered source code of "https://github.com/jkjung-avt/tensorrt_demos" and change trt_yolo.py to custom_yolo.py 