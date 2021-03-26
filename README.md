# RioOCR
this OCR is a by-product of my own research
文字列は単に検出した文字を座標ごとにソートして分割しただけのものとなります

現在識別可能な文字は68文字：['&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'e', 'E', 'F', 'L', 'S', 'ア', 'ウ', 'エ', 'ク', 'シ', 'ス', 'セ', 'ソ', 'タ', 'テ', 'ト', 'フ', 'ム', 'ョ', 'ル', 'ワ', 'ン', '英', '画', '階', '学', '究', '共', '研', '験', '庫', '工', '合', '鎖', '事', '室', '実', '生', '倉', '像', '段', '知', '長', '糖', '同', '能', '品', '富', '部', '務', '命', '薬', '理', 'ー（伸ばし棒）']

完ぺきではないため商業利用不可

学習済みモデルは大きすぎるため、必要があれば共有いたします。

# Example

![image](https://user-images.githubusercontent.com/56717608/112598439-ec20cd80-8e51-11eb-92a7-774e22112092.png)
![image](https://user-images.githubusercontent.com/56717608/112598454-f04ceb00-8e51-11eb-912a-2030b7a57a66.png)

# How to use
download every thing and get trained model from URLs
and then put them in same folder 

after this 
change program line
https://github.com/sryuu/RioOCR/blob/cd3c42cedafac9bc1becd176bf5873bde63a178a/main.py#L112
put your image path here
and input number to cut image for read line
