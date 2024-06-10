import  os
filePath  =  './jyxstxtqj_downcc.com '

# file_inf  =  codecs.open(filePath  +  '/inf.txt',  "r",  encoding='gb23 12')
filePath = r"jyxstxtqj_downcc.com"  # 文件夹路径
fileList = os.listdir(filePath)
stop_punctuation=open('cn_punctuation.txt', encoding='utf-8')
stop_punctuation_words = stop_punctuation.read().split("\n")


#读取txt文件，去除关于网址的字符，将txt内容存入列表
corpus  =  []
for  file  in  fileList:
    path  =  os.path.join(filePath,  file)
    with  open(os.path.abspath(path),  "r",  encoding='gb18030')  as  file :       #errors="ignore"
        filecontext  =  file.read()
        filecontext  =  filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com",  '' )
        filecontext  =  filecontext.replace("本书来自www.cr173.com免费txt小说下载站",  '')
        corpus.append(filecontext)
        file.close()

#去除标点符号和英文字符，根据标点符号将中文字符按行存入语料库，得到整个语料库
fw  =  open('corpus_sentence.txt',  'w',  encoding='utf-8')
sentence  =   ''
for  filecontext  in  corpus:
    for  x  in  filecontext:
        if  len(x.encode('utf-8'))  ==  3  and  x  not  in  stop_punctuation_words:
            sentence  +=   x
        if  x  in  ['\n',  '。',  '？',  '！',  '，',  '；',  '：','．']  and  sentence  != '\n':   # 以部分中文符号为分割换行
            fw.write(sentence.strip()  + '\n')   # 按行存入语料文件
            sentence  =   ''
fw.close()