from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_restful import reqparse
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import json

app = Flask(__name__)

@app.route('/')
def home():
    return redirect(url_for('initial_screen'))

@app.route('/home')
def initial_screen():
    return render_template('initial_screen.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/api/document', methods=['GET'])
def document():
    import json
    from collections import OrderedDict
    args = request.args
    query = args.get('query')
    
    doc = OrderedDict()
    
    if query == 'nature_language':
        lang = 'Python'
        color = 'python'
        github = 'https://github.com/seokmin12/nature_language'
        
        doc['api'] = 'https://api.seokmin.kro.kr/nature_language?query={news-title}'
        doc['parameter'] = ['query']
        doc['framework'] = ['Tensorflow', 'KoNLPY']
        doc['type'] = ['String']
        doc['method'] = ['GET']
        doc['Required'] = ['O']
        doc['description'] = ['뉴스 기사의 제목을 입력합니다.']
        doc['response'] = ['"result": "긍정적인 기사입니다."']
    if query == 'stock_predict':
        lang = 'Python'
        color = 'python'
        github = 'https://github.com/seokmin12/stock-predict'
        
        doc['api'] = 'https://api.seokmin.kro.kr/stock_predict?symbol={symbol}'
        doc['parameter'] = ['symbol']
        doc['framework'] = ['Tensorflow']
        doc['type'] = ['String']
        doc['method'] = ['GET']
        doc['Required'] = ['O']
        doc['description'] = ['예측하고 싶은 종목 코드를 입력합니다.']
        doc['response'] = ['"result": "68900원"']
    if query == 'face_recognition':
        lang = 'Python'
        color = 'python'
        github = 'https://github.com/seokmin12/face-recognition'
        
        doc['api'] = 'https://api.seokmin.kro.kr/face_recognition'
        doc['parameter'] = ['Face']
        doc['framework'] = ['OpenCV']
        doc['type'] = ['Image / Form-Data']
        doc['method'] = ['POST']
        doc['Required'] = ['O']
        doc['description'] = ['얼굴만 추출하고 싶은 사진을 업로드합니다.']
        doc['response'] = ['"result": "(Encoded String)"']

    if query == 'ocr':
        lang = 'Node js'
        color = 'nodejs'
        github = 'https://github.com/seokmin12/OCR'
        doc['api'] = 'https://api.seokmin.kro.kr/ocr'
        doc['parameter'] = ['uploadfile', 'lang']
        doc['framework'] = ['Tesseract']
        doc['type'] = ['Image / Form-data', 'String']
        doc['method'] = ['POST', 'POST']
        doc['Required'] = ['O', 'O']
        doc['description'] = ['텍스트를 추출하고 싶은 사진을 업로드합니다.', '텍스트의 언어를 입력합니다.']
        doc['response'] = ['"result": "안녕하세요"']

        json.dumps(doc, ensure_ascii=False, indent="\t")
    return render_template('document.html', name=query, lang=lang, color=color, github=github, doc=doc)

@app.route('/main')
def main():
    import requests
    from bs4 import BeautifulSoup
    await def crawling():
        symbol_list = ['005930', '000660', '035420', '035720']
        stock_list = []
        for i in symbol_list:
            url = 'https://finance.naver.com/item/main.naver?code=' + i

            response = requests.get(url)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            now_price = soup.select_one('#chart_area > div.rate_info > div > p.no_today > em > span.blind').text
            percent = soup.select('div > p.no_exday > em > span.blind')[1].text

            PlusMinus = soup.select('div > p.no_exday > em > span.ico')[0].text
            
            stock_list.append(now_price)
            if PlusMinus == '보합':
                stock_list.append(percent + '%')
                stock_list.append('No change than yesterday')
                stock_list.append('#000000')
            if PlusMinus == '하락':
                stock_list.append('-' + percent + '%')
                stock_list.append('Less price than yesterday')
                stock_list.append('#0059d1')
            if PlusMinus == '상승':
                stock_list.append('+' + percent + '%')
                stock_list.append('More price than yesterday')
                stock_list.append('#dc3545')

        samsung = stock_list[0:4]
        sk = stock_list[4:8]
        naver = stock_list[8:12]
        kakao = stock_list[12:16]
        return samsung, sk, naver, kakao
    
    samsung, sk, naver, kakao = crawling()

        
    return render_template('api.html', samsung=samsung, sk=sk, naver=naver, kakao=kakao)

@app.route('/api/machine_learning/nature_language', methods=['POST'])
def sentiment_predict():
    from konlpy.tag import Okt
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    import numpy as np
    import re
    import os
    import pickle
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    okt = Okt()

    max_len = 20
    max_words = 35000

    # load model
    model = load_model('/Users/seokmin/Desktop/python/Flask/news_model.h5')

    # loading tokenizer
    with open('/Users/seokmin/Desktop/python/Flask/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    def sentiment_predict(new_sentence):
        clean_sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》·]', '', new_sentence)
        clean_sentence = okt.morphs(clean_sentence, stem=True)  # 토큰화
        clean_sentence = [word for word in clean_sentence if not word in stopwords]  # 불용어 제거
        sequences = tokenizer.texts_to_sequences([clean_sentence])  # 정수 인코딩
        pad_new = pad_sequences(sequences, maxlen=max_len)  # 패딩
        score = model.predict(pad_new)
        predict_score = np.argmax(score)
        if predict_score == 0:
            return "부정적인 기사입니다."
        elif predict_score == 1:
            return "중립적인 기사입니다."
        elif predict_score == 2:
            return "긍정적인 기사입니다."

    return redirect(url_for('main', sentiment_predict=sentiment_predict(request.form['news']), news_title=request.form['news']))

@app.route('/api/machine_learning/stock_predict', methods=['POST'])
def stock_predict():
    from tensorflow.keras.models import load_model
    import FinanceDataReader as fdr
    import os
    import pandas as pd
    import numpy as np
    
    symbol = request.form['stock_symbol']
    
    def get_regression(data):
        x_data = list(range(6000, 6000 + 10 * len(data), 10))
        y_data = data

        # X, Y의 평균을 구합니다.
        x_bar = sum(x_data) / len(x_data)
        y_bar = sum(y_data) / len(y_data)

        # 최소제곱법으로 a, b를 구합니다.
        a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(y_data, x_data))])
        a /= sum([(x - x_bar) ** 2 for x in x_data])
        b = y_bar - a * x_bar

        line_x = np.arange(min(x_data), max(x_data), 0.01)
        line_y = a * line_x + b
        if (line_y[-1] - line_y[0]) > 0:
            return 1
        else:
            return 0
    
    model = load_model(
        '/Users/seokmin/Desktop/python/Flask/stock_model.h5')
    
    predict_df = fdr.DataReader(symbol)

    data = list(predict_df['Close'][-6:-1])

    predict_data_raw = {'1': data[0],
                        '2': data[1],
                        '3': data[2],
                        '4': data[3],
                        '5': data[4],
                        'regression': get_regression(data)}

    predict_data_set = pd.DataFrame(predict_data_raw, index=[0])

    prediction = model.predict(predict_data_set)
    prediction = round(prediction[0][0])
    stock_predict = str(format(prediction, ',')) + '원'
    return redirect(url_for('main', stock_predict=stock_predict, symbol=symbol))


@app.route('/api/machine_learning/face_recognition', methods=['POST'])
def face_recognition():
    import cv2
    import cvlib as cv
    import os
    
    img = request.files['face_uploadfile']
    
    app.config['UPLOAD_FOLDER'] = '/Users/seokmin/Desktop/python/Flask/static/uploadimage'
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(path)
    
    im = cv2.imread('/Users/seokmin/Desktop/python/Flask/static/uploadimage/' + img.filename)  # 이미지 읽기
    # detect faces (얼굴 검출)
    faces, confidences = cv.detect_face(im)

    for face in faces:
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]
        # draw rectangle over face
        cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 0)  # 검출된 얼굴 위에 박스 그리기
        # gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        face_crop = np.copy(im[startY:endY, startX:endX])
        cv2.imwrite(f'/Users/seokmin/Desktop/python/Flask/static/uploadimage/face_crop.jpeg', face_crop)
        
    cropped_img = cv2.imread('/Users/seokmin/Desktop/python/Flask/static/uploadimage/face_crop.jpeg')
    return redirect(url_for('main', cropped=cropped_img))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)
