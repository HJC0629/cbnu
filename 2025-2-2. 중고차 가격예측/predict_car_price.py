import tensorflow as tf
import pandas as pd
import joblib
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'car_price_model.h5')
preprocessor_path = os.path.join(base_dir, 'car_preprocessor.pkl')


if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    print("[Error] 모델이나 전처리기 파일이 없습니다.")
    print("먼저 car_price_prediction.py를 실행해서 학습을 완료해주세요.")
    exit()


model = tf.keras.models.load_model(model_path, compile=False)
preprocessor = joblib.load(preprocessor_path)
print(">> 차량 정보를 입력해주세요.")
print("=" * 40)



def get_user_input():
    try:
        print("\n[1] 차량 연식 (예: 2015)")
        year = int(input(">>> 입력: "))

        print("\n[2] 신차 구매 당시 가격 (단위: 만원)")
        print("(예: 3000만 원이면 '3000' 입력, 1억 5천이면 '15000' 입력)")
        price_krw = float(input(">>> 입력: "))

        # [핵심] 만원 -> Lakhs 변환 (AI가 이해하는 단위로 변경)
        # 1 Lakh ≈ 160만 원
        present_price_lakhs = price_krw / 160

        print("\n[3] 주행 거리 (km, 예: 50000)")
        kms = int(input(">>> 입력: "))

        print("\n[4] 연료 종류 (Petrol / Diesel / CNG 중 하나 입력)")
        print("(휘발유: Petrol, 경유: Diesel, 가스: CNG)")
        fuel = input(">>> 입력: ")

        print("\n[5] 판매자 유형 (Dealer / Individual 중 하나 입력)")
        print("(매매상사: Dealer, 개인거래: Individual)")
        seller = input(">>> 입력: ")

        print("\n[6] 변속기 (Manual / Automatic 중 하나 입력)")
        print("(수동: Manual, 오토: Automatic)")
        trans = input(">>> 입력: ")

        print("\n[7] 이전 소유자 수 (0이면 1인신조, 예: 0)")
        owner = int(input(">>> 입력: "))

        data = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price_lakhs],
            'Kms_Driven': [kms],
            'Fuel_Type': [fuel],
            'Seller_Type': [seller],
            'Transmission': [trans],
            'Owner': [owner]
        })
        return data, price_krw

    except ValueError:
        print("\n[Error] 숫자를 입력해야 하는 곳에 문자를 입력했습니다.")
        return None, None



input_df, original_price_krw = get_user_input()

if input_df is not None:
    try:
        # 전처리
        input_processed = preprocessor.transform(input_df)

        # 예측 (결과는 Lakhs 단위로 나옴)
        pred_lakhs = model.predict(input_processed)[0][0]

        # [핵심] 결과 변환: Lakhs -> 만원
        pred_krw = pred_lakhs * 160


        depreciation = ((original_price_krw - pred_krw) / original_price_krw) * 100

        print("\n" + "=" * 40)
        print(f" AI 중고차 가격 분석 결과")
        print("=" * 40)
        print(f"신차 가격      : {original_price_krw:.0f} 만 원")
        print(f"예상 중고 가격 : {pred_krw:.0f} 만 원")
        print(f"예상 감가율    : -{depreciation:.1f}%")
        print("=" * 40)

    except Exception as e:
        print(f"\n[Error] 입력값이 데이터셋 형식과 맞지 않습니다.")
        print(f"철자(Petrol, Dealer 등)를 정확히 입력했는지 확인해주세요.")
        print(f"에러 내용: {e}")