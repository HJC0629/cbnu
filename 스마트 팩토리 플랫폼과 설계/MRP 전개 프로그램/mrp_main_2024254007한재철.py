import pandas as pd

# 엑셀 파일 경로 및 시트 이름
file_path = 'MRP_입력정보_재철.xlsx'
mps_sheet = 'MPS'
irf_sheet = 'IRF'
bom_sheet = 'BOM'
mrp_output_sheet = 'MRP출력'

# MPS 및 IRF 데이터 읽기
mps_df = pd.read_excel(file_path, sheet_name=mps_sheet)
irf_df = pd.read_excel(file_path, sheet_name=irf_sheet)
bom_df = pd.read_excel(file_path, sheet_name=bom_sheet)

# 주차 범위 설정 (1주차부터 17주차)
weeks = list(range(1, 18))

# MRP 출력 테이블 생성 및 초기화: 모든 값을 0으로 설정
columns = ['총소요량', '예정입고', '예상재고', '순소요량', '계획수주', '계획발주']
mrp_results = pd.DataFrame(0, index=weeks, columns=pd.MultiIndex.from_product([['A', 'B', 'C', 'D'], columns]))


# 품목별 재고 데이터를 dict로 정리
item_info = {}
for index, row in irf_df.iterrows():
    품목코드 = row['품목코드']
    인도기간 = row['인도기간']
    안전재고 = row['안전재고']
    주문량 = row['주문량']

    # 품목코드별로 정보 저장
    item_info[품목코드] = {
        '인도기간': 인도기간,
        '안전재고': 안전재고,
        '주문량': 주문량 #최소주문량
    }

#print(item_info)
# 초기화 함수: 재고 및 예정 입고 세팅
def init_inventory(irf_df, mrp_results):
    for index, row in irf_df.iterrows():
        item_code = row['품목코드']
        current_stock = row['현재재고'] - row['안전재고']  # 현재고에서 안전재고 차감

        # 주차별 초기 예상재고 설정
        for week in weeks:
            mrp_results.loc[week, (item_code, '예상재고')] = current_stock

        # 예정 입고량을 해당 주차에 반영
        if not pd.isna(row['예정입고일']) and not pd.isna(row['예정입고량']):
            in_stock_week = int(row['예정입고일'])
            in_stock_amount = row['예정입고량']

            if in_stock_week in weeks:  # 예정 입고 주차가 유효한지 확인
                mrp_results.loc[in_stock_week, (item_code, '예정입고')] = in_stock_amount

                # 예상재고에 예정 입고량 반영
                for week in weeks:
                    if week >= in_stock_week:  # 예정입고일 이후 주차에 대해서만 반영
                        mrp_results.loc[week, (item_code, '예상재고')] += in_stock_amount

# MPS 로 각 품목별 수요량 세팅
def init_mrp(mps_df, mrp_results):
    for index, row in mps_df.iterrows():
        item_code = row['품목코드']
        order_amount = row['수량']
        due_week = row['납기']

        # 납기 주차에 수요량 설정
        mrp_results.loc[due_week, (item_code, '총소요량')] = order_amount


def calculate_bom_requirements(bom_df, parent_code,amount):
    bom_info = []  # BOM 정보를 저장할 리스트 초기화

    # 부모 코드에 해당하는 BOM 항목 필터링
    bom_items = bom_df[bom_df['Parent'] == parent_code]

    for _, row in bom_items.iterrows():
        child_item = row['Child']
        qty = int(row['Qty'] * amount)  # np.int64에서 int로 변환
        # 하위 품목 정보를 딕셔너리 형태로 저장
        bom_info.append({
            'child_code': child_item,
            'qty': qty
        })

    return bom_info


def run_mrp(item_info, mrp_results, bom_df):
    for item_code in item_info.keys():
        for week in weeks:
            due_week = week  # 납기
            pre_week = week - 1  # 지난주
            if pre_week <= 0:
                pre_week = 1

            order_requirements = mrp_results.loc[week, (item_code, "총소요량")]  # 품목별 총소요량

            pre_stock = mrp_results.loc[pre_week, (item_code, "예상재고")]  # 지난주재고, 첫주차일 경우 현재고

            # 총소요량이 지난주의 예상재고보다 클경우에만 계산
            if order_requirements > pre_stock:
                duration = item_info[item_code]["인도기간"]
                order_amount_base = item_info[item_code]["주문량"]  # 최소주문단위

                mrp_results.loc[week, (item_code, "순소요량")] = order_requirements - pre_stock  # 총소요량 - 재고

                plan_order_amount = order_requirements - pre_stock  # 계획수주
               # if item_code =="C":
                   # print(plan_order_amount)
                   # print(plan_order_amount % order_amount_base)
                if plan_order_amount % order_amount_base != 0:  # 계획수주량이 최소주문단위로 나누어떨어지지 않을 때
                    if plan_order_amount /order_amount_base  <= 1:  # 1보다 작으면 최소주문단위만큼 진행
                        plan_order_amount = order_amount_base
                    else:
                        tmp_multiply = plan_order_amount // order_amount_base
                        plan_order_amount = order_amount_base * tmp_multiply

                mrp_results.loc[week, (item_code, "계획수주")] = plan_order_amount
                mrp_results.loc[week - duration, (item_code, "계획발주")] = plan_order_amount

                # 하위 BOM 요구 사항 계산
                bom_requirements = calculate_bom_requirements(bom_df, item_code, plan_order_amount)
                for tmp_bom_requirement in bom_requirements:
                    child_code = tmp_bom_requirement['child_code']
                    bom_duration = item_info[child_code]["인도기간"]  # 하위 품목 인도기간
                    # 하위 품목의 총소요량을 업데이트
                    mrp_results.loc[week-duration, (child_code, "총소요량")] += tmp_bom_requirement['qty']
                    #mrp_results.loc[week-duration, (child_code, "순소요량")] =  mrp_results.loc[week-duration, (child_code, "총소요량")] - mrp_results.loc[week-duration-1, (child_code, "예상재고")]

                # 마지막에 예상재고 계산
                calculate_stock = mrp_results.loc[week, (item_code, "계획수주")] - order_requirements + pre_stock
                mrp_results.loc[week, (item_code, "예상재고")] = calculate_stock
            elif order_requirements >0 :
                #총소요량은 0보다 큰데 재고가 넘칠경우 수주는 안해도됨, 기존재고에서 재고수량만 조정하기
                duration = item_info[item_code]["인도기간"]

                mrp_results.loc[week, (item_code, "순소요량")] = 0 # 수주량이 0이라서 0임

                # 하위 BOM 요구 사항 계산
                bom_requirements = calculate_bom_requirements(bom_df, item_code, plan_order_amount)
                for tmp_bom_requirement in bom_requirements:
                    child_code = tmp_bom_requirement['child_code']
                    bom_duration = item_info[child_code]["인도기간"]  # 하위 품목 인도기간
                    # 하위 품목의 총소요량을 업데이트
                    mrp_results.loc[week - duration, (child_code, "총소요량")] += tmp_bom_requirement['qty']
                    # mrp_results.loc[week-duration, (child_code, "순소요량")] =  mrp_results.loc[week-duration, (child_code, "총소요량")] - mrp_results.loc[week-duration-1, (child_code, "예상재고")]

                # 마지막에 예상재고 계산
                calculate_stock = mrp_results.loc[week, (item_code, "계획수주")] - order_requirements + pre_stock
                mrp_results.loc[week, (item_code, "예상재고")] = calculate_stock
            else:
                # 총소요량이 0인 경우, 순소요량 및 계획수주를 0으로 설정
                mrp_results.loc[week, (item_code, "순소요량")] = 0
                mrp_results.loc[week, (item_code, "계획수주")] = 0
                mrp_results.loc[week, (item_code, "예상재고")] = pre_stock
                if mrp_results.loc[week, (item_code, "예정입고")] > 0:
                    mrp_results.loc[week, (item_code, "예상재고")] = pre_stock + mrp_results.loc[week, (item_code, "예정입고")]



# IRF 데이터와 MPS 계산
init_inventory(irf_df, mrp_results)
init_mrp(mps_df, mrp_results)

# MRP 전개 실행
run_mrp(item_info,mrp_results,bom_df)


# MRP 결과를 엑셀로 저장
mrp_results.to_excel('MRP_결과.xlsx', index=True, header=True)
