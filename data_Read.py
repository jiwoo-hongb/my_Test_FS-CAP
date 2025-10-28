# bindingDB 데이터 확인용 코드 입니다.
# import pandas as pd

# file = "BindingDB_All_202211.tsv"

# # 헤더만 읽기
# header = pd.read_csv(file, sep="\t", nrows=0)
# print("총 컬럼 수:", len(header.columns))
# for i, col in enumerate(header.columns, 1):
#     print(f"{i:02d}. {col}")
#=============================================================

# bindingDB 2022 vs 2025 비교 코드 입니다.
# import pandas as pd

# # 🔹 파일 경로 설정 (파일명은 실제 위치에 맞게 수정)
# file_2022 = "BindingDB_All_202211.tsv"
# file_2025 = "BindingDB_All_202510.tsv"

# # 🔹 헤더만 읽기 (속도 빠름)
# cols_2022 = pd.read_csv(file_2022, sep="\t", nrows=0).columns
# cols_2025 = pd.read_csv(file_2025, sep="\t", nrows=0).columns

# # 🔹 set 변환
# set_2022 = set(cols_2022)
# set_2025 = set(cols_2025)

# # 🔹 비교
# common_cols = sorted(list(set_2022 & set_2025))
# added_cols = sorted(list(set_2025 - set_2022))
# removed_cols = sorted(list(set_2022 - set_2025))

# # 🔹 출력 포맷 함수
# def print_section(title, data):
#     print("\n" + "="*100)
#     print(f"🟦 {title} ({len(data)}개)")
#     print("="*100)
#     if len(data) == 0:
#         print("(없음)")
#     else:
#         for i, col in enumerate(data, 1):
#             print(f"{i:03d}. {col}")

# # 🔹 콘솔 출력
# print_section("✅ 공통 컬럼 (두 버전에 모두 존재)", common_cols)
# print_section("🆕 신규 추가 컬럼 (2025 버전에만 존재)", added_cols)
# print_section("❌ 제거된 컬럼 (2022 버전에만 존재)", removed_cols)

# # 🔹 간단 요약
# print("\n" + "="*100)
# print("📊 요약")
# print("="*100)
# print(f"공통 컬럼 수: {len(common_cols)}")
# print(f"신규 컬럼 수 (2025 전용): {len(added_cols)}")
# print(f"제거 컬럼 수 (2022 전용): {len(removed_cols)}")
# ============================================================

# pickle 파일 구조 확인
import pickle

with open("bindingdb_data.pickle", "rb") as f:
    data = pickle.load(f)

# 객체 타입 확인
print(type(data))

# 예를 들어 data가 튜플이라면:
for i, part in enumerate(data):
    print(i, type(part), 
          # 만약 배열이나 리스트라면 길이나 shape 출력
          getattr(part, "shape", None), getattr(part, "__len__", None))
