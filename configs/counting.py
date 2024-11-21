import json
from collections import Counter

# COCO annotations 파일 경로 (예: train2017)
annotations_path = "/home/sumin/DINOD_MODEL/dataset/COCO/annotations/instances_val2017.json"

# JSON 파일 로드
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# 각 객체의 image_id를 리스트로 추출
image_ids = [ann['image_id'] for ann in coco_data['annotations']]

# 이미지별 객체 개수 계산
image_object_count = Counter(image_ids)

# 가장 많이 포함된 객체 개수
max_objects = max(image_object_count.values())

print(f"COCO 데이터셋에서 한 이미지에 포함된 객체의 최대 개수: {max_objects}")
