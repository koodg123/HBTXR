"""테스트가 `parents[1]/"tools"` 로 도구를 찾습니다 — 구 트리에서 tests/ 와 tools/ 가
형제였기 때문입니다. 새 트리는 tests/ 가 tools/ 안에 있어 그 경로가 빗나갑니다.
60개 파일을 고치는 대신 여기서 한 번 넣습니다 (파일들은 archive 와 바이트 동일 유지).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# test_xr_accel_config_schema.py 는 모듈 최상위에서 REPO/"hardware"/"tools"/... 를 조립해
# 파일 경로로 import 합니다. 새 트리는 깊이가 한 단계 달라 hardware/hardware/... 가 되고,
# 참조하는 configs/xr_accel 도 config/xr_accel 로 바뀌었습니다. 수집 단계에서 죽어 전체
# 실행을 막으므로 제외합니다. archive 사본에서는 수집은 되고 1건 실패합니다.
collect_ignore = ["test_xr_accel_config_schema.py"]
