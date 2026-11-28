"""Concise console progress; retain every original event and value in JSONL."""
import json
import time

LABELS = {'key':'열쇠','door':'문','exit':'출구'}
DETAIL = {'시작','접근 후보 검사','이동 가능한 접근 후보 선택','기존 지도 관측 후보 검사',
          '목표까지의 경로를 확인하는 중이야.','도착 후 카메라와 거리를 다시 확인할게.',
          '물체 도착 재확인','출발 위치로 복귀 중'}
MILESTONES = {
    '열쇠 삭제 요청이 승인됐어. 카메라로 사라졌는지 확인할게.':'[획득] 열쇠가 사라졌는지 영상 확인 중',
    '열쇠 획득 완료. 카메라로 빨간 문을 찾을게.':'[획득 완료] 문 탐색 시작',
    '열쇠를 사용해 문 열기를 요청했어.':'[문] 열쇠를 사용해 문 열기 요청',
    '문 열림 확인 완료. 열린 문을 통과할게.':'[문 열림] 통과 준비 중',
    '라이다와 Nav2 충돌 검사를 유지하며 열린 문을 통과할게.':'[문 통과] 이동 중',
    '문 통과 확인 완료. 카메라로 초록색 exit_marker를 찾을게.':'[문 통과 완료] 출구 탐색 시작',
    'exit_marker 도착 확인 완료!':'[완료] 출구 도착 확인',
    '미로 탈출 임무 완료.':'[완료] 미로 탈출 임무 성공',
    '사용자가 카메라 임무를 멈췄어.':'[종료] 사용자가 임무 중지',
}


class ConsolePresenter:
    def __init__(self):
        self.last = {}

    def present(self, message, values, phase, wall, terminal_code=0):
        if terminal_code:
            return 'error',message
        if message in DETAIL:
            return None
        level,period = 'info',5
        label = LABELS.get(values.get('label'),values.get('label','목표물'))
        if message=='이동 확인':
            distance = values.get('distance')
            text = '[이동] 목표까지 '+format(distance,'.2f')+'m' if isinstance(distance,(int,float)) else '[이동] 이동 중'
            period = 15
            key = message,phase
        else:
            text = MILESTONES.get(message,message)
            if message.startswith('카메라 임무 시작:'):
                text = '[시작] 열쇠 탐색 → 문 열기 → 출구 이동'
            elif message=='카메라에서 목표를 확인했어.':
                text = '[발견] '+label+' 인식'
            elif message=='카메라로 확인한 물체 가까이 이동할게.':
                text = '[접근] '+label+' 근처로 이동'
            elif message=='카메라 재탐색을 시도할게.':
                text = '[재인식] '+label+'을 다시 찾는 중 ('+str(values.get('attempt',1))+'/3)'
            elif message=='카메라 관측 지점으로 재방문할게.':
                if values.get('source')=='memory':
                    return None
                text = '[재탐색] '+label+' 관측 위치로 이동 ('+str(values.get('used','?'))+'/'+str(values.get('limit','?'))+')'
            elif message=='이전에 카메라로 본 물체의 관측 위치를 우선 재확인할게.':
                text = '[기억] 이전에 '+label+'을 본 위치로 이동'
            elif message=='카메라로 주변을 둘러볼게.':
                if values.get('purpose')=='reacquire':
                    return None
                text = '[관측] 주변 확인 중'
            elif message=='접근 후보 실패. 다른 위치를 검사할게.':
                level,text = 'warning','[재시도] '+label+'의 다른 접근 위치 검사'
            elif message=='관측 지점 재확인 결과':
                level = 'info' if values.get('success') else 'warning'
                text = '[재탐색] 관측 '+('완료' if values.get('success') else '실패')+' · '+label+' 탐색 계속'
            elif message=='카메라 준비를 기다리는 중이야.':
                missing = [name for key,name in (('rgb','컬러 영상'),('depth','깊이 영상'),('info','보정 정보'))
                           if not values.get(key)]
                text = '[준비] 카메라 대기'+(' · '+', '.join(missing) if missing else ' · 좌표 연결 확인 중')
                period = 30
            elif '기다리는 중이야' in message:
                period = 30
            key = message,phase,text
        previous = self.last.get(key)
        self.last[key] = wall if previous is None or wall-previous>=period else previous
        if previous is not None and wall-previous<period:
            return None
        return level,text


class ConsoleLogging:
    def emit(self, message, **values):
        # Keep the same detailed record structure used by the original Mission.
        self.log.write(json.dumps({'wall':time.time(),'sim':self.sim(),'state':self.state.phase,
                                   'message':message,**values},ensure_ascii=False)+'\n')
        self.log.flush()
        if not hasattr(self,'_console_presenter'):
            self._console_presenter = ConsolePresenter()
        output = self._console_presenter.present(message,values,self.state.phase,time.monotonic(),
                                                getattr(self,'_console_terminal_code',0))
        if output is not None:
            level,text = output
            getattr(self.get_logger(),level)(text)

    def stop(self, message, code):
        self._console_terminal_code = code
        super().stop(message,code)
