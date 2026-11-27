#!/bin/bash
# 문제 확인용: 방탈출 노드 / Nav2 / explore_lite 로그를 ~/debug.txt 하나로 모음
LOG_DIR=/tmp/escape_logs
{
    echo "=== FSM (방탈출 노드)"
    tail -n 50 "$LOG_DIR/fsm.log" 2>/dev/null
    echo
    echo "=== NAV2 ('Passing new path' 줄 제외)"
    grep -v "Passing new path" "$LOG_DIR/nav2.log" 2>/dev/null | tail -n 40
    echo
    echo "=== EXPLORE_LITE"
    tail -n 15 /tmp/explore_lite.log 2>/dev/null
} > ~/debug.txt
echo "저장 완료: ~/debug.txt  (gedit ~/debug.txt 로 열어서 전체 복사)"
