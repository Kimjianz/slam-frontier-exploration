import io
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock
from maze_escape.console_logging import ConsoleLogging,ConsolePresenter


class ConsoleTests(unittest.TestCase):
    def test_coordinates_and_candidate_details_are_not_printed(self):
        presenter = ConsolePresenter()
        self.assertIsNone(presenter.present('접근 후보 검사',{'usable':2},'SELECT',100))
        output = presenter.present('카메라에서 목표를 확인했어.',
                                   {'label':'key','position':[1,2,3]},'SELECT',100)
        self.assertEqual(output,('info','[발견] 열쇠 인식'))

    def test_progress_is_throttled_but_still_updates(self):
        presenter = ConsolePresenter()
        self.assertIsNotNone(presenter.present('이동 확인',{'distance':1},'NAVIGATING',100))
        self.assertIsNone(presenter.present('이동 확인',{'distance':.9},'NAVIGATING',103))
        self.assertIsNotNone(presenter.present('이동 확인',{'distance':.8},'NAVIGATING',115))

    def test_terminal_error_is_never_suppressed_as_a_duplicate_or_detail(self):
        presenter = ConsolePresenter()
        self.assertEqual(presenter.present('접근 후보 검사',{},'SELECT',100,2),('error','접근 후보 검사'))
        self.assertEqual(presenter.present('접근 후보 검사',{},'SELECT',101,2),('error','접근 후보 검사'))

    def test_retries_are_visible_as_warnings(self):
        output = ConsolePresenter().present('접근 후보 실패. 다른 위치를 검사할게.',
                                          {'label':'door'},'SELECT',100)
        self.assertEqual(output[0],'warning')
        self.assertIn('문',output[1])

    def test_full_events_are_retained_in_file_even_when_console_is_quiet(self):
        class Base:
            def stop(self,message,code):
                self.emit(message)
                self.done = True
        class Recorder(ConsoleLogging,Base):
            pass
        node = Recorder()
        node.log = io.StringIO()
        node.sim = lambda:10
        node.state = NS(phase='SELECT')
        logger = MagicMock()
        node.get_logger = lambda:logger
        node.emit('접근 후보 검사',usable=2,position=[1,2,3])
        logger.info.assert_not_called()
        node.stop('접근 실패 원인',2)
        logger.error.assert_called_once_with('접근 실패 원인')
        records = [json.loads(line) for line in node.log.getvalue().splitlines()]
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]['position'],[1,2,3])
        self.assertEqual(records[1]['message'],'접근 실패 원인')


if __name__=='__main__':
    unittest.main()
