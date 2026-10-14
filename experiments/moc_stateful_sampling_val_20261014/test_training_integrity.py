import unittest
from training_integrity import committed_labels,inspect_calls

class IntegrityTests(unittest.TestCase):
    def test_uncommitted_and_invalid_question_labels_are_excluded(self):
        labels=[{"question_id":q,"label":1} for q in ("valid","unfinished","limited")]
        questions=[{"question_id":"valid","status":"ok"},{"question_id":"limited","status":"invalid_output_limit"}]
        self.assertEqual(committed_labels(labels,questions),[labels[0]])

    def test_limited_call_is_flagged_without_losing_its_cost(self):
        calls=[{"event_type":"llm_call","request":{"model":"m"},"response":{"done":True,
                 "eval_count":2000,"prompt_eval_count":100,"done_reason":"length"}}]
        _,limited,usage=inspect_calls(calls,"m")
        self.assertEqual(len(limited),1)
        self.assertEqual(usage,{"main_calls":1,"prompt_tokens":100,"completion_tokens":2000})

    def test_missing_or_wrong_backend_response_fails_closed(self):
        with self.assertRaises(AssertionError):
            inspect_calls([],"m")
        with self.assertRaises(AssertionError):
            inspect_calls([{"event_type":"llm_call","request":{"model":"other"},"response":{}}],"m")

if __name__=="__main__":
    unittest.main()
