"""Checkout audit accepts LF/CRLF/mixed files, but rejects substantive edits."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hygiene_under_test',REPO/'scripts/check_claim_hygiene.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class CheckoutAuditTests(unittest.TestCase):
    def check_clone(self, autocrlf, *, mutate=False, mixed=False, mutate_frozen=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'checkout'
            subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(root)],check=True,capture_output=True)
            subprocess.run(['git','config','core.autocrlf',autocrlf],cwd=root,check=True)
            subprocess.run(['git','checkout','--quiet','HEAD'],cwd=root,check=True,capture_output=True)
            path=root/'data/load_test_telemetry.json'
            raw=path.read_bytes()
            if autocrlf=='true':self.assertIn(b'\r\n',raw)
            else:self.assertNotIn(b'\r\n',raw)
            if mixed:
                # Git can retain existing LF files beside newer CRLF checkouts.
                for name in ('data/baseline_llm_measurements.json','data/load_test_telemetry.json','docs/archive/README.before_claim_hygiene.md','docs/STATUS.md'):
                    file=root/name
                    file.write_bytes(file.read_bytes().replace(b'\r\n',b'\n'))
            if mutate_frozen:
                file=root/'data/baseline_llm_measurements.json'
                file.write_bytes(file.read_bytes().replace(b'{',b'{"unexpected_field":true,',1))
            if mutate:
                data=path.read_bytes()
                self.assertIn(b'query_text',data)
                path.write_bytes(data.replace(b'query_text',b'query_EDIT',1))
            old=audit.ROOT
            try:
                audit.ROOT=root
                if mutate or mutate_frozen:
                    with self.assertRaisesRegex(AssertionError,'historical telemetry changed|frozen runtime/data/artifact changed'):
                        audit.verify()
                else:
                    result=audit.verify()
                    self.assertTrue(result['runtime_data_r5_artifacts_unchanged'])
                    self.assertEqual([20,1,1],[r['auto_reuse_n'] for r in result['timing_recomputations']])
            finally:audit.ROOT=old

    def test_lf_checkout(self):self.check_clone('false')
    def test_crlf_checkout(self):self.check_clone('true')
    def test_crlf_content_edit_rejected(self):self.check_clone('true',mutate=True)
    def test_mixed_checkout(self):self.check_clone('true',mixed=True)
    def test_mixed_frozen_edit_rejected(self):self.check_clone('true',mixed=True,mutate_frozen=True)

if __name__=='__main__':unittest.main()
