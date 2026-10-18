"""Checkout-aware audit accepts Git CRLF conversion but still rejects edits."""
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
    def check_clone(self, autocrlf, *, mutate=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'checkout'
            subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(root)],check=True,capture_output=True)
            subprocess.run(['git','config','core.autocrlf',autocrlf],cwd=root,check=True)
            subprocess.run(['git','checkout','--quiet','HEAD'],cwd=root,check=True,capture_output=True)
            path=root/'data/load_test_telemetry.json'
            raw=path.read_bytes()
            if autocrlf=='true':self.assertIn(b'\r\n',raw)
            else:self.assertNotIn(b'\r\n',raw)
            if mutate:
                # A substantive text change must not disappear in EOL handling.
                self.assertIn(b'query_text',raw)
                path.write_bytes(raw.replace(b'query_text',b'query_EDIT',1))
            old=audit.ROOT
            try:
                audit.ROOT=root
                if mutate:
                    with self.assertRaisesRegex(AssertionError,'historical telemetry changed'):audit.verify()
                else:
                    result=audit.verify()
                    self.assertTrue(result['runtime_data_r5_artifacts_unchanged'])
                    self.assertEqual([20,1,1],[r['auto_reuse_n'] for r in result['timing_recomputations']])
            finally:audit.ROOT=old

    def test_lf_checkout(self):self.check_clone('false')
    def test_crlf_checkout(self):self.check_clone('true')
    def test_crlf_content_edit_rejected(self):self.check_clone('true',mutate=True)

if __name__=='__main__':unittest.main()
